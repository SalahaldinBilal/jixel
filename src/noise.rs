/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//! Decoder-side grain synthesis: the JPEG XL noise feature (frame flag
//! `kNoise`).
//!
//! The decoder draws three planes of white noise, high-passes them with a
//! fixed 5x5 kernel and adds them to the XYB image scaled by an 8-point
//! strength LUT indexed by the pixel's intensity `(Y ± X) / 2`. The LUT is
//! the whole encoder-side parameter space: 8 values in `[0, 1)`, 10 bits
//! each, written in LfGlobal after the patch and spline dictionaries.

use crate::bit_writer::BitWriter;
use crate::image::Image3F;

/// Number of LUT points; point `i` sits at intensity `i / 6`, so the last
/// point covers intensities above 1.
pub(crate) const NUM_POINTS: usize = 8;
const PRECISION: f32 = 1024.0;
/// Largest value that still rounds to the 10-bit maximum.
const LUT_MAX: f32 = 1023.4999 / PRECISION;

/// Spec `kNoise` frame flag.
pub(crate) const FLAG_NOISE: u64 = 1;

/// Grain synthesis the decoder applies on top of a lossy VarDCT frame
/// (see [`EncodeConfig::grain`](crate::EncodeConfig::grain)).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Grain {
    /// No synthesized grain (the default).
    #[default]
    Off,
    /// Photon shot noise of a 35 mm sensor downsampled to the image size at
    /// this ISO sensitivity, libjxl's `--photon_noise_iso` model. Tiny below
    /// ISO 800 on small images; film-like from ISO 3200 up.
    PhotonIso(f32),
    /// Film-grain synthesis in AV1's manner: the fine noise the source
    /// carries at each intensity is measured on its flat areas, removed from
    /// the coded image, and re-synthesized by the decoder at this multiple of
    /// the measured level. `1.0` gives back the source's own grain, so the
    /// decoded picture keeps its texture where quantization would have
    /// smoothed it, and fewer bits go to noise; larger values stylize.
    /// Sources with no flat areas to measure on get none.
    Measured(f32),
    /// Explicit strength LUT, one value in `0..1` per intensity point
    /// `i / 6` (the last point covers intensities above 1).
    Lut([f32; NUM_POINTS]),
}

impl Grain {
    pub(crate) fn enabled(self) -> bool {
        !matches!(self, Grain::Off)
    }
}

/// The serialized noise parameters of one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NoiseParams {
    pub(crate) lut: [f32; NUM_POINTS],
}

impl NoiseParams {
    /// Bits the LfGlobal field costs.
    pub(crate) const BITS: usize = NUM_POINTS * 10;

    /// False when every point rounds to zero: the frame then drops the flag.
    pub(crate) fn has_any(&self) -> bool {
        self.lut.iter().any(|v| v.abs() > 1e-3)
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        for &v in &self.lut {
            let q = (v.clamp(0.0, LUT_MAX) * PRECISION).round() as u64;
            w.write(10, q.min(1023));
        }
    }

    /// Resolve the configured grain for the frame. Measured grain also
    /// removes the measured noise from the coded image, so the decoder's
    /// grain replaces it instead of stacking on whatever survives
    /// quantization.
    pub(crate) fn resolve(
        grain: Grain,
        ctx: &crate::encoding_context::EncodingContext,
        scratch: &mut crate::coder_scratch::CoderScratch,
        distance: f32,
        opsin: &mut Image3F,
    ) -> Option<Self> {
        let params = match grain {
            Grain::Off => return None,
            Grain::PhotonIso(iso) if iso > 0.0 => photon_noise(opsin.xsize(), opsin.ysize(), iso),
            Grain::PhotonIso(_) => return None,
            Grain::Measured(mul) if mul > 0.0 => {
                let measured = measured_grain(opsin);
                // From d=1.5 on the quantizer removes the grain itself;
                // denoising there costs quality for almost no bytes.
                let denoise_scale = ((3.0 - distance) / 1.5).clamp(0.0, 1.0);
                if measured.has_any() && denoise_scale > 0.0 {
                    denoise(ctx, scratch, opsin, &measured, denoise_scale);
                }
                let mut lut = measured.lut;
                for v in &mut lut {
                    *v = (*v * mul).clamp(0.0, LUT_MAX);
                }
                NoiseParams { lut }
            }
            Grain::Measured(_) => return None,
            Grain::Lut(lut) => NoiseParams { lut },
        };
        params.has_any().then_some(params)
    }

    /// Strength at an intensity, as the decoder interpolates it.
    fn strength_at(&self, intensity: f32) -> f32 {
        let scale = (NUM_POINTS - 2) as f32;
        let pos = (intensity * scale).max(0.0);
        if pos >= scale + 1.0 {
            return self.lut[NUM_POINTS - 1];
        }
        let i = pos.floor() as usize;
        let f = pos - i as f32;
        self.lut[i] + (self.lut[i + 1] - self.lut[i]) * f
    }
}

/// Standard deviation the decoder adds to Y per unit of LUT strength
/// (measured on flat grays through libjxl and jxl-oxide).
const Y_SIGMA_PER_UNIT: f32 = 0.494;

/// Remove the measured grain from the image to be coded, the way AV1's
/// film-grain encoders do: a sliding 8x8 DCT Wiener filter (stride 4, the
/// four overlapping estimates averaged) shrinks the higher AC coefficients
/// against the noise variance expected at the block's intensity, so the
/// grain the decoder re-synthesizes is not also spent bits on, while
/// structure above the noise floor and the lowest bands pass. `scale` fades
/// the filter out toward the distances where quantization removes the grain
/// anyway. Correlated (coarse) grain is handled as well as white noise,
/// which a small bilateral window cannot do.
fn denoise(
    ctx: &crate::encoding_context::EncodingContext,
    scratch: &mut crate::coder_scratch::CoderScratch,
    opsin: &mut Image3F,
    measured: &NoiseParams,
    scale: f32,
) {
    const N: usize = 8;
    const STRIDE: usize = 4;
    // Shrinkage threshold relative to the noise standard deviation per
    // channel; chroma noise in XYB is a fraction of the luma noise.
    const THRESH_K: [f32; 3] = [1.5, 3.0, 3.0];
    let w = opsin.xsize();
    let h = opsin.ysize();
    if w < N || h < N {
        return;
    }
    let src = opsin.clone();
    // Orthonormal DCT-II basis.
    let basis: [[f32; N]; N] = std::array::from_fn(|k| {
        std::array::from_fn(|n| {
            let s = if k == 0 {
                (1.0 / N as f32).sqrt()
            } else {
                (2.0 / N as f32).sqrt()
            };
            s * (std::f32::consts::PI * (2 * n + 1) as f32 * k as f32 / (2 * N) as f32).cos()
        })
    });
    // Window origins cover the image, the last column/row clamped inside.
    let origins = |len: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..len.saturating_sub(N)).step_by(STRIDE).collect();
        v.push(len - N);
        v
    };
    let xs = origins(w);
    let ys = origins(h);
    let mut acc = Image3F::new(w, h);
    let mut count = vec![0f32; w * h];
    let mut block = [[0f32; N]; N];
    let mut tmp = [[0f32; N]; N];
    for &y0 in &ys {
        for &x0 in &xs {
            // Noise level at the block's mean intensity.
            let mut mean = 0.0;
            for y in 0..N {
                let rx = &src.plane_row(0, y0 + y)[x0..x0 + N];
                let ry = &src.plane_row(1, y0 + y)[x0..x0 + N];
                for x in 0..N {
                    mean += 0.5 * (rx[x] + ry[x]);
                }
            }
            let sigma_y = measured.strength_at(mean / (N * N) as f32) * Y_SIGMA_PER_UNIT * scale;
            for c in 0..3 {
                let thresh = THRESH_K[c] * sigma_y;
                let t2 = thresh * thresh;
                for y in 0..N {
                    block[y].copy_from_slice(&src.plane_row(c, y0 + y)[x0..x0 + N]);
                }
                // Forward: tmp = B · block · Bᵀ.
                for v in 0..N {
                    for u in 0..N {
                        let mut s = 0.0;
                        for y in 0..N {
                            let mut r = 0.0;
                            for x in 0..N {
                                r += basis[u][x] * block[y][x];
                            }
                            s += basis[v][y] * r;
                        }
                        tmp[v][u] = s;
                    }
                }
                // Wiener-style shrinkage of the AC coefficients.
                if t2 > 0.0 {
                    for v in 0..N {
                        for u in 0..N {
                            // DC and the three lowest AC bands pass: grain
                            // lives above them, structure below.
                            if u + v < 3 {
                                continue;
                            }
                            let c2 = tmp[v][u] * tmp[v][u];
                            tmp[v][u] *= ((c2 - t2) / c2).max(0.0);
                        }
                    }
                }
                // Inverse: block = Bᵀ · tmp · B.
                for y in 0..N {
                    for x in 0..N {
                        let mut s = 0.0;
                        for v in 0..N {
                            let mut r = 0.0;
                            for u in 0..N {
                                r += basis[u][x] * tmp[v][u];
                            }
                            s += basis[v][y] * r;
                        }
                        block[y][x] = s;
                    }
                }
                for y in 0..N {
                    let row = &mut acc.plane_row_mut(c, y0 + y)[x0..x0 + N];
                    for x in 0..N {
                        row[x] += block[y][x];
                    }
                }
            }
            for y in 0..N {
                for n in &mut count[(y0 + y) * w + x0..(y0 + y) * w + x0 + N] {
                    *n += 1.0;
                }
            }
        }
    }
    crate::encode_image::for_each_linear_band(opsin, ctx, scratch, |start, band| {
        for (c, plane) in band.into_iter().enumerate() {
            for (k, v) in plane.iter_mut().enumerate() {
                let p = start + k;
                *v = acc.plane_row(c, p / w)[p % w] / count[p];
            }
        }
    });
}

/// Photon shot noise of a full-frame sensor at `iso`, as libjxl models it:
/// read noise, Poisson shot noise and photo-response non-uniformity summed
/// in quadrature per pixel, converted to an XYB-domain standard deviation
/// through the opsin cube root and normalized to the decoder's noise scale.
pub(crate) fn photon_noise(xsize: usize, ysize: usize, iso: f32) -> NoiseParams {
    // Daylight spectrum: photons per lx·s per µm².
    const PHOTONS_PER_LX_S_PER_UM2: f32 = 11260.0;
    // Effective quantum efficiency of a 2010s CFA sensor.
    const QUANTUM_EFFICIENCY: f32 = 0.20;
    const PHOTO_RESPONSE_NON_UNIFORMITY: f32 = 0.005;
    // Electrons rms.
    const READ_NOISE: f32 = 3.0;
    // 35 mm sensor in µm².
    const SENSOR_AREA_UM2: f32 = 36000.0 * 24000.0;

    let bias = crate::xyb::OPSIN_BIAS;
    let bias_cbrt = -crate::xyb::NEG_BIAS_CBRT;
    // Focal-plane exposure for 18% gray in lx·s (ISO = 10 lx·s / H).
    let h_18 = 10.0 / iso;
    let pixel_area_um2 = SENSOR_AREA_UM2 / (xsize.max(1) * ysize.max(1)) as f32;
    let electrons_18 = QUANTUM_EFFICIENCY * PHOTONS_PER_LX_S_PER_UM2 * h_18 * pixel_area_um2;

    let mut lut = [0f32; NUM_POINTS];
    for (i, out) in lut.iter_mut().enumerate() {
        // Point `i` is intensity i/6, i.e. XYB (0, y, y) with y = 2 * i/6.
        let y = 2.0 * i as f32 / (NUM_POINTS - 2) as f32;
        let linear = ((y - bias_cbrt).powi(3) + bias).max(0.0);
        let electrons = electrons_18 * (linear / 0.18);
        let noise = (READ_NOISE * READ_NOISE
            + electrons
            + (PHOTO_RESPONSE_NON_UNIFORMITY * electrons).powi(2))
        .sqrt();
        let linear_noise = noise * (0.18 / electrons_18);
        let opsin_derivative = (1.0 / 3.0) / (linear - bias).cbrt().powi(2);
        let opsin_noise = linear_noise * opsin_derivative;
        *out = (opsin_noise / DECODER_NOISE_SCALE).clamp(0.0, LUT_MAX);
    }
    NoiseParams { lut }
}

/// Standard deviation the decoder adds to Y per unit of LUT strength:
/// `norm_const` 0.22 × √2 for the red and green draws × 1.13, the standard
/// deviation of one high-passed noise plane.
const DECODER_NOISE_SCALE: f32 = 0.22 * std::f32::consts::SQRT_2 * 1.13;

/// AV1-style grain estimation. Flat 8x8 patches (by a robust rank-order
/// SAD texture score) give the source's fine noise level per intensity; the
/// LUT is fit to those samples.
pub(crate) fn measured_grain(opsin: &Image3F) -> NoiseParams {
    let samples = noise_samples(opsin);
    let mut lut = fit_lut(&samples);
    for v in &mut lut {
        *v = v.clamp(0.0, LUT_MAX);
    }
    NoiseParams { lut }
}

/// One flat patch: mean intensity `(X + Y) / 2` and the mean absolute
/// response of a 3x3 Laplacian, which for white noise of standard deviation
/// σ is about 2.2 σ.
struct NoiseSample {
    intensity: f32,
    level: f32,
}

const BLOCK: usize = 8;

/// Flat-patch detection and per-patch noise level, after libjxl's
/// `GetNoiseParameter`: the texture score of a patch is the mean of the
/// smallest half of its 3x4 sub-patch SADs against the center sub-patch;
/// the histogram mode of the scores is taken as the flat-patch level.
fn noise_samples(opsin: &Image3F) -> Vec<NoiseSample> {
    let xsize = opsin.xsize();
    let ysize = opsin.ysize();
    if xsize < BLOCK || ysize < BLOCK {
        return Vec::new();
    }
    let intensity =
        |x: usize, y: usize| 0.5 * (opsin.plane(0).row(y)[x] + opsin.plane(1).row(y)[x]);

    let bx = xsize / BLOCK;
    let by = ysize / BLOCK;
    let mut scores = Vec::with_capacity(bx * by);
    let mut hist = [0u32; 256];
    for gy in 0..by {
        for gx in 0..bx {
            let score = texture_score(gx * BLOCK, gy * BLOCK, &intensity);
            scores.push(score);
            hist[((score * 256.0) as usize).min(255)] += 1;
        }
    }
    let mode = hist
        .iter()
        .enumerate()
        .max_by_key(|&(_, &n)| n)
        .map_or(0, |(i, _)| i);
    let threshold = mode as f32 / 256.0;
    // A strong regular pattern dominates the histogram; nothing flat to
    // measure on.
    if threshold > 0.15 || threshold <= 0.0 {
        return Vec::new();
    }

    static LAPLACIAN: [[f32; 3]; 3] = [
        [-0.25, -1.0, -0.25],
        [-1.0, 5.0, -1.0],
        [-0.25, -1.0, -0.25],
    ];
    let mut samples = Vec::new();
    for gy in 0..by {
        for gx in 0..bx {
            if scores[gy * bx + gx] > threshold {
                continue;
            }
            let x0 = gx * BLOCK;
            let y0 = gy * BLOCK;
            let mut mean = 0.0;
            let mut level = 0.0;
            for y in 0..BLOCK {
                for x in 0..BLOCK {
                    mean += intensity(x0 + x, y0 + y);
                    let mut filtered = 0.0;
                    for (dy, row) in LAPLACIAN.iter().enumerate() {
                        let sy = reflect(y, dy as isize - 1);
                        for (dx, &k) in row.iter().enumerate() {
                            let sx = reflect(x, dx as isize - 1);
                            filtered += k * intensity(x0 + sx, y0 + sy);
                        }
                    }
                    level += filtered.abs();
                }
            }
            let n = (BLOCK * BLOCK) as f32;
            samples.push(NoiseSample {
                intensity: mean / n,
                level: level / n,
            });
        }
    }
    samples
}

/// Mirror a tap that leaves the patch back inside it.
fn reflect(p: usize, d: isize) -> usize {
    let q = p as isize + d;
    if (0..BLOCK as isize).contains(&q) {
        q as usize
    } else {
        (p as isize - d) as usize
    }
}

fn texture_score(x0: usize, y0: usize, intensity: &impl Fn(usize, usize) -> f32) -> f32 {
    const SW: usize = 3;
    const SH: usize = 4;
    const OFFSET: usize = 2;
    let mut sads = Vec::with_capacity((BLOCK - SW) * (BLOCK - SH));
    for yb in 0..BLOCK - SH {
        for xb in 0..BLOCK - SW {
            let mut sad = 0.0;
            for cy in 0..SH {
                for cx in 0..SW {
                    let wnd = intensity(x0 + xb + cx, y0 + yb + cy);
                    let center = intensity(x0 + OFFSET + cx, y0 + OFFSET + cy);
                    sad += (center - wnd).abs();
                }
            }
            sads.push(sad);
        }
    }
    sads.sort_by(f32::total_cmp);
    let keep = sads.len() / 2;
    sads[..keep].iter().sum::<f32>() / keep as f32
}

/// Mean |3x3 Laplacian| of the intensity `(X + Y) / 2` that one unit of LUT
/// strength produces in the decoded image (measured on flat mid-gray through
/// libjxl and jxl-oxide, which agree). Dividing a source patch's Laplacian
/// level by it gives the strength that reproduces that level.
const LAPLACIAN_LEVEL_PER_UNIT: f32 = 1.157;

/// Least-squares fit of the 8-point LUT to the samples with a small
/// smoothness prior, by Gauss-Seidel on the normal equations. The result
/// is in decoder LUT units.
fn fit_lut(samples: &[NoiseSample]) -> [f32; NUM_POINTS] {
    if samples.is_empty() {
        return [0.0; NUM_POINTS];
    }
    let scale = (NUM_POINTS - 2) as f32;
    let segment = |s: &NoiseSample| {
        let pos = (s.intensity.max(0.0) * scale).min(scale + 1.0 - 1e-4);
        (pos, (pos.floor() as usize).min(NUM_POINTS - 2))
    };
    // Noise is the floor of the patch levels: texture only adds to it. Per
    // LUT segment keep the lower half of the levels, which drops the flat
    // patches that still carry some structure.
    let mut floors = vec![Vec::new(); NUM_POINTS - 1];
    for s in samples {
        floors[segment(s).1].push(s.level);
    }
    let cutoffs: Vec<f32> = floors
        .iter_mut()
        .map(|v| {
            if v.is_empty() {
                return f32::INFINITY;
            }
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        })
        .collect();
    // Normal equations of a piecewise-linear fit: A w = b.
    let mut a = [[0f64; NUM_POINTS]; NUM_POINTS];
    let mut b = [0f64; NUM_POINTS];
    for s in samples {
        let (pos, i) = segment(s);
        if s.level > cutoffs[i] {
            continue;
        }
        let f = (pos - i as f32) as f64;
        let target = (s.level / LAPLACIAN_LEVEL_PER_UNIT) as f64;
        let w = [1.0 - f, f];
        for (di, wi) in w.iter().enumerate() {
            b[i + di] += wi * target;
            for (dj, wj) in w.iter().enumerate() {
                a[i + di][i + dj] += wi * wj;
            }
        }
    }
    // Smoothness prior: reg × n × Σ (w[i] − w[i+1])².
    let reg = 0.005 * samples.len() as f64;
    for i in 0..NUM_POINTS - 1 {
        a[i][i] += reg;
        a[i + 1][i + 1] += reg;
        a[i][i + 1] -= reg;
        a[i + 1][i] -= reg;
    }
    // Gauss-Seidel on the (positive definite) system.
    let mut w = [0f64; NUM_POINTS];
    for _ in 0..200 {
        for i in 0..NUM_POINTS {
            let mut r = b[i];
            for j in 0..NUM_POINTS {
                if j != i {
                    r -= a[i][j] * w[j];
                }
            }
            w[i] = if a[i][i] > 1e-12 {
                (r / a[i][i]).max(0.0)
            } else {
                0.0
            };
        }
    }
    std::array::from_fn(|i| w[i] as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The LUT libjxl writes for `--photon_noise_iso=6400` on a 768x512
    /// image; the decoded grain amplitude matched cjxl's within 0.3%.
    #[test]
    fn photon_noise_matches_libjxl() {
        let lut = photon_noise(768, 512, 6400.0).lut;
        let expected = [
            0.0027226498,
            0.0143037215,
            0.00695769,
            0.0064165955,
            0.0070083137,
            0.008079084,
            0.0093738455,
            0.010781403,
        ];
        for (got, want) in lut.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn lut_serializes_to_ten_bit_fields() {
        let params = NoiseParams {
            lut: [0.0, 0.5, 1.0, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        };
        let mut w = BitWriter::new();
        params.write(&mut w);
        assert_eq!(w.bits_written(), NoiseParams::BITS);
        let bytes = w.into_bytes();
        // Fields are ten bits each, LSB first: 0, then 512.
        let first = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
        assert_eq!(first & 0x3ff, 0);
        assert_eq!((first >> 10) & 0x3ff, 512);
        // 1.0 clamps to the ten-bit maximum.
        assert_eq!(
            (first >> 20) & 0xf | ((u32::from(bytes[3]) & 0x3f) << 4),
            1023
        );
    }

    #[test]
    fn flat_image_measures_no_grain() {
        let mut img = Image3F::new(64, 64);
        for c in 0..3 {
            for y in 0..64 {
                img.plane_row_mut(c, y).fill(0.3);
            }
        }
        assert!(!measured_grain(&img).has_any());
    }

    /// White noise of a known standard deviation on a flat field comes back
    /// as a LUT near the strength that reproduces its Laplacian level.
    #[test]
    fn white_noise_measures_its_level() {
        let (w, h) = (256, 256);
        let mut img = Image3F::new(w, h);
        let sigma = 0.01f32;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut gauss = || {
            // Sum of 12 uniforms minus 6: a unit-variance Gaussian.
            let mut s = 0.0f32;
            for _ in 0..12 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                s += (state >> 40) as f32 / (1u64 << 24) as f32;
            }
            s - 6.0
        };
        for y in 0..h {
            for x in 0..w {
                let n = sigma * gauss();
                img.plane_row_mut(0, y)[x] = 0.0;
                img.plane_row_mut(1, y)[x] = 0.5 + n;
                img.plane_row_mut(2, y)[x] = 0.5 + n;
            }
        }
        let lut = measured_grain(&img).lut;
        // Intensity (X+Y)/2 = 0.25 is halfway through the second segment.
        // Mean |Laplacian| of white noise: sqrt(2/pi) * ||kernel|| * sigma/2.
        let level =
            (2.0 / std::f32::consts::PI).sqrt() * (25.0f32 + 4.0 + 0.25).sqrt() * sigma * 0.5;
        let want = level / LAPLACIAN_LEVEL_PER_UNIT;
        let got = lut[1] * 0.5 + lut[2] * 0.5;
        assert!(
            got > want * 0.6 && got < want * 1.2,
            "lut {lut:?}, want about {want}"
        );
    }
}
