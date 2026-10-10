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

use crate::aq_blur::{GaussianBlur, WINDOW};
use crate::image::Plane;
use std::sync::OnceLock;

/// SSIM constants for a luma plane with unit dynamic range.
pub(crate) const C1: f32 = 1e-4;
pub(crate) const C2: f32 = 9e-4;
/// Wang et al. 2003 scale weights.
static MS_WEIGHTS: [f32; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

/// Blurred first and second moments, all with the same length.
pub(crate) struct SsimMoments<'a> {
    pub(crate) mx: &'a [f32],
    pub(crate) my: &'a [f32],
    pub(crate) sxx: &'a [f32],
    pub(crate) syy: &'a [f32],
    pub(crate) sxy: &'a [f32],
}

impl SsimMoments<'_> {
    pub(crate) fn assert_len(&self, n: usize) {
        assert_eq!(self.mx.len(), n);
        assert_eq!(self.my.len(), n);
        assert_eq!(self.sxx.len(), n);
        assert_eq!(self.syy.len(), n);
        assert_eq!(self.sxy.len(), n);
    }

    pub(crate) fn tail(&self, start: usize) -> SsimMoments<'_> {
        SsimMoments {
            mx: &self.mx[start..],
            my: &self.my[start..],
            sxx: &self.sxx[start..],
            syy: &self.syy[start..],
            sxy: &self.sxy[start..],
        }
    }
}

// SIMD kernels write a prefix and return its length. Preserve the scalar
// arithmetic order and use full division: small variance differences matter.
type SsimMomentsFn = fn(&SsimMoments<'_>, &mut [f32], &mut [f32]) -> usize;

fn selected_moments_fn() -> SsimMomentsFn {
    static MOMENTS: OnceLock<SsimMomentsFn> = OnceLock::new();
    *MOMENTS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            return |src, l, cs| unsafe { crate::avx::ssim_moments_avx2(src, l, cs) };
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return |src, l, cs| unsafe { crate::sse::ssim_moments_sse41(src, l, cs) };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return |src, l, cs| unsafe { crate::neon::ssim_moments_neon(src, l, cs) };
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            crate::wasm::ssim_moments_wasm
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
        ssim_moments_scalar
    })
}

fn ssim_from_moments(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) {
    ssim_from_moments_with_kernel(src, l, cs, selected_moments_fn());
}

fn ssim_from_moments_with_kernel(
    src: &SsimMoments<'_>,
    l: &mut [f32],
    cs: &mut [f32],
    kernel: SsimMomentsFn,
) {
    src.assert_len(l.len());
    assert_eq!(l.len(), cs.len());
    let done = kernel(src, l, cs);
    if done < l.len() {
        ssim_moments_scalar(&src.tail(done), &mut l[done..], &mut cs[done..]);
    }
}

fn ssim_moments_scalar(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) -> usize {
    for i in 0..l.len() {
        let (a, b) = (src.mx[i], src.my[i]);
        let vx = src.sxx[i] - a * a;
        let vy = src.syy[i] - b * b;
        let cov = src.sxy[i] - a * b;
        l[i] = (2.0 * a * b + C1) / (a * a + b * b + C1);
        cs[i] = (2.0 * cov + C2) / (vx + vy + C2);
    }
    l.len()
}

/// Luminance and contrast-structure maps of SSIM.
fn ssim_maps(x: &[f32], y: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut mx = vec![0.0f32; n];
    let mut my = vec![0.0f32; n];
    let mut sxx = vec![0.0f32; n];
    let mut syy = vec![0.0f32; n];
    let mut sxy = vec![0.0f32; n];
    {
        let mut blur = GaussianBlur::new(w);
        blur.blur(x, h, &mut mx);
        blur.blur(y, h, &mut my);
        {
            let prod: Vec<f32> = x.iter().zip(y).map(|(a, b)| a * b).collect();
            blur.blur(&prod, h, &mut sxy);
        }
        {
            let sq: Vec<f32> = x.iter().map(|a| a * a).collect();
            blur.blur(&sq, h, &mut sxx);
        }
        {
            let sq: Vec<f32> = y.iter().map(|a| a * a).collect();
            blur.blur(&sq, h, &mut syy);
        }
    }
    let mut l = vec![0.0f32; n];
    let mut cs = vec![0.0f32; n];
    ssim_from_moments(
        &SsimMoments {
            mx: &mx,
            my: &my,
            sxx: &sxx,
            syy: &syy,
            sxy: &sxy,
        },
        &mut l,
        &mut cs,
    );
    (l, cs)
}

type DownsampleRowsFn = fn(&[f32], &[f32], &mut [f32]) -> usize;

fn selected_downsample_fn() -> DownsampleRowsFn {
    static ROWS: OnceLock<DownsampleRowsFn> = OnceLock::new();
    *ROWS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            return |top, bottom, dst| unsafe {
                crate::avx::downsample_rows_avx2(top, bottom, dst)
            };
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return |top, bottom, dst| unsafe {
                crate::sse::downsample_rows_sse41(top, bottom, dst)
            };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return |top, bottom, dst| unsafe {
                crate::neon::downsample_rows_neon(top, bottom, dst)
            };
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            crate::wasm::downsample_rows_wasm
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
        downsample_rows_scalar
    })
}

fn downsample_rows_scalar(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    for (x, dst) in dst.iter_mut().enumerate() {
        let i = 2 * x;
        *dst = 0.25 * (top[i] + top[i + 1] + bottom[i] + bottom[i + 1]);
    }
    dst.len()
}

/// Half-size box filter. Incomplete pairs at the right and bottom are dropped.
fn downsample(a: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (w2, h2) = (w / 2, h / 2);
    let mut out = vec![0.0f32; w2 * h2];
    downsample_into(a, w, h, &mut out, selected_downsample_fn());
    (out, w2, h2)
}

fn downsample_into(a: &[f32], w: usize, h: usize, out: &mut [f32], rows: DownsampleRowsFn) {
    assert_eq!(a.len(), w * h);
    let (w2, h2) = (w / 2, h / 2);
    assert_eq!(out.len(), w2 * h2);
    if w2 == 0 || h2 == 0 {
        return;
    }
    for y in 0..h2 {
        let i = 2 * y * w;
        let top = &a[i..i + w];
        let bottom = &a[i + w..i + 2 * w];
        let dst = &mut out[y * w2..(y + 1) * w2];
        let done = rows(top, bottom, dst);
        if done < w2 {
            downsample_rows_scalar(&top[2 * done..], &bottom[2 * done..], &mut dst[done..]);
        }
    }
}

fn flatten(p: &Plane<f32>) -> Vec<f32> {
    let (w, h) = (p.xsize(), p.ysize());
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        v.extend_from_slice(&p.row(y)[..w]);
    }
    v
}

/// Scales below this one are skipped: the first carries 4.5% of the weight
/// and three quarters of the blur work.
const FIRST_SCALE: usize = 1;

/// `1 - MS-SSIM` of `recon` against `source` (mean pooling at every scale)
/// together with its per-pixel map: the scale weights applied to `1 - cs`
/// (`1 - l*cs` at the last scale), each scale upsampled to full resolution.
/// The map's mean approximates the deficit, so a region's mean is that
/// region's share of the frame deficit.
pub(crate) fn msssim_deficit_and_map(source: &Plane<f32>, recon: &Plane<f32>) -> (f32, Vec<f32>) {
    let (w0, h0) = (source.xsize(), source.ysize());
    let (mut w, mut h) = (w0, h0);
    let mut x = flatten(source);
    let mut y = flatten(recon);
    let mut map = vec![0.0f32; w0 * h0];
    let mut value = 1.0f64;
    let mut used = 0.0f64;
    for (scale, &weight) in MS_WEIGHTS.iter().enumerate() {
        if w < WINDOW || h < WINDOW {
            break;
        }
        let last = scale + 1 == MS_WEIGHTS.len();
        if scale >= FIRST_SCALE {
            let (l, cs) = ssim_maps(&x, &y, w, h);
            let step = 1usize << scale;
            let mut sum = 0.0f64;
            for py in 0..h0 {
                let sy = (py / step).min(h - 1);
                let row = &mut map[py * w0..(py + 1) * w0];
                for (px, slot) in row.iter_mut().enumerate() {
                    let sx = (px / step).min(w - 1);
                    let i = sy * w + sx;
                    let m = if last { l[i] * cs[i] } else { cs[i] };
                    *slot += weight * (1.0 - m).max(0.0);
                }
            }
            for i in 0..w * h {
                sum += if last {
                    (l[i] * cs[i]) as f64
                } else {
                    cs[i] as f64
                };
            }
            value *= (sum / (w * h) as f64).max(1e-6).powf(weight as f64);
            used += weight as f64;
        }
        if last {
            break;
        }
        let (nx, nw, nh) = downsample(&x, w, h);
        let (ny, _, _) = downsample(&y, w, h);
        x = nx;
        y = ny;
        w = nw;
        h = nh;
    }
    if used <= 0.0 {
        return (0.0, map);
    }
    if used < 0.999 {
        for v in &mut map {
            *v /= used as f32;
        }
    }
    ((1.0 - value.powf(1.0 / used)) as f32, map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available_downsample_kernels() -> Vec<(&'static str, DownsampleRowsFn)> {
        #[allow(unused_mut)]
        let mut kernels: Vec<(&str, DownsampleRowsFn)> = vec![
            ("dispatch", selected_downsample_fn()),
            ("scalar", downsample_rows_scalar),
        ];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            kernels.push(("neon", |a, b, d| unsafe {
                crate::neon::downsample_rows_neon(a, b, d)
            }));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", |a, b, d| unsafe {
                crate::avx::downsample_rows_avx2(a, b, d)
            }));
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            kernels.push(("sse4.1", |a, b, d| unsafe {
                crate::sse::downsample_rows_sse41(a, b, d)
            }));
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        kernels.push(("wasm", crate::wasm::downsample_rows_wasm));
        kernels
    }

    fn downsample_reference(a: &[f32], w: usize, h: usize) -> Vec<f32> {
        let (w2, h2) = (w / 2, h / 2);
        let mut out = vec![0.0; w2 * h2];
        for y in 0..h2 {
            for x in 0..w2 {
                let i = 2 * y * w + 2 * x;
                out[y * w2 + x] = 0.25 * (a[i] + a[i + 1] + a[i + w] + a[i + w + 1]);
            }
        }
        out
    }

    fn check_downsample(a: &[f32], w: usize, h: usize) {
        let expected = downsample_reference(a, w, h);
        const GUARD: f32 = -12345.0;
        for (name, kernel) in available_downsample_kernels() {
            for offset in 0..4 {
                let n = expected.len();
                let mut out = vec![GUARD; offset + n + 4];
                out[offset..offset + n].fill(f32::NAN);
                downsample_into(a, w, h, &mut out[offset..offset + n], kernel);
                for (i, (a, b)) in out[offset..offset + n].iter().zip(&expected).enumerate() {
                    if b.is_nan() {
                        assert!(a.is_nan(), "{name}: {w}x{h}, index={i}");
                    } else {
                        assert_eq!(a.to_bits(), b.to_bits(), "{name}: {w}x{h}, index={i}");
                    }
                }
                assert!(
                    out[..offset]
                        .iter()
                        .chain(&out[offset + n..])
                        .all(|&v| v == GUARD)
                );
            }
        }
        let (out, w2, h2) = downsample(a, w, h);
        assert_eq!((w2, h2), (w / 2, h / 2));
        assert_eq!(out.len(), expected.len());
    }

    #[test]
    fn downsample_kernels_match_reference_for_tails_odd_sizes_and_alignment() {
        for w in (0..=69).chain([127, 128, 129, 255, 256, 257]) {
            for h in (0..=7).chain([11, 12, 23, 47]) {
                let offset = (w + h) % 4;
                let mut data = vec![0.0; offset + w * h];
                for y in 0..h {
                    for x in 0..w {
                        data[offset + y * w + x] = if x + 1 == w || y + 1 == h {
                            1000.0 + x as f32 + y as f32
                        } else {
                            ((x * 71 + y * 113) % 101) as f32 / 50.0 - 1.0
                        };
                    }
                }
                check_downsample(&data[offset..], w, h);
            }
        }
    }

    #[test]
    fn downsample_keeps_scalar_sum_order_for_cancellation_and_extremes() {
        for w in [2, 3, 8, 9, 16, 17, 32, 33, 64, 65, 129] {
            let mut data = vec![0.0; w * 2];
            for x in (0..w - 1).step_by(2) {
                data[x] = 1e20;
                data[x + 1] = 1.0;
                data[w + x] = -1e20;
                data[w + x + 1] = 1.0;
            }
            // Regrouping into two row sums would return zero instead of 0.25.
            check_downsample(&data, w, 2);
            assert!(downsample(&data, w, 2).0.iter().all(|&v| v == 0.25));
            for value in [
                0.4,
                -0.0,
                f32::MIN_POSITIVE,
                f32::MAX,
                f32::INFINITY,
                f32::NAN,
            ] {
                data.fill(value);
                check_downsample(&data, w, 2);
            }
        }
    }

    #[test]
    #[should_panic]
    fn downsample_rejects_mismatched_dimensions() {
        downsample(&[0.0; 8], 3, 3);
    }

    fn available_moments_kernels() -> Vec<(&'static str, SsimMomentsFn)> {
        #[allow(unused_mut)]
        let mut kernels: Vec<(&str, SsimMomentsFn)> = vec![
            ("dispatch", selected_moments_fn()),
            ("scalar", ssim_moments_scalar),
        ];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            kernels.push(("neon", |s, l, cs| unsafe {
                crate::neon::ssim_moments_neon(s, l, cs)
            }));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", |s, l, cs| unsafe {
                crate::avx::ssim_moments_avx2(s, l, cs)
            }));
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            kernels.push(("sse4.1", |s, l, cs| unsafe {
                crate::sse::ssim_moments_sse41(s, l, cs)
            }));
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        kernels.push(("wasm", crate::wasm::ssim_moments_wasm));
        kernels
    }

    fn check_moments(src: &SsimMoments<'_>) {
        let n = src.mx.len();
        let mut expected_l = vec![f32::NAN; n];
        let mut expected_cs = vec![f32::NAN; n];
        ssim_moments_scalar(src, &mut expected_l, &mut expected_cs);
        const GUARD: f32 = -12345.0;
        for (name, kernel) in available_moments_kernels() {
            for offset in 0..4 {
                let mut l = vec![GUARD; offset + n + 4];
                let mut cs = l.clone();
                l[offset..offset + n].fill(f32::NAN);
                cs[offset..offset + n].fill(f32::NAN);
                ssim_from_moments_with_kernel(
                    src,
                    &mut l[offset..offset + n],
                    &mut cs[offset..offset + n],
                    kernel,
                );
                for (map, actual, expected) in [
                    ("l", &l[offset..offset + n], &expected_l),
                    ("cs", &cs[offset..offset + n], &expected_cs),
                ] {
                    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                        if b.is_nan() {
                            assert!(a.is_nan(), "{name}: {map}[{i}], n={n}");
                        } else {
                            assert_eq!(a.to_bits(), b.to_bits(), "{name}: {map}[{i}], n={n}");
                        }
                    }
                }
                assert!(
                    l[..offset]
                        .iter()
                        .chain(&l[offset + n..])
                        .all(|&v| v == GUARD)
                );
                assert!(
                    cs[..offset]
                        .iter()
                        .chain(&cs[offset + n..])
                        .all(|&v| v == GUARD)
                );
            }
        }
    }

    #[test]
    fn moment_kernels_match_scalar_for_all_tails_and_unaligned_buffers() {
        for n in (0..=97).chain([127, 128, 129, 255, 1025, 4099]) {
            for offset in 0..4 {
                let mut inputs = [const { Vec::new() }; 5];
                for i in 0..offset + n {
                    let a = ((i * 31 + 13) % 101) as f32 / 100.0;
                    let b = ((i * 67 + 23) % 97) as f32 / 96.0;
                    let vx = (i % 7) as f32 * 0.00001;
                    let vy = (i % 13) as f32 * 0.00002;
                    let cov = ((i % 5) as f32 - 2.0) * 0.00001;
                    for (dst, v) in
                        inputs
                            .iter_mut()
                            .zip([a, b, a * a + vx, b * b + vy, a * b + cov])
                    {
                        dst.push(v);
                    }
                }
                check_moments(&SsimMoments {
                    mx: &inputs[0][offset..],
                    my: &inputs[1][offset..],
                    sxx: &inputs[2][offset..],
                    syy: &inputs[3][offset..],
                    sxy: &inputs[4][offset..],
                });
            }
        }
    }

    #[test]
    fn moment_kernels_preserve_flat_variances_and_extreme_arithmetic() {
        let a = [0.0, 0.37, 1.0, -0.7, 1e-20, f32::MAX * 0.75, 0.0, 0.0];
        let b = [0.0, 0.37, 1.0, 0.6, 1e-20, 1e-30, 0.0, 0.0];
        let mx: Vec<_> = (0..67).map(|i| a[i % a.len()]).collect();
        let my: Vec<_> = (0..67).map(|i| b[i % b.len()]).collect();
        let mut sxx: Vec<_> = mx.iter().map(|a| a * a).collect();
        let mut syy: Vec<_> = my.iter().map(|b| b * b).collect();
        let mut sxy: Vec<_> = mx.iter().zip(&my).map(|(a, b)| a * b).collect();
        for i in (6..67).step_by(8) {
            sxx[i] = -C2 / 2.0;
            syy[i] = -C2 / 2.0; // Zero denominator: keep full division semantics.
        }
        for i in (7..67).step_by(8) {
            sxy[i] = f32::NAN;
        }
        check_moments(&SsimMoments {
            mx: &mx,
            my: &my,
            sxx: &sxx,
            syy: &syy,
            sxy: &sxy,
        });
    }

    #[test]
    #[should_panic]
    fn moment_maps_reject_mismatched_input_lengths() {
        let one = [0.0];
        ssim_from_moments(
            &SsimMoments {
                mx: &one,
                my: &[],
                sxx: &one,
                syy: &one,
                sxy: &one,
            },
            &mut [0.0],
            &mut [0.0],
        );
    }

    fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Plane<f32> {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                p.row_mut(y)[x] = f(x, y);
            }
        }
        p
    }

    #[test]
    fn identical_planes_have_zero_deficit() {
        let p = plane(64, 48, |x, y| ((x * 7 + y * 3) % 17) as f32 / 17.0);
        let (d, map) = msssim_deficit_and_map(&p, &p);
        assert!(d.abs() < 1e-5 && map.iter().all(|&v| v < 1e-5));
    }

    #[test]
    fn deficit_grows_with_noise_and_pools_the_worst_tile() {
        let src = plane(96, 64, |x, y| 0.3 + 0.2 * ((x / 8 + y / 8) % 2) as f32);
        let small = plane(96, 64, |x, y| {
            src.row(y)[x] + 0.01 * ((x * 13 + y * 7) % 5) as f32
        });
        let large = plane(96, 64, |x, y| {
            src.row(y)[x] + 0.05 * ((x * 13 + y * 7) % 5) as f32
        });
        let ds = msssim_deficit_and_map(&src, &small).0;
        let dl = msssim_deficit_and_map(&src, &large).0;
        assert!(0.0 < ds && ds < dl, "{ds} {dl}");
        // Only the top-left 32px tile is damaged: it must pool worst.
        let local = plane(96, 64, |x, y| {
            src.row(y)[x]
                + if x < 32 && y < 32 {
                    0.05 * ((x * 13 + y * 7) % 5) as f32
                } else {
                    0.0
                }
        });
        let (d, map) = msssim_deficit_and_map(&src, &local);
        let mean = map.iter().sum::<f32>() / map.len() as f32;
        assert!((mean - d).abs() < 0.3 * d, "map mean {mean} vs deficit {d}");
        let tile = |tx: usize, ty: usize| -> f32 {
            let mut acc = 0.0;
            for y in ty * 32..ty * 32 + 32 {
                for x in tx * 32..tx * 32 + 32 {
                    acc += map[y * 96 + x];
                }
            }
            acc / 1024.0
        };
        assert!(
            tile(0, 0) > 2.0 * tile(2, 1),
            "{} {}",
            tile(0, 0),
            tile(2, 1)
        );
    }
}
