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
//! Fixed symmetric Gaussian filter for the AQ loop's MS-SSIM moments.

use std::sync::OnceLock;

pub(crate) const WINDOW: usize = 11;
pub(crate) const RADIUS: usize = WINDOW / 2;
pub(crate) type BlurRows<'a> = [&'a [f32]; WINDOW];

// SIMD kernels return the processed prefix; the driver handles scalar tails.
type HorizontalFn = fn(&[f32], &[f32; WINDOW], &mut [f32]) -> usize;
type VerticalFn = fn(&BlurRows<'_>, &[f32; WINDOW], &mut [f32]) -> usize;

#[derive(Clone, Copy)]
struct Kernels {
    horizontal: HorizontalFn,
    vertical: VerticalFn,
}

#[cfg(any(
    test,
    not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))
))]
const SCALAR: Kernels = Kernels {
    horizontal: horizontal_scalar,
    vertical: vertical_scalar,
};

fn selected_kernels() -> Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    *KERNELS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            return Kernels {
                horizontal: |src, k, dst| unsafe { crate::avx::blur_horizontal_avx2(src, k, dst) },
                vertical: |rows, k, dst| unsafe { crate::avx::blur_vertical_avx2(rows, k, dst) },
            };
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Kernels {
                horizontal: |src, k, dst| unsafe { crate::sse::blur_horizontal_sse41(src, k, dst) },
                vertical: |rows, k, dst| unsafe { crate::sse::blur_vertical_sse41(rows, k, dst) },
            };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Kernels {
                horizontal: |src, k, dst| unsafe { crate::neon::blur_horizontal_neon(src, k, dst) },
                vertical: |rows, k, dst| unsafe { crate::neon::blur_vertical_neon(rows, k, dst) },
            };
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            Kernels {
                horizontal: crate::wasm::blur_horizontal_wasm,
                vertical: crate::wasm::blur_vertical_wasm,
            }
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
        SCALAR
    })
}

fn kernel() -> &'static [f32; WINDOW] {
    static KERNEL: OnceLock<[f32; WINDOW]> = OnceLock::new();
    KERNEL.get_or_init(|| {
        const SIGMA: f32 = 1.5;
        let mut k = [0.0; WINDOW];
        let mut sum = 0.0;
        for (i, v) in k.iter_mut().enumerate() {
            let d = i as f32 - RADIUS as f32;
            *v = (-0.5 * d * d / (SIGMA * SIGMA)).exp();
            sum += *v;
        }
        for v in &mut k {
            *v /= sum;
        }
        k
    })
}

/// Reused for the five moment planes at one scale. Only WINDOW rows are kept,
/// so the horizontal intermediate stays in cache instead of filling a frame.
pub(crate) struct GaussianBlur {
    width: usize,
    ring: Vec<f32>,
    padded: Vec<f32>,
    kernels: Kernels,
}

impl GaussianBlur {
    pub(crate) fn new(width: usize) -> Self {
        Self {
            width,
            ring: vec![0.0; width * WINDOW],
            padded: vec![0.0; width + 2 * RADIUS],
            kernels: selected_kernels(),
        }
    }

    /// Separable Gaussian blur with edge replication in both directions.
    pub(crate) fn blur(&mut self, src: &[f32], height: usize, out: &mut [f32]) {
        let w = self.width;
        assert_eq!(src.len(), w * height);
        assert_eq!(out.len(), src.len());
        if w == 0 || height == 0 {
            return;
        }
        let k = kernel();
        let mut next_row = 0;
        for y in 0..height {
            let last_row = (y + RADIUS).min(height - 1);
            while next_row <= last_row {
                let row = &src[next_row * w..(next_row + 1) * w];
                self.padded[..RADIUS].fill(row[0]);
                self.padded[RADIUS..RADIUS + w].copy_from_slice(row);
                self.padded[RADIUS + w..].fill(row[w - 1]);
                let slot = next_row % WINDOW;
                let dst = &mut self.ring[slot * w..(slot + 1) * w];
                let done = (self.kernels.horizontal)(&self.padded, k, dst);
                horizontal_scalar(&self.padded[done..], k, &mut dst[done..]);
                next_row += 1;
            }
            // At most WINDOW consecutive source rows are live. Repeated edge
            // rows point to the same slot without copying or re-filtering them.
            let rows: BlurRows<'_> = std::array::from_fn(|i| {
                let sy = (y + i).saturating_sub(RADIUS).min(height - 1);
                let slot = sy % WINDOW;
                &self.ring[slot * w..(slot + 1) * w]
            });
            let dst = &mut out[y * w..(y + 1) * w];
            let done = (self.kernels.vertical)(&rows, k, dst);
            if done < w {
                let tail = rows.map(|row| &row[done..]);
                vertical_scalar(&tail, k, &mut dst[done..]);
            }
        }
    }
}

// Center plus symmetric pairs, as in libblur. Separate multiplies/adds keep
// SIMD and scalar tails consistent without requiring FMA on x86.
fn horizontal_scalar(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    for (x, dst) in dst.iter_mut().enumerate() {
        let mut acc = src[x + RADIUS] * k[RADIUS];
        for i in 0..RADIUS {
            acc += (src[x + i] + src[x + WINDOW - 1 - i]) * k[i];
        }
        *dst = acc;
    }
    dst.len()
}

fn vertical_scalar(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    for (x, dst) in dst.iter_mut().enumerate() {
        let mut acc = rows[RADIUS][x] * k[RADIUS];
        for i in 0..RADIUS {
            acc += (rows[i][x] + rows[WINDOW - 1 - i][x]) * k[i];
        }
        *dst = acc;
    }
    dst.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available_kernels() -> Vec<(&'static str, Kernels)> {
        #[allow(unused_mut)]
        let mut kernels = vec![("dispatch", selected_kernels()), ("scalar", SCALAR)];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            kernels.push((
                "neon",
                Kernels {
                    horizontal: |s, k, d| unsafe { crate::neon::blur_horizontal_neon(s, k, d) },
                    vertical: |s, k, d| unsafe { crate::neon::blur_vertical_neon(s, k, d) },
                },
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push((
                "avx2",
                Kernels {
                    horizontal: |s, k, d| unsafe { crate::avx::blur_horizontal_avx2(s, k, d) },
                    vertical: |s, k, d| unsafe { crate::avx::blur_vertical_avx2(s, k, d) },
                },
            ));
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            kernels.push((
                "sse4.1",
                Kernels {
                    horizontal: |s, k, d| unsafe { crate::sse::blur_horizontal_sse41(s, k, d) },
                    vertical: |s, k, d| unsafe { crate::sse::blur_vertical_sse41(s, k, d) },
                },
            ));
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        kernels.push((
            "wasm",
            Kernels {
                horizontal: crate::wasm::blur_horizontal_wasm,
                vertical: crate::wasm::blur_vertical_wasm,
            },
        ));
        kernels
    }

    // Independent full-frame, eleven-tap implementation from the old metric.
    fn reference(src: &[f32], w: usize, h: usize) -> Vec<f32> {
        let k = kernel();
        let mut tmp = vec![0.0; src.len()];
        let mut out = vec![0.0; src.len()];
        for y in 0..h {
            for x in 0..w {
                for (i, &weight) in k.iter().enumerate() {
                    let sx = (x + i).saturating_sub(RADIUS).min(w - 1);
                    tmp[y * w + x] += weight * src[y * w + sx];
                }
            }
        }
        for y in 0..h {
            for (i, &weight) in k.iter().enumerate() {
                let sy = (y + i).saturating_sub(RADIUS).min(h - 1);
                for x in 0..w {
                    out[y * w + x] += weight * tmp[sy * w + x];
                }
            }
        }
        out
    }

    fn check(src: &[f32], w: usize, h: usize) {
        let expected = reference(src, w, h);
        let mut scalar = vec![f32::NAN; src.len()];
        let mut blur = GaussianBlur::new(w);
        blur.kernels = SCALAR;
        blur.blur(src, h, &mut scalar);
        for (name, kernels) in available_kernels() {
            blur.kernels = kernels;
            let mut out = vec![f32::NAN; src.len()];
            blur.blur(src, h, &mut out);
            for (i, ((actual, expected), scalar)) in
                out.iter().zip(&expected).zip(&scalar).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    scalar.to_bits(),
                    "{name}: {w}x{h}, index={i}"
                );
                assert!(
                    (actual - expected).abs() <= 5e-7,
                    "{name}: {w}x{h}, index={i}, {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn matches_reference_for_edges_tails_and_ring_wraps() {
        let widths = (1..=40).chain([47, 63, 64, 65, 127, 128, 129]);
        for w in widths {
            for h in (1..=13).chain([23, 37]) {
                let src: Vec<f32> = (0..w * h)
                    .map(|i| ((i * 71 + i / w * 113) % 101) as f32 / 50.0 - 1.0)
                    .collect();
                check(&src, w, h);
            }
        }
    }

    #[test]
    fn preserves_constants_and_clamps_corner_impulses() {
        for (w, h) in [(1, 1), (3, 2), (17, 29), (65, 23)] {
            check(&vec![0.4; w * h], w, h);
            for i in [0, w - 1, (h - 1) * w, w * h - 1, (h / 2) * w + w / 2] {
                let mut src = vec![0.0; w * h];
                src[i] = 1.0;
                check(&src, w, h);
            }
        }
    }

    #[test]
    fn scratch_can_be_reused_for_different_heights() {
        let w = 19;
        let mut blur = GaussianBlur::new(w);
        for h in [25, 1, 3, 37, 12] {
            let src: Vec<f32> = (0..w * h).map(|i| (i % 47) as f32 / 47.0).collect();
            let mut out = vec![f32::NAN; src.len()];
            blur.blur(&src, h, &mut out);
            for (a, b) in out.iter().zip(reference(&src, w, h)) {
                assert!((a - b).abs() <= 5e-7);
            }
        }
        assert_eq!(blur.ring.len(), w * WINDOW);
    }

    #[test]
    fn empty_images_are_supported() {
        GaussianBlur::new(0).blur(&[], 4, &mut []);
        GaussianBlur::new(7).blur(&[], 0, &mut []);
    }
}
