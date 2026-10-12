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

use std::sync::OnceLock;

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
        {
            |top, bottom, dst| unsafe { crate::neon::downsample_rows_neon(top, bottom, dst) }
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            crate::wasm::downsample_rows_wasm
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
            all(target_arch = "aarch64", feature = "neon")
        )))]
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

/// Shared SIMD box filter. A final unpaired column repeats the right edge
/// when the destination requests ceil rather than floor dimensions.
pub(crate) fn downsample_rows(top: &[f32], bottom: &[f32], dst: &mut [f32]) {
    downsample_rows_with_kernel(top, bottom, dst, selected_downsample_fn());
}

fn downsample_rows_with_kernel(
    top: &[f32],
    bottom: &[f32],
    dst: &mut [f32],
    rows: DownsampleRowsFn,
) {
    assert_eq!(top.len(), bottom.len());
    assert!(dst.len() <= top.len().div_ceil(2));
    let pairs = dst.len().min(top.len() / 2);
    let done = rows(top, bottom, &mut dst[..pairs]);
    downsample_rows_scalar(&top[2 * done..], &bottom[2 * done..], &mut dst[done..pairs]);
    if pairs < dst.len() {
        let i = 2 * pairs;
        dst[pairs] = 0.25 * (top[i] + top[i] + bottom[i] + bottom[i]);
    }
}

/// Half-size box filter. Incomplete pairs at the right and bottom are dropped.
#[cfg(test)]
fn downsample(a: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (w2, h2) = (w / 2, h / 2);
    let mut out = vec![0.0f32; w2 * h2];
    downsample_into(a, w, h, &mut out, selected_downsample_fn());
    (out, w2, h2)
}

#[cfg(test)]
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
        downsample_rows_with_kernel(top, bottom, dst, rows);
    }
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

    #[test]
    fn downsample_rows_support_floor_and_ceil_with_edge_replication() {
        for w in (0usize..=40).chain([63, 64, 65, 127, 129]) {
            let top: Vec<f32> = (0..w).map(|x| ((x * 71) % 101) as f32 - 50.0).collect();
            let bottom: Vec<f32> = (0..w)
                .map(|x| ((x * 113) % 101) as f32 / 50.0 - 1.0)
                .collect();
            for n in [w / 2, w.div_ceil(2)] {
                let expected: Vec<f32> = (0..n)
                    .map(|x| {
                        let i = 2 * x;
                        let j = (i + 1).min(w - 1);
                        0.25 * (top[i] + top[j] + bottom[i] + bottom[j])
                    })
                    .collect();
                for (name, kernel) in available_downsample_kernels() {
                    let mut out = vec![f32::NAN; n];
                    downsample_rows_with_kernel(&top, &bottom, &mut out, kernel);
                    for (a, b) in out.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{name}: width={w}, outputs={n}");
                    }
                }
            }
        }
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
}
