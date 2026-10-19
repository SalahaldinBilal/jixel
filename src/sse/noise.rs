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
use crate::noise::{NOISE_AC_MASK, texture_score_from_simd_sads};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub(crate) fn noise_laplacian_stats_sse41(patch: &[f32; 64]) -> [f32; 2] {
    let zero = _mm_set1_ps(0.0);
    let quarter = _mm_set1_ps(-0.25);
    let five = _mm_set1_ps(5.0);
    let sign = _mm_set1_ps(-0.0);
    let mut rows = [[zero; 6]; 8];
    for (src, dst) in patch.as_chunks::<8>().0.iter().zip(&mut rows) {
        let lo = unsafe { _mm_loadu_ps(src.as_ptr()) };
        let hi = unsafe { _mm_loadu_ps(src.as_ptr().add(4)) };
        // Left / center / right, with mirrors [1, 0, ...] and [..., 7, 6].
        *dst = [
            _mm_shuffle_ps::<0x91>(lo, lo),
            _mm_castsi128_ps(_mm_alignr_epi8::<12>(
                _mm_castps_si128(hi),
                _mm_castps_si128(lo),
            )),
            lo,
            hi,
            _mm_castsi128_ps(_mm_alignr_epi8::<4>(
                _mm_castps_si128(hi),
                _mm_castps_si128(lo),
            )),
            _mm_shuffle_ps::<0xb9>(hi, hi),
        ];
    }
    let mut abs = [0.0; 64];
    for y in 0..8 {
        let top = &rows[if y == 0 { 1 } else { y - 1 }];
        let center = &rows[y];
        let bottom = &rows[if y == 7 { 6 } else { y + 1 }];
        // Match the scalar tap order without fused multiply-adds.
        for half in 0..2 {
            let mut filtered = _mm_add_ps(zero, _mm_mul_ps(quarter, top[half]));
            filtered = _mm_sub_ps(filtered, top[2 + half]);
            filtered = _mm_add_ps(filtered, _mm_mul_ps(quarter, top[4 + half]));
            filtered = _mm_sub_ps(filtered, center[half]);
            filtered = _mm_add_ps(filtered, _mm_mul_ps(five, center[2 + half]));
            filtered = _mm_sub_ps(filtered, center[4 + half]);
            filtered = _mm_add_ps(filtered, _mm_mul_ps(quarter, bottom[half]));
            filtered = _mm_sub_ps(filtered, bottom[2 + half]);
            filtered = _mm_add_ps(filtered, _mm_mul_ps(quarter, bottom[4 + half]));
            unsafe {
                _mm_storeu_ps(
                    abs.as_mut_ptr().add(y * 8 + half * 4),
                    _mm_andnot_ps(sign, filtered),
                )
            };
        }
    }
    crate::noise::noise_laplacian_stats_from_abs(patch, &abs)
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn noise_shrink_sse41(coeffs: &mut [f32; 64], t2: f32) {
    if !(t2 > 0.0) {
        return;
    }
    let threshold = _mm_set1_ps(t2);
    let zero = _mm_set1_ps(0.0);
    for (p, mask) in coeffs
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(NOISE_AC_MASK.as_chunks::<4>().0)
    {
        let c = unsafe { _mm_loadu_ps(p.as_ptr()) };
        let c2 = _mm_mul_ps(c, c);
        let gain = _mm_div_ps(_mm_sub_ps(c2, threshold), c2);
        let shrunk = _mm_mul_ps(c, _mm_max_ps(gain, zero));
        let mask = unsafe { _mm_loadu_si128(mask.as_ptr().cast()) };
        unsafe {
            _mm_storeu_ps(
                p.as_mut_ptr(),
                _mm_blendv_ps(c, shrunk, _mm_castsi128_ps(mask)),
            )
        };
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn noise_texture_score_sse41(patch: &[f32; 64]) -> f32 {
    let mut sads = [0.0; 20];
    for yb in 0..4 {
        let mut sad = _mm_set1_ps(0.0);
        for cy in 0..4 {
            for cx in 0..3 {
                let center = _mm_set1_ps(patch[(2 + cy) * 8 + 2 + cx]);
                let pixels = unsafe { _mm_loadu_ps(patch.as_ptr().add((yb + cy) * 8 + cx)) };
                sad = _mm_add_ps(
                    sad,
                    _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(center, pixels)),
                );
            }
        }
        unsafe { _mm_storeu_ps(sads.as_mut_ptr().add(yb * 5), sad) };
    }
    texture_score_from_simd_sads(patch, sads)
}
