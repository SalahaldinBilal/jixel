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
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn noise_laplacian_stats_avx2(patch: &[f32; 64]) -> [f32; 2] {
    let zero = _mm256_set1_ps(0.0);
    let quarter = _mm256_set1_ps(-0.25);
    let five = _mm256_set1_ps(5.0);
    let sign = _mm256_set1_ps(-0.0);
    let left = _mm256_setr_epi32(1, 0, 1, 2, 3, 4, 5, 6);
    let right = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 6);
    let mut rows = [[zero; 3]; 8];
    for (src, dst) in patch.as_chunks::<8>().0.iter().zip(&mut rows) {
        let center = unsafe { _mm256_loadu_ps(src.as_ptr()) };
        *dst = [
            _mm256_permutevar8x32_ps(center, left),
            center,
            _mm256_permutevar8x32_ps(center, right),
        ];
    }
    let mut abs = [0.0; 64];
    for y in 0..8 {
        let top = &rows[if y == 0 { 1 } else { y - 1 }];
        let center = &rows[y];
        let bottom = &rows[if y == 7 { 6 } else { y + 1 }];
        // Match the scalar tap order without fused multiply-adds.
        let mut filtered = _mm256_add_ps(zero, _mm256_mul_ps(quarter, top[0]));
        filtered = _mm256_sub_ps(filtered, top[1]);
        filtered = _mm256_add_ps(filtered, _mm256_mul_ps(quarter, top[2]));
        filtered = _mm256_sub_ps(filtered, center[0]);
        filtered = _mm256_add_ps(filtered, _mm256_mul_ps(five, center[1]));
        filtered = _mm256_sub_ps(filtered, center[2]);
        filtered = _mm256_add_ps(filtered, _mm256_mul_ps(quarter, bottom[0]));
        filtered = _mm256_sub_ps(filtered, bottom[1]);
        filtered = _mm256_add_ps(filtered, _mm256_mul_ps(quarter, bottom[2]));
        unsafe {
            _mm256_storeu_ps(
                abs.as_mut_ptr().add(y * 8),
                _mm256_andnot_ps(sign, filtered),
            )
        };
    }
    crate::noise::noise_laplacian_stats_from_abs(patch, &abs)
}

#[target_feature(enable = "avx2")]
pub(crate) fn noise_shrink_avx2(coeffs: &mut [f32; 64], t2: f32) {
    if t2 <= 0.0 || t2.is_nan() {
        return;
    }
    let threshold = _mm256_set1_ps(t2);
    let zero = _mm256_set1_ps(0.0);
    for (p, mask) in coeffs
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(NOISE_AC_MASK.as_chunks::<8>().0)
    {
        let c = unsafe { _mm256_loadu_ps(p.as_ptr()) };
        let c2 = _mm256_mul_ps(c, c);
        let gain = _mm256_div_ps(_mm256_sub_ps(c2, threshold), c2);
        let shrunk = _mm256_mul_ps(c, _mm256_max_ps(gain, zero));
        let mask = unsafe { _mm256_loadu_si256(mask.as_ptr().cast()) };
        unsafe {
            _mm256_storeu_ps(
                p.as_mut_ptr(),
                _mm256_blendv_ps(c, shrunk, _mm256_castsi256_ps(mask)),
            )
        };
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn noise_texture_score_avx2(patch: &[f32; 64]) -> f32 {
    let mut sads = [0.0; 20];
    let sign = _mm256_set1_ps(-0.0);
    // Eight independent SADs: four horizontal origins in two patch rows.
    for yb in (0..4).step_by(2) {
        let mut sad = _mm256_setzero_ps();
        for cy in 0..4 {
            for cx in 0..3 {
                let center = _mm256_set1_ps(patch[(2 + cy) * 8 + 2 + cx]);
                let lo = unsafe { _mm_loadu_ps(patch.as_ptr().add((yb + cy) * 8 + cx)) };
                let hi = unsafe { _mm_loadu_ps(patch.as_ptr().add((yb + 1 + cy) * 8 + cx)) };
                let pixels = _mm256_set_m128(hi, lo);
                sad = _mm256_add_ps(sad, _mm256_andnot_ps(sign, _mm256_sub_ps(center, pixels)));
            }
        }
        unsafe {
            _mm_storeu_ps(sads.as_mut_ptr().add(yb * 5), _mm256_castps256_ps128(sad));
            _mm_storeu_ps(
                sads.as_mut_ptr().add((yb + 1) * 5),
                _mm256_extractf128_ps::<1>(sad),
            );
        }
    }
    texture_score_from_simd_sads(patch, sads)
}
