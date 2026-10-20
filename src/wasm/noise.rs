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
use std::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub(crate) fn noise_laplacian_stats_wasm(patch: &[f32; 64]) -> [f32; 2] {
    let zero = f32x4_splat(0.0);
    let quarter = f32x4_splat(-0.25);
    let five = f32x4_splat(5.0);
    let mut rows = [[zero; 6]; 8];
    for (src, dst) in patch.as_chunks::<8>().0.iter().zip(&mut rows) {
        let lo = unsafe { v128_load(src.as_ptr().cast()) };
        let hi = unsafe { v128_load(src.as_ptr().add(4).cast()) };
        // Left / center / right, with mirrors [1, 0, ...] and [..., 7, 6].
        *dst = [
            i32x4_shuffle::<1, 0, 1, 2>(lo, lo),
            i32x4_shuffle::<3, 4, 5, 6>(lo, hi),
            lo,
            hi,
            i32x4_shuffle::<1, 2, 3, 4>(lo, hi),
            i32x4_shuffle::<1, 2, 3, 2>(hi, hi),
        ];
    }
    let mut abs = [0.0; 64];
    for y in 0..8 {
        let top = &rows[if y == 0 { 1 } else { y - 1 }];
        let center = &rows[y];
        let bottom = &rows[if y == 7 { 6 } else { y + 1 }];
        // Match the scalar tap order without fused multiply-adds.
        for half in 0..2 {
            let mut filtered = f32x4_add(zero, f32x4_mul(quarter, top[half]));
            filtered = f32x4_sub(filtered, top[2 + half]);
            filtered = f32x4_add(filtered, f32x4_mul(quarter, top[4 + half]));
            filtered = f32x4_sub(filtered, center[half]);
            filtered = f32x4_add(filtered, f32x4_mul(five, center[2 + half]));
            filtered = f32x4_sub(filtered, center[4 + half]);
            filtered = f32x4_add(filtered, f32x4_mul(quarter, bottom[half]));
            filtered = f32x4_sub(filtered, bottom[2 + half]);
            filtered = f32x4_add(filtered, f32x4_mul(quarter, bottom[4 + half]));
            unsafe {
                v128_store(
                    abs.as_mut_ptr().add(y * 8 + half * 4).cast(),
                    f32x4_abs(filtered),
                )
            };
        }
    }
    crate::noise::noise_laplacian_stats_from_abs(patch, &abs)
}

#[target_feature(enable = "simd128")]
pub(crate) fn noise_shrink_wasm(coeffs: &mut [f32; 64], t2: f32) {
    if t2 <= 0.0 || t2.is_nan() {
        return;
    }
    let threshold = f32x4_splat(t2);
    let zero = f32x4_splat(0.0);
    for (p, mask) in coeffs
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(NOISE_AC_MASK.as_chunks::<4>().0)
    {
        let c = unsafe { v128_load(p.as_ptr().cast()) };
        let c2 = f32x4_mul(c, c);
        let gain = f32x4_div(f32x4_sub(c2, threshold), c2);
        let shrunk = f32x4_mul(c, v128_bitselect(gain, zero, f32x4_gt(gain, zero)));
        let mask = unsafe { v128_load(mask.as_ptr().cast()) };
        unsafe { v128_store(p.as_mut_ptr().cast(), v128_bitselect(shrunk, c, mask)) };
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn noise_texture_score_wasm(patch: &[f32; 64]) -> f32 {
    let mut sads = [0.0; 20];
    for yb in 0..4 {
        let mut sad = f32x4_splat(0.0);
        for cy in 0..4 {
            for cx in 0..3 {
                let center = f32x4_splat(patch[(2 + cy) * 8 + 2 + cx]);
                let pixels = unsafe { v128_load(patch.as_ptr().add((yb + cy) * 8 + cx).cast()) };
                sad = f32x4_add(sad, f32x4_abs(f32x4_sub(center, pixels)));
            }
        }
        unsafe { v128_store(sads.as_mut_ptr().add(yb * 5).cast(), sad) };
    }
    texture_score_from_simd_sads(patch, sads)
}
