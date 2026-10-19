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
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn noise_laplacian_stats_neon(patch: &[f32; 64]) -> [f32; 2] {
    let zero = vdupq_n_f32(0.0);
    let quarter = vdupq_n_f32(-0.25);
    let five = vdupq_n_f32(5.0);
    let mut rows = [[zero; 6]; 8];
    for (src, dst) in patch.as_chunks::<8>().0.iter().zip(&mut rows) {
        let lo = unsafe { vld1q_f32(src.as_ptr()) };
        let hi = unsafe { vld1q_f32(src.as_ptr().add(4)) };
        // Left / center / right, with mirrors [1, 0, ...] and [..., 7, 6].
        *dst = [
            vextq_f32::<3>(vdupq_laneq_f32::<1>(lo), lo),
            vextq_f32::<3>(lo, hi),
            lo,
            hi,
            vextq_f32::<1>(lo, hi),
            vextq_f32::<1>(hi, vdupq_laneq_f32::<2>(hi)),
        ];
    }
    let mut abs = [0.0; 64];
    for y in 0..8 {
        let top = &rows[if y == 0 { 1 } else { y - 1 }];
        let center = &rows[y];
        let bottom = &rows[if y == 7 { 6 } else { y + 1 }];
        // Match the scalar tap order without fused multiply-adds.
        for half in 0..2 {
            let mut filtered = vaddq_f32(zero, vmulq_f32(quarter, top[half]));
            filtered = vsubq_f32(filtered, top[2 + half]);
            filtered = vaddq_f32(filtered, vmulq_f32(quarter, top[4 + half]));
            filtered = vsubq_f32(filtered, center[half]);
            filtered = vaddq_f32(filtered, vmulq_f32(five, center[2 + half]));
            filtered = vsubq_f32(filtered, center[4 + half]);
            filtered = vaddq_f32(filtered, vmulq_f32(quarter, bottom[half]));
            filtered = vsubq_f32(filtered, bottom[2 + half]);
            filtered = vaddq_f32(filtered, vmulq_f32(quarter, bottom[4 + half]));
            unsafe { vst1q_f32(abs.as_mut_ptr().add(y * 8 + half * 4), vabsq_f32(filtered)) };
        }
    }
    crate::noise::noise_laplacian_stats_from_abs(patch, &abs)
}

#[target_feature(enable = "neon")]
pub(crate) fn noise_shrink_neon(coeffs: &mut [f32; 64], t2: f32) {
    if t2 <= 0.0 || t2.is_nan() {
        return;
    }
    let threshold = vdupq_n_f32(t2);
    let zero = vdupq_n_f32(0.0);
    for (p, mask) in coeffs
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(NOISE_AC_MASK.as_chunks::<4>().0)
    {
        let c = unsafe { vld1q_f32(p.as_ptr()) };
        let c2 = vmulq_f32(c, c);
        let gain = vdivq_f32(vsubq_f32(c2, threshold), c2);
        let shrunk = vmulq_f32(c, vmaxnmq_f32(gain, zero));
        let mask = unsafe { vld1q_u32(mask.as_ptr()) };
        unsafe { vst1q_f32(p.as_mut_ptr(), vbslq_f32(mask, shrunk, c)) };
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn noise_texture_score_neon(patch: &[f32; 64]) -> f32 {
    let mut sads = [0.0; 20];
    for yb in 0..4 {
        let mut sad = vdupq_n_f32(0.0);
        for cy in 0..4 {
            for cx in 0..3 {
                let center = vdupq_n_f32(patch[(2 + cy) * 8 + 2 + cx]);
                let pixels = unsafe { vld1q_f32(patch.as_ptr().add((yb + cy) * 8 + cx)) };
                sad = vaddq_f32(sad, vabsq_f32(vsubq_f32(center, pixels)));
            }
        }
        unsafe { vst1q_f32(sads.as_mut_ptr().add(yb * 5), sad) };
    }
    texture_score_from_simd_sads(patch, sads)
}
