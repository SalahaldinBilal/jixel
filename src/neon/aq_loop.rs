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
use crate::aq_loop::{C1, C2, SsimMoments};
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn ssim_moments_neon(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) -> usize {
    src.assert_len(l.len());
    assert_eq!(l.len(), cs.len());
    let done = moments::<4>(src, l, cs);
    done + moments::<1>(&src.tail(done), &mut l[done..], &mut cs[done..])
}

#[inline]
#[target_feature(enable = "neon")]
fn moments<const N: usize>(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let two = vdupq_n_f32(2.0);
    let c1 = vdupq_n_f32(C1);
    let c2 = vdupq_n_f32(C2);
    let mut done = 0;
    for (block, (l, cs)) in l
        .chunks_exact_mut(N * LANES)
        .zip(cs.chunks_exact_mut(N * LANES))
        .enumerate()
    {
        let x = block * N * LANES;
        for lane in 0..N {
            // The entry point checks all lengths; loads and stores cover
            // only complete vectors. Keep the scalar order without FMA.
            unsafe {
                let offset = x + lane * LANES;
                let a = vld1q_f32(src.mx.as_ptr().add(offset));
                let b = vld1q_f32(src.my.as_ptr().add(offset));
                let xx = vld1q_f32(src.sxx.as_ptr().add(offset));
                let yy = vld1q_f32(src.syy.as_ptr().add(offset));
                let xy = vld1q_f32(src.sxy.as_ptr().add(offset));
                let aa = vmulq_f32(a, a);
                let bb = vmulq_f32(b, b);
                let ab = vmulq_f32(a, b);
                let vx = vsubq_f32(xx, aa);
                let vy = vsubq_f32(yy, bb);
                let cov = vsubq_f32(xy, ab);
                let l_num = vaddq_f32(vmulq_f32(vmulq_f32(two, a), b), c1);
                let l_den = vaddq_f32(vaddq_f32(aa, bb), c1);
                let cs_num = vaddq_f32(vmulq_f32(two, cov), c2);
                let cs_den = vaddq_f32(vaddq_f32(vx, vy), c2);
                vst1q_f32(l.as_mut_ptr().add(lane * LANES), vdivq_f32(l_num, l_den));
                vst1q_f32(cs.as_mut_ptr().add(lane * LANES), vdivq_f32(cs_num, cs_den));
            }
        }
        done += N * LANES;
    }
    done
}

#[target_feature(enable = "neon")]
pub(crate) fn downsample_rows_neon(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    assert!(top.len() / 2 >= dst.len());
    assert!(bottom.len() / 2 >= dst.len());
    let done = downsample::<2>(top, bottom, dst);
    done + downsample::<1>(&top[2 * done..], &bottom[2 * done..], &mut dst[done..])
}

#[inline]
#[target_feature(enable = "neon")]
fn downsample<const N: usize>(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let quarter = vdupq_n_f32(0.25);
    let mut done = 0;
    for (block, dst) in dst.chunks_exact_mut(N * LANES).enumerate() {
        let x = block * N * LANES;
        for lane in 0..N {
            // Each output vector consumes two input vectors from each row.
            // Entry-point checks cover every load; stores fit this chunk.
            // Sum the top pair, then each bottom value to keep scalar rounding.
            unsafe {
                let offset = 2 * (x + lane * LANES);
                let t0 = vld1q_f32(top.as_ptr().add(offset));
                let t1 = vld1q_f32(top.as_ptr().add(offset + LANES));
                let b0 = vld1q_f32(bottom.as_ptr().add(offset));
                let b1 = vld1q_f32(bottom.as_ptr().add(offset + LANES));
                let top_pairs = vpaddq_f32(t0, t1);
                let c = vuzp1q_f32(b0, b1);
                let d = vuzp2q_f32(b0, b1);
                let sum = vaddq_f32(vaddq_f32(top_pairs, c), d);
                vst1q_f32(dst.as_mut_ptr().add(lane * LANES), vmulq_f32(sum, quarter));
            }
        }
        done += N * LANES;
    }
    done
}
