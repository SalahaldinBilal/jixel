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
use core::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub(crate) fn ssim_moments_wasm(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) -> usize {
    src.assert_len(l.len());
    assert_eq!(l.len(), cs.len());
    let done = moments::<4>(src, l, cs);
    done + moments::<1>(&src.tail(done), &mut l[done..], &mut cs[done..])
}

#[inline]
#[target_feature(enable = "simd128")]
fn moments<const N: usize>(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let two = f32x4_splat(2.0);
    let c1 = f32x4_splat(C1);
    let c2 = f32x4_splat(C2);
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
                let a = v128_load(src.mx.as_ptr().add(offset).cast());
                let b = v128_load(src.my.as_ptr().add(offset).cast());
                let xx = v128_load(src.sxx.as_ptr().add(offset).cast());
                let yy = v128_load(src.syy.as_ptr().add(offset).cast());
                let xy = v128_load(src.sxy.as_ptr().add(offset).cast());
                let aa = f32x4_mul(a, a);
                let bb = f32x4_mul(b, b);
                let ab = f32x4_mul(a, b);
                let vx = f32x4_sub(xx, aa);
                let vy = f32x4_sub(yy, bb);
                let cov = f32x4_sub(xy, ab);
                let l_num = f32x4_add(f32x4_mul(f32x4_mul(two, a), b), c1);
                let l_den = f32x4_add(f32x4_add(aa, bb), c1);
                let cs_num = f32x4_add(f32x4_mul(two, cov), c2);
                let cs_den = f32x4_add(f32x4_add(vx, vy), c2);
                v128_store(
                    l.as_mut_ptr().add(lane * LANES).cast(),
                    f32x4_div(l_num, l_den),
                );
                v128_store(
                    cs.as_mut_ptr().add(lane * LANES).cast(),
                    f32x4_div(cs_num, cs_den),
                );
            }
        }
        done += N * LANES;
    }
    done
}

#[target_feature(enable = "simd128")]
pub(crate) fn downsample_rows_wasm(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    assert!(top.len() / 2 >= dst.len());
    assert!(bottom.len() / 2 >= dst.len());
    let done = downsample::<4>(top, bottom, dst);
    done + downsample::<1>(&top[2 * done..], &bottom[2 * done..], &mut dst[done..])
}

#[inline]
#[target_feature(enable = "simd128")]
fn downsample<const N: usize>(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let quarter = f32x4_splat(0.25);
    let mut done = 0;
    for (block, dst) in dst.chunks_exact_mut(N * LANES).enumerate() {
        let x = block * N * LANES;
        for lane in 0..N {
            // Each output vector consumes two input vectors from each row.
            // Entry-point checks cover every load; stores fit this chunk.
            // Sum the top pair, then each bottom value to keep scalar rounding.
            unsafe {
                let offset = 2 * (x + lane * LANES);
                let t0 = v128_load(top.as_ptr().add(offset).cast());
                let t1 = v128_load(top.as_ptr().add(offset + LANES).cast());
                let b0 = v128_load(bottom.as_ptr().add(offset).cast());
                let b1 = v128_load(bottom.as_ptr().add(offset + LANES).cast());
                let a = i32x4_shuffle::<0, 2, 4, 6>(t0, t1);
                let b = i32x4_shuffle::<1, 3, 5, 7>(t0, t1);
                let c = i32x4_shuffle::<0, 2, 4, 6>(b0, b1);
                let d = i32x4_shuffle::<1, 3, 5, 7>(b0, b1);
                let sum = f32x4_add(f32x4_add(f32x4_add(a, b), c), d);
                v128_store(
                    dst.as_mut_ptr().add(lane * LANES).cast(),
                    f32x4_mul(sum, quarter),
                );
            }
        }
        done += N * LANES;
    }
    done
}
