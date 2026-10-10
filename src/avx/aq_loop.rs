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
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn ssim_moments_avx2(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) -> usize {
    src.assert_len(l.len());
    assert_eq!(l.len(), cs.len());
    let done = moments::<4>(src, l, cs);
    done + moments::<1>(&src.tail(done), &mut l[done..], &mut cs[done..])
}

#[inline]
#[target_feature(enable = "avx2")]
fn moments<const N: usize>(src: &SsimMoments<'_>, l: &mut [f32], cs: &mut [f32]) -> usize {
    const LANES: usize = 8;
    let two = _mm256_set1_ps(2.0);
    let c1 = _mm256_set1_ps(C1);
    let c2 = _mm256_set1_ps(C2);
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
                let a = _mm256_loadu_ps(src.mx.as_ptr().add(offset));
                let b = _mm256_loadu_ps(src.my.as_ptr().add(offset));
                let xx = _mm256_loadu_ps(src.sxx.as_ptr().add(offset));
                let yy = _mm256_loadu_ps(src.syy.as_ptr().add(offset));
                let xy = _mm256_loadu_ps(src.sxy.as_ptr().add(offset));
                let aa = _mm256_mul_ps(a, a);
                let bb = _mm256_mul_ps(b, b);
                let ab = _mm256_mul_ps(a, b);
                let vx = _mm256_sub_ps(xx, aa);
                let vy = _mm256_sub_ps(yy, bb);
                let cov = _mm256_sub_ps(xy, ab);
                let l_num = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(two, a), b), c1);
                let l_den = _mm256_add_ps(_mm256_add_ps(aa, bb), c1);
                let cs_num = _mm256_add_ps(_mm256_mul_ps(two, cov), c2);
                let cs_den = _mm256_add_ps(_mm256_add_ps(vx, vy), c2);
                _mm256_storeu_ps(
                    l.as_mut_ptr().add(lane * LANES),
                    _mm256_div_ps(l_num, l_den),
                );
                _mm256_storeu_ps(
                    cs.as_mut_ptr().add(lane * LANES),
                    _mm256_div_ps(cs_num, cs_den),
                );
            }
        }
        done += N * LANES;
    }
    done
}

#[target_feature(enable = "avx2")]
pub(crate) fn downsample_rows_avx2(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    assert!(top.len() / 2 >= dst.len());
    assert!(bottom.len() / 2 >= dst.len());
    let done = downsample::<4>(top, bottom, dst);
    done + downsample::<1>(&top[2 * done..], &bottom[2 * done..], &mut dst[done..])
}

#[inline]
#[target_feature(enable = "avx2")]
fn downsample<const N: usize>(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    const LANES: usize = 8;
    let quarter = _mm256_set1_ps(0.25);
    let mut done = 0;
    for (block, dst) in dst.chunks_exact_mut(N * LANES).enumerate() {
        let x = block * N * LANES;
        for lane in 0..N {
            // Each output vector consumes two input vectors from each row.
            // Entry-point checks cover every load; stores fit this chunk.
            // Sum the top pair, then each bottom value to keep scalar rounding.
            unsafe {
                let offset = 2 * (x + lane * LANES);
                let t0 = _mm256_loadu_ps(top.as_ptr().add(offset));
                let t1 = _mm256_loadu_ps(top.as_ptr().add(offset + LANES));
                let b0 = _mm256_loadu_ps(bottom.as_ptr().add(offset));
                let b1 = _mm256_loadu_ps(bottom.as_ptr().add(offset + LANES));
                let top_pairs = _mm256_hadd_ps(t0, t1);
                let c = _mm256_shuffle_ps::<0x88>(b0, b1);
                let d = _mm256_shuffle_ps::<0xdd>(b0, b1);
                let sum = _mm256_add_ps(_mm256_add_ps(top_pairs, c), d);
                // Horizontal add/shuffle operate within 128-bit lanes:
                // reorder output pairs [0, 1, 4, 5, 2, 3, 6, 7] to scan order.
                let value = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xd8>(
                    _mm256_castps_si256(_mm256_mul_ps(sum, quarter)),
                ));
                _mm256_storeu_ps(dst.as_mut_ptr().add(lane * LANES), value);
            }
        }
        done += N * LANES;
    }
    done
}
