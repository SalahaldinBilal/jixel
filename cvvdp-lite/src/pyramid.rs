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

//! Separable five-tap Gaussian reduction and sparse expansion. Reduction
//! reflects including edge samples; expansion replicates coarse edge samples,
//! matching the original's zero interleaving and explicit padding.

pub(crate) fn reduce(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, tmp: &mut Vec<f32>) {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    ensure_len(tmp, w * nh);
    ensure_len(dst, nw * nh);
    for y in 0..nh {
        let rows =
            std::array::from_fn::<_, 5, _>(|k| reflect((2 * y) as isize + k as isize - 2, h) * w);
        for x in 0..w {
            tmp[y * w + x] = 0.05 * (src[rows[0] + x] + src[rows[4] + x])
                + 0.25 * (src[rows[1] + x] + src[rows[3] + x])
                + 0.4 * src[rows[2] + x];
        }
    }
    for y in 0..nh {
        let row = &tmp[y * w..(y + 1) * w];
        for x in 0..nw {
            let xx = (2 * x) as isize;
            dst[y * nw + x] = 0.05 * (row[reflect(xx - 2, w)] + row[reflect(xx + 2, w)])
                + 0.25 * (row[reflect(xx - 1, w)] + row[reflect(xx + 1, w)])
                + 0.4 * row[2 * x];
        }
    }
}

pub(crate) fn expand(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, tmp: &mut Vec<f32>) {
    let (sw, sh) = (w.div_ceil(2), h.div_ceil(2));
    ensure_len(tmp, sw * h);
    ensure_len(dst, w * h);
    for y in 0..h {
        let sy = y / 2;
        let before = sy.saturating_sub(1) * sw;
        let center = sy * sw;
        let after = (sy + 1).min(sh - 1) * sw;
        for x in 0..sw {
            tmp[y * sw + x] = if y % 2 == 0 {
                0.1 * (src[before + x] + src[after + x]) + 0.8 * src[center + x]
            } else {
                0.5 * (src[center + x] + src[after + x])
            };
        }
    }
    for y in 0..h {
        let row = &tmp[y * sw..(y + 1) * sw];
        for x in 0..w {
            let sx = x / 2;
            dst[y * w + x] = if x % 2 == 0 {
                0.1 * (row[sx.saturating_sub(1)] + row[(sx + 1).min(sw - 1)]) + 0.8 * row[sx]
            } else {
                0.5 * (row[sx] + row[(sx + 1).min(sw - 1)])
            };
        }
    }
}

fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    if i < 0 {
        (-i - 1) as usize
    } else if i >= n as isize {
        (2 * n) - i as usize - 1
    } else {
        i as usize
    }
}

pub(crate) fn ensure_len(buffer: &mut Vec<f32>, length: usize) {
    if buffer.len() < length {
        buffer.resize(length, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_odd_even_and_degenerate_pyramids() {
        for (w, h) in [(1, 1), (1, 9), (10, 1), (3, 4), (4, 3), (63, 64), (64, 63)] {
            let src = vec![1.0; w * h];
            let (mut reduced, mut expanded, mut tmp) = (Vec::new(), Vec::new(), Vec::new());
            reduce(&src, w, h, &mut reduced, &mut tmp);
            expand(&reduced, w, h, &mut expanded, &mut tmp);
            assert!(
                reduced
                    .iter()
                    .chain(&expanded)
                    .all(|v| (*v - 1.0).abs() < 1e-6)
            );
        }
    }

    #[test]
    fn symmetric_reduce_duplicates_right_edge() {
        let mut reduced = Vec::new();
        reduce(
            &[0.0, 0.0, 0.0, 1.0, 1.0],
            5,
            1,
            &mut reduced,
            &mut Vec::new(),
        );
        assert!((reduced[2] - 0.95).abs() < 1e-6);
    }
}
