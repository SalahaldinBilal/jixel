// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Copyright (c) Radzivon Bartoshyk 2026. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//! Invisible-color fill adapted from libjxl's `SimplifyInvisible` in
//! `lib/jxl/enc_frame.cc`. Applied to unpadded color planes, never to alpha.

use crate::encode_image::AlphaPlane;
use crate::image::Image3F;
use std::cell::Cell;

pub(crate) fn fill_invisible(image: &mut Image3F, alpha: &AlphaPlane) {
    debug_assert_eq!(alpha.len(), image.xsize() * image.ysize());
    match alpha {
        AlphaPlane::U8(data) => fill_with_alpha(image, data, f32::from),
        AlphaPlane::U16 { data, .. } => fill_with_alpha(image, data, f32::from),
        AlphaPlane::F32(data) => fill_with_alpha(image, data, |a| f32::from_bits(a as u32)),
    }
}

fn fill_with_alpha<T: Copy>(image: &mut Image3F, alpha: &[T], opacity: impl Fn(T) -> f32) {
    let width = image.xsize();
    if width == 0 || alpha.iter().all(|&a| opacity(a) != 0.0) {
        return;
    }
    // Keep libjxl's in-place, top-to-bottom / left-to-right traversal: the
    // filled left and upper-right samples propagate color into empty regions.
    for c in 0..3 {
        // Cell lets neighbor iterators share references while pixels are
        // updated in place, without allocation or unsafe code.
        let pixels = Cell::from_mut(image.plane_mut(c).as_mut_slice()).as_slice_of_cells();
        let mut rows = pixels
            .chunks_exact(width)
            .zip(alpha.chunks_exact(width))
            .peekable();
        let mut previous = None;
        while let Some(row) = rows.next() {
            // Reuse the current row for missing neighbors; the const flags
            // remove their contributions and vertical edge checks from the loop.
            match (previous, rows.peek().copied()) {
                (Some(up), Some(down)) => fill_row::<_, true, true>(row, up, down, &opacity),
                (Some(up), None) => fill_row::<_, true, false>(row, up, row, &opacity),
                (None, Some(down)) => fill_row::<_, false, true>(row, row, down, &opacity),
                (None, None) => fill_row::<_, false, false>(row, row, row, &opacity),
            }
            previous = Some(row);
        }
    }
}

#[inline]
fn fill_row<'a, T: Copy, const UP: bool, const DOWN: bool>(
    row: (&'a [Cell<f32>], &[T]),
    up: (&'a [Cell<f32>], &[T]),
    down: (&'a [Cell<f32>], &[T]),
    opacity: &impl Fn(T) -> f32,
) {
    let mut samples = row
        .0
        .iter()
        .zip(row.1.iter().copied())
        .zip(up.0.iter().zip(up.1.iter().copied()))
        .zip(down.0.iter().zip(down.1.iter().copied()))
        .map(|((pixel, up), down)| [pixel, up, down]);
    let Some(mut current) = samples.next() else {
        return;
    };
    let mut left: Option<(&'a Cell<f32>, bool)> = None;
    let mut fill = |[(pixel, a), (up, up_a), (down, down_a)]: [(&'a Cell<f32>, T); 3],
                    right: Option<[(&'a Cell<f32>, T); 3]>| {
        let a = opacity(a);
        if a == 0.0 {
            let mut sum = 0.0;
            let mut weight = 0.0;
            if let Some((value, visible)) = left {
                sum += value.get();
                weight += 1.0;
                if visible {
                    sum += value.get();
                    weight += 1.0;
                }
            }
            if let Some([right, up_right, down_right]) = right {
                let (right, right_a) = right;
                let (up_right, up_right_a) = up_right;
                let (down_right, down_right_a) = down_right;
                if UP {
                    sum += up_right.get();
                    weight += 1.0;
                }
                if opacity(right_a) > 0.0 {
                    sum += 2.0 * right.get();
                    weight += 2.0;
                }
                if UP && opacity(up_right_a) > 0.0 {
                    sum += 2.0 * up_right.get();
                    weight += 2.0;
                }
                if DOWN && opacity(down_right_a) > 0.0 {
                    sum += 2.0 * down_right.get();
                    weight += 2.0;
                }
            }
            if UP && opacity(up_a) > 0.0 {
                sum += 2.0 * up.get();
                weight += 2.0;
            }
            if DOWN && opacity(down_a) > 0.0 {
                sum += 2.0 * down.get();
                weight += 2.0;
            }
            pixel.set(if weight > 1.0 { sum / weight } else { sum });
        }
        left = Some((pixel, a > 0.0));
    };
    // Every iteration has a right neighbor; handle the last pixel separately.
    for right in samples {
        fill(current, Some(right));
        current = right;
    }
    fill(current, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize, values: &[f32]) -> Image3F {
        let mut image = Image3F::new(width, height);
        for c in 0..3 {
            image.plane_mut(c).as_mut_slice().copy_from_slice(values);
        }
        image
    }

    #[test]
    fn fills_only_zero_alpha_and_propagates_in_scan_order() {
        // The lower-middle pixel gives the upper-right neighbor weight 3
        // and its visible left neighbor weight 2. The last pixel then uses
        // the already-filled left neighbor (weight 1) and the top (weight 2).
        let mut image = image(3, 2, &[2.0, 99.0, 4.0, 6.0, 99.0, 99.0]);
        let alpha = AlphaPlane::U8(vec![255, 0, 1, 128, 0, 0]);
        fill_invisible(&mut image, &alpha);
        for c in 0..3 {
            let expected = [2.0, 3.0, 4.0, 6.0, 4.8, 4.266_667];
            for (&got, want) in image.plane_data(c).iter().zip(expected) {
                assert!((got - want).abs() < 1e-6, "{got} != {want}");
            }
        }
        assert_eq!(alpha.get_i32(4), 0);
        assert_eq!(alpha.get_i32(3), 128);
    }

    #[test]
    fn single_rows_columns_and_fully_transparent_images() {
        for (width, height, alpha, source, expected) in [
            (0, 0, vec![], vec![], vec![]),
            (0, 3, vec![], vec![], vec![]),
            (3, 0, vec![], vec![], vec![]),
            (1, 1, vec![0], vec![99.0], vec![0.0]),
            (3, 1, vec![0, 1, 0], vec![99.0, 2.0, 99.0], vec![2.0; 3]),
            (1, 3, vec![0, 1, 0], vec![99.0, 2.0, 99.0], vec![2.0; 3]),
            (2, 2, vec![0; 4], vec![99.0; 4], vec![0.0; 4]),
        ] {
            let mut image = image(width, height, &source);
            fill_invisible(&mut image, &AlphaPlane::U8(alpha));
            for c in 0..3 {
                assert_eq!(image.plane_data(c), expected);
            }
        }
    }

    #[test]
    fn float_alpha_distinguishes_zero_from_positive_opacity() {
        let mut image = image(3, 2, &[10.0, 20.0, 99.0, 40.0, 99.0, 60.0]);
        let alpha = AlphaPlane::F32(
            [-0.5f32, f32::NAN, -0.0, f32::INFINITY, 0.0, 0.25]
                .map(|a| a.to_bits() as i32)
                .to_vec(),
        );
        fill_invisible(&mut image, &alpha);
        // Negative and NaN alpha stay untouched and contribute no visible
        // weight. Negative zero fills, and infinity counts as positive opacity.
        let expected = [
            10.0,
            20.0,
            140.0 / 3.0,
            40.0,
            (80.0 + 140.0 / 3.0 + 120.0) / 5.0,
            60.0,
        ];
        for c in 0..3 {
            assert_eq!(image.plane_data(c), expected);
        }
    }

    #[test]
    fn high_depth_and_float_alpha_use_opacity_not_storage_bits() {
        let mut reference = image(3, 1, &[99.0, 2.0, 99.0]);
        fill_invisible(&mut reference, &AlphaPlane::U8(vec![0, 1, 0]));
        for alpha in [
            AlphaPlane::U16 {
                data: vec![0, 1, 0],
                bits: 16,
            },
            AlphaPlane::F32([0.0f32, 0.5, -0.0].map(|v| v.to_bits() as i32).to_vec()),
        ] {
            let mut image = image(3, 1, &[99.0, 2.0, 99.0]);
            fill_invisible(&mut image, &alpha);
            for c in 0..3 {
                assert_eq!(image.plane_data(c), reference.plane_data(c));
            }
        }
    }
}
