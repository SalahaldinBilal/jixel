/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

//! Tests for squeeze.

use super::extra::ExtraSqueeze;
use crate::encode_image::AlphaPlane;

/// A soft-edged disc: the anti-aliased, feathered alpha lossy coding is for.
fn soft_disc(w: usize, h: usize) -> AlphaPlane {
    AlphaPlane::U8(
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let r = ((x - w as f32 / 2.0).hypot(y - h as f32 / 2.0)) / (w.min(h) as f32);
                (255.0 * ((0.45 - r) / 0.2).clamp(0.0, 1.0)).round() as u8
            })
            .collect(),
    )
}

#[test]
fn unit_steps_reconstruct_the_plane_exactly() {
    let (w, h) = (300, 270);
    let plane = soft_disc(w, h);
    let squeezed = ExtraSqueeze::new(&plane, w, h, 1e-3, false).unwrap();
    assert!(squeezed.quants.iter().all(|&q| q == 1));
    let want: Vec<i32> = (0..w * h).map(|i| plane.get_i32(i)).collect();
    assert_eq!(squeezed.reconstruct(), want);
}

#[test]
fn quantized_reconstruction_stays_close() {
    let (w, h) = (300, 270);
    let plane = soft_disc(w, h);
    let squeezed = ExtraSqueeze::new(&plane, w, h, 4.0, false).unwrap();
    assert!(squeezed.quants.iter().any(|&q| q > 1));
    let rec = squeezed.reconstruct();
    let err: Vec<i32> = rec
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.clamp(0, 255) - plane.get_i32(i)).abs())
        .collect();
    let mean = err.iter().sum::<i32>() as f32 / err.len() as f32;
    assert!(err.iter().any(|&e| e > 0), "distance 4 should be lossy");
    assert!(mean < 2.0, "mean alpha error {mean}");
}
