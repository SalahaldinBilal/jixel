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

use crate::xyb::quantize_xyb_channels;

use crate::image::Image3F;
use crate::quant_weights::INV_DC_QUANT;

#[test]
fn slow_lossless_encode_of_repetitive_image() {
    // Highly repetitive => lz_has_repetition() is true => deep path runs.
    let (w, h) = (512usize, 512usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = (((x / 8) + (y / 8)) % 2) as u8 * 255;
            let i = (y * w + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    let cfg = crate::EncodeConfig::default()
        .with_lossless(true)
        .with_speed(crate::Speed::Slow)
        .with_num_threads(1);
    let out = crate::encode_image(&rgb, w, h, &cfg).expect("encode failed");
    eprintln!("PROBE encoded {} bytes", out.len());
    assert!(!out.is_empty());
}

#[test]
fn packed_quantization_matches_the_channel_formula() {
    let mut atlas = Image3F::new(5, 3);
    for i in 0..15 {
        atlas.plane_row_mut(0, i / 5)[i % 5] = (i as f32 - 7.0) / 41.0;
        atlas.plane_row_mut(1, i / 5)[i % 5] = i as f32 / 29.0;
        atlas.plane_row_mut(2, i / 5)[i % 5] = (15 - i) as f32 / 31.0;
    }

    for lattice_scale in [1, 2, 8] {
        let got = quantize_xyb_channels(&atlas, lattice_scale);
        let m = lattice_scale as f32;
        for i in 0..15 {
            let yq = (atlas.plane_data(1)[i] * INV_DC_QUANT[1] * m).round() as i32;
            assert_eq!(got[0][i], yq);
            assert_eq!(
                got[1][i],
                (atlas.plane_data(0)[i] * INV_DC_QUANT[0] * m).round() as i32
            );
            assert_eq!(
                got[2][i],
                (atlas.plane_data(2)[i] * INV_DC_QUANT[2] * m).round() as i32 - yq
            );
        }
    }
}
