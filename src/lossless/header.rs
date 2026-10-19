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

//! Modular frame headers, transforms, patch dictionaries, and table of contents.

use super::GroupLayout;
use super::rct::write_rct_transform;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{Token, optimize_entropy_code, pack_signed, write_entropy_code, write_token};
use crate::patches::{ModularFrameKind, NUM_PATCH_CONTEXTS, PATCH_REF_ID, PatchReference};

pub(super) fn write_u64(value: u64, w: &mut BitWriter) {
    match value {
        0 => w.write(2, 0),
        1..=16 => {
            w.write(2, 1);
            w.write(4, value - 1);
        }
        17..=272 => {
            w.write(2, 2);
            w.write(8, value - 17);
        }
        _ => unreachable!("lossless frame flags fit the short U64 form"),
    }
}

fn write_frame_dimension(value: usize, w: &mut BitWriter) {
    let value = value as u64;
    if value < 256 {
        w.write(2, 0);
        w.write(8, value);
    } else if value < 2304 {
        w.write(2, 1);
        w.write(11, value - 256);
    } else if value < 18688 {
        w.write(2, 2);
        w.write(14, value - 2304);
    } else {
        w.write(2, 3);
        w.write(30, value - 18688);
    }
}

pub(super) fn write_frame_header_modular_kind(
    has_alpha: bool,
    kind: ModularFrameKind<'_>,
    layout: GroupLayout,
    w: &mut BitWriter,
) {
    match kind {
        ModularFrameKind::Regular => write_frame_header_modular(has_alpha, layout, w),
        ModularFrameKind::RgbSavedFirst => {
            write_frame_header_modular_layer(has_alpha, false, layout, w)
        }
        ModularFrameKind::RgbBlendLast => {
            write_frame_header_modular_layer(has_alpha, true, layout, w)
        }
        ModularFrameKind::Patched { saved_first, .. } => {
            write_frame_header_modular_flags(has_alpha, 2, saved_first, layout, w)
        }
        ModularFrameKind::RgbCrop {
            x0,
            y0,
            width,
            height,
            last,
        } => write_frame_header_modular_crop(has_alpha, [x0, y0, width, height], layout, last, w),
        ModularFrameKind::XybReferenceOnly {
            width,
            height,
            slot,
        } => write_frame_header_modular_xyb_reference(width, height, slot, has_alpha, layout, w),
        ModularFrameKind::XybBlend { last } => {
            write_frame_header_modular_xyb_blend(has_alpha, last, layout, w)
        }
        ModularFrameKind::ReferenceOnly { width, height } => {
            w.write(1, 0); // all_default = false
            w.write(2, 0b10); // reference-only frame
            w.write(1, 1); // Modular
            write_u64(0, w); // flags
            w.write(1, 0); // color transform = None
            w.write(2, 0); // upsampling = 1
            if has_alpha {
                w.write(2, 0); // extra-channel upsampling = 1
            }
            w.write(2, layout.shift as u64);
            // Reference-only frames do not serialize Passes.
            w.write(1, 1); // custom size
            write_frame_dimension(width, w);
            write_frame_dimension(height, w);
            // No blending and no is_last for reference-only frames.
            w.write(2, PATCH_REF_ID as u64); // save_as_reference = 3
            w.write(1, 1); // save_before_color_transform = true
            w.write(2, 0); // empty name
            w.write(1, 0); // loop filter not all-default
            w.write(1, 0); // no gaborish
            w.write(2, 0); // no EPF
            w.write(2, 0); // no loop-filter extensions
            w.write(2, 0); // no frame-header extensions
        }
    }
}

/// Frame header of the ink frame: a regular modular frame of the lossy
/// (`xyb_encoded`) codestream, kBlend over reference slot 0 with extra
/// channel 0 as its alpha, for the colour channels and the alpha itself.
fn write_frame_header_modular_xyb_blend(
    has_alpha: bool,
    last: bool,
    layout: GroupLayout,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    // No do_YCbCr bit: with xyb_encoded set the color transform is XYB.
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00); // ec_upsampling[0] = 1
    }
    w.write(2, layout.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    for _ in 0..1 + usize::from(has_alpha) {
        w.write(2, 2); // mode = kBlend
        w.write(2, 0); // alpha_channel = 0
        w.write(1, 0); // clamp = false
        w.write(2, 0); // source = 0
    }
    if last {
        w.write(1, 1); // is_last
    } else {
        w.write(1, 0); // not the last frame
        // The blended canvas replaces reference slot 0 for the next ink
        // frame; kBlend does not reset the canvas, so no save_before_ct bit.
        w.write(2, 0); // save_as_reference = 0
    }
    w.write(2, 0b00); // name length = 0
    w.write(1, 0); // loop_filter NOT all_default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // 0 EPF iters
    w.write(2, 0b00); // no LF extensions
    w.write(2, 0b00); // no FH extensions
}

/// Frame header for a modular reference-only frame inside the lossy
/// (`xyb_encoded`) codestream. Identical to the lossless variant except that
/// the container sets `xyb_encoded`, so the do_YCbCr bit is absent and the
/// frame's color transform is implicitly XYB. Saved to the modular slot so a
/// hybrid plan can also emit the VarDCT atlas in slot 3.
pub(super) fn write_frame_header_modular_xyb_reference(
    width: usize,
    height: usize,
    slot: u32,
    has_alpha: bool,
    layout: GroupLayout,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b10); // reference-only frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    // No do_YCbCr bit: with xyb_encoded set the color transform is XYB.
    w.write(2, 0); // upsampling = 1
    if has_alpha {
        w.write(2, 0); // ec_upsampling[0] = 1
    }
    w.write(2, layout.shift as u64);
    // Reference-only frames do not serialize Passes.
    w.write(1, 1); // custom size
    write_frame_dimension(width, w);
    write_frame_dimension(height, w);
    // No blending and no is_last for reference-only frames.
    w.write(2, slot as u64); // save_as_reference
    w.write(1, 1); // save_before_color_transform = true
    w.write(2, 0); // empty name
    // Loop filter off: the atlas is stored exactly as coded; gaborish or EPF
    // would smear the glyph edges patches exist to preserve.
    w.write(1, 0); // loop filter not all-default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // no EPF
    w.write(2, 0); // no loop-filter extensions
    w.write(2, 0); // no frame-header extensions
}

/// Frame header of a lossless layer frame: the palette base saved to
/// reference slot 0 (not last), or the coverage frame blended over it
/// (kBlend with extra channel 0 as alpha, last).
fn write_frame_header_modular_layer(
    has_alpha: bool,
    blend_last: bool,
    layout: GroupLayout,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(1, 0); // do_ycbcr = false
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00); // ec_upsampling[0] = 1
    }
    w.write(2, layout.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    if blend_last {
        for _ in 0..1 + usize::from(has_alpha) {
            w.write(2, 2); // mode = kBlend
            w.write(2, 0); // alpha_channel = 0
            w.write(1, 0); // clamp = false
            w.write(2, 0); // source = 0
        }
        w.write(1, 1); // is_last
    } else {
        w.write(2, 0b00); // blending = Replace
        if has_alpha {
            w.write(2, 0b00); // ec blending = Replace
        }
        w.write(1, 0); // not the last frame
        w.write(2, 0); // save_as_reference = 0
        w.write(1, 0); // save_before_color_transform = false
    }
    w.write(2, 0b00); // name length = 0
    w.write(1, 0); // loop_filter NOT all_default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // 0 EPF iters
    w.write(2, 0b00); // no LF extensions
    w.write(2, 0b00); // no FH extensions
}

pub(super) fn write_frame_header_modular(has_alpha: bool, layout: GroupLayout, w: &mut BitWriter) {
    write_frame_header_modular_flags(has_alpha, 0, false, layout, w);
}

/// Frame header of a region frame: a regular modular frame cropped to
/// `rect` (x0, y0, width, height), kReplace over reference slot 0; the
/// composite goes back to slot 0 unless this is the last frame. A partial
/// frame always signals its source and never `save_before_color_transform`.
fn write_frame_header_modular_crop(
    has_alpha: bool,
    rect: [usize; 4],
    layout: GroupLayout,
    last: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(1, 0); // do_ycbcr = false
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00); // ec_upsampling[0] = 1
    }
    w.write(2, layout.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 1); // have_crop
    write_frame_dimension(2 * rect[0], w); // PackSigned(x0)
    write_frame_dimension(2 * rect[1], w);
    write_frame_dimension(rect[2], w);
    write_frame_dimension(rect[3], w);
    for _ in 0..1 + usize::from(has_alpha) {
        w.write(2, 0b00); // blending = Replace
        w.write(2, 0); // source = 0
    }
    if last {
        w.write(1, 1); // is_last
    } else {
        w.write(1, 0); // not the last frame
        w.write(2, 0); // save_as_reference = 0
    }
    w.write(2, 0b00); // name length = 0
    w.write(1, 0); // loop_filter NOT all_default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // 0 EPF iters
    w.write(2, 0b00); // no LF extensions
    w.write(2, 0b00); // no FH extensions
}

fn write_frame_header_modular_flags(
    has_alpha: bool,
    flags: u64,
    saved_first: bool,
    layout: GroupLayout,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(flags, w);
    w.write(1, 0); // do_ycbcr = false   (xyb_encoded=0 so this is serialized)
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(2, layout.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    if has_alpha {
        w.write(2, 0b00);
    }
    if saved_first {
        w.write(1, 0); // not the last frame
        w.write(2, 0); // save_as_reference = 0
        w.write(1, 0); // save_before_color_transform = false
    } else {
        w.write(1, 1); // is_last
    }
    w.write(2, 0b00); // name length = 0
    w.write(1, 0); // loop_filter NOT all_default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // 0 EPF iters
    w.write(2, 0b00); // no LF extensions
    w.write(2, 0b00); // no FH extensions
}

pub(crate) fn write_patch_dictionary(
    references: &[PatchReference],
    has_alpha: bool,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    const NUM_REF: u32 = 0;
    const REFERENCE_FRAME: u32 = 1;
    const PATCH_SIZE: u32 = 2;
    const REF_POSITION: u32 = 3;
    const POSITION: u32 = 4;
    const BLEND_MODE: u32 = 5;
    const OFFSET: u32 = 6;
    const COUNT: u32 = 7;
    const CLAMP: u32 = 9;

    let mut tokens = Vec::new();
    tokens.push(Token::new(NUM_REF, references.len() as u32));
    for reference in references {
        tokens.push(Token::new(REFERENCE_FRAME, reference.ref_frame));
        tokens.push(Token::new(REF_POSITION, reference.atlas_x as u32));
        tokens.push(Token::new(REF_POSITION, reference.atlas_y as u32));
        tokens.push(Token::new(PATCH_SIZE, (reference.width - 1) as u32));
        tokens.push(Token::new(PATCH_SIZE, (reference.height - 1) as u32));
        tokens.push(Token::new(COUNT, (reference.positions.len() - 1) as u32));
        for (i, &(x, y)) in reference.positions.iter().enumerate() {
            if i == 0 {
                tokens.push(Token::new(POSITION, x as u32));
                tokens.push(Token::new(POSITION, y as u32));
            } else {
                let (px, py) = reference.positions[i - 1];
                tokens.push(Token::new(OFFSET, pack_signed(x as i32 - px as i32)));
                tokens.push(Token::new(OFFSET, pack_signed(y as i32 - py as i32)));
            }
            // color = Mul (3), Add (2) or Replace (1); Mul also signals clamp = false
            tokens.push(Token::new(
                BLEND_MODE,
                if reference.mul {
                    3
                } else if reference.add {
                    2
                } else {
                    1
                },
            ));
            if reference.mul {
                tokens.push(Token::new(CLAMP, 0));
            }
            if has_alpha {
                tokens.push(Token::new(BLEND_MODE, 0)); // alpha = None
            }
        }
    }
    // Dense text makes the dictionary a real share of the frame (a third of
    // a text page): race the prefix code against a config-selected rANS code.
    let mut prefix_bits = BitWriter::new();
    {
        let code = optimize_entropy_code(&tokens, NUM_PATCH_CONTEXTS, &mut scratch.huffman_pool);
        let code_ref = code.as_ref();
        prefix_bits.write(1, 0); // patch dictionary entropy stream has no LZ77
        write_entropy_code(&code_ref, &mut scratch.huffman_pool, &mut prefix_bits);
        for &token in &tokens {
            write_token(token, &code_ref, &mut prefix_bits);
        }
    }
    let mut ans_bits = BitWriter::new();
    {
        let code = crate::entropy::optimize_entropy_code_ac_streams(
            std::iter::once(tokens.as_slice()),
            NUM_PATCH_CONTEXTS,
            &mut scratch.huffman_pool,
            true,
            crate::Speed::Slow,
            None,
        );
        let code_ref = code.as_ref();
        ans_bits.write(1, 0);
        write_entropy_code(&code_ref, &mut scratch.huffman_pool, &mut ans_bits);
        if code_ref.use_prefix_code {
            for &token in &tokens {
                write_token(token, &code_ref, &mut ans_bits);
            }
        } else {
            crate::entropy::write_ans_tokens(
                &tokens,
                code_ref.context_map,
                code_ref.ans_symbols,
                code_ref.ans_reverse_maps,
                code_ref.hybrid_uint_configs,
                &mut ans_bits,
            );
        }
    }
    if ans_bits.bits_written() < prefix_bits.bits_written() {
        w.append(&ans_bits);
    } else {
        w.append(&prefix_bits);
    }
}

pub(super) fn write_modular_transforms(nb_chans: usize, rct_type: u32, w: &mut BitWriter) {
    if nb_chans >= 3 {
        // transforms count u2S(0, 1, Bits(4)+2, Bits(8)+18): selector 1 = Val(1) → 1 transform.
        w.write(2, 0b01);
        write_rct_transform(rct_type, w);
    } else {
        w.write(2, 0b00); // 0 transforms
    }
}

pub(super) fn write_toc_entry(byte_len: usize, w: &mut BitWriter) {
    static OFFSETS: [usize; 4] = [0, 1024, 17_408, 4_211_712];
    static BITS: [usize; 4] = [10, 14, 22, 30];
    let mut bucket = 0usize;
    while bucket < 3 && byte_len >= OFFSETS[bucket + 1] {
        bucket += 1;
    }
    w.write(2, bucket as u64);
    w.write(BITS[bucket], (byte_len - OFFSETS[bucket]) as u64);
}
