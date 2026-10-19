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

//! Lossless floating-point modular frames.

use super::header::{write_frame_header_modular, write_toc_entry};
use super::predictor::{selected_grad_pack_interior_fn, tokenize_all};
use super::tree::{build_balanced_tree_tokens, write_tree_and_pixel_code_nolz};
use super::{GROUP_DIM, GroupLayout, LF_GROUP_DIM};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::{Token, optimize_entropy_code, write_token};
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;

pub(crate) fn encode_frame_lossless_float(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let pool = ThreadPool::new_lossless(num_threads);
    let mut scratch = Box::new(CoderScratch::lossless());
    encode_frame_lossless_float_with_pool(linear, alpha, &pool, &mut scratch, writer);
}

fn encode_frame_lossless_float_with_pool(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    let xsize = linear.xsize();
    let ysize = linear.ysize();
    let nb_chans = 3usize + if alpha.is_some() { 1 } else { 0 };

    let xsize_groups = xsize.div_ceil(GROUP_DIM);
    let ysize_groups = ysize.div_ceil(GROUP_DIM);
    let num_ac_groups = xsize_groups * ysize_groups;
    let xsize_dc_groups = xsize.div_ceil(LF_GROUP_DIM);
    let ysize_dc_groups = ysize.div_ceil(LF_GROUP_DIM);
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
    let single_group = num_ac_groups == 1;
    let grad_pack_fn = selected_grad_pack_interior_fn();

    // Float-bit values reach ~2^30, where the Weighted Predictor's internal
    // weight/divlookup arithmetic is not bit-verified against libjxl. Gradient
    // (clamp(W+N-NW), computed in i64) is exact at any magnitude and the
    // decoder reconstructs it identically, so the float path uses gradient on
    // every channel. (Predictor id 5 = gradient.)
    const GRADIENT_PRED: u32 = 5;
    let predictors = [GRADIENT_PRED; 4];
    let tree_tokens = build_balanced_tree_tokens(&predictors[..nb_chans]);

    write_frame_header_modular(alpha.is_some(), GroupLayout::DEFAULT, writer);

    if single_group {
        let mut section = BitWriter::new();
        section.write(1, 1); // dc_quant all_default = 1
        section.write(1, 0); // has_tree = 0
        section.write(1, 0); // use_global_tree = 0
        section.write(1, 1); // wp_default = 1
        section.write(2, 0b00); // 0 transforms (no RCT for float bits)

        let tokens = tokenize_all(
            linear,
            alpha,
            xsize,
            ysize,
            0,
            0,
            xsize,
            ysize,
            3,
            &predictors,
            grad_pack_fn,
            pool,
            scratch,
        );
        let code = optimize_entropy_code(&tokens, nb_chans, &mut scratch.huffman_pool);
        write_tree_and_pixel_code_nolz(&tree_tokens, &code, scratch, &mut section);
        for t in &tokens {
            write_token(*t, &code.as_ref(), &mut section);
        }
        section.zero_pad_to_byte();

        writer.write(1, 0);
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(std::slice::from_mut(&mut section));
        writer.zero_pad_to_byte();
    } else {
        let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
        let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

        // Tokenize each AC group (group-local) and pool for one global code.
        let group_tokens: Vec<Vec<Token>> =
            pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);
                tokenize_all(
                    linear,
                    alpha,
                    xsize,
                    ysize,
                    x0,
                    y0,
                    gw,
                    gh,
                    3,
                    &predictors,
                    grad_pack_fn,
                    pool,
                    scratch,
                )
            });
        let mut all_tokens: Vec<Token> = Vec::new();
        for tokens in &group_tokens {
            all_tokens.extend_from_slice(tokens);
        }
        let code = optimize_entropy_code(&all_tokens, nb_chans, &mut scratch.huffman_pool);

        // Section 0: DC global (tree + pixel code, both no-LZ77) + GroupHeader.
        sections[0].write(1, 1); // dc_quant
        sections[0].write(1, 1); // has_tree = 1
        write_tree_and_pixel_code_nolz(&tree_tokens, &code, scratch, &mut sections[0]);
        sections[0].write(1, 1); // use_global_tree
        sections[0].write(1, 1); // wp_default
        sections[0].write(2, 0b00); // 0 transforms
        sections[0].zero_pad_to_byte();

        for section in &mut sections[1..1 + num_dc_groups] {
            section.write(1, 1);
            section.write(1, 1);
            section.write(2, 0);
            section.zero_pad_to_byte();
        }

        let ac_global_idx = 1 + num_dc_groups;
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].zero_pad_to_byte();

        for (section, tokens) in sections[2 + num_dc_groups..][..num_ac_groups]
            .iter_mut()
            .zip(&group_tokens[..num_ac_groups])
        {
            section.write(1, 1);
            section.write(1, 1);
            section.write(2, 0);
            for t in tokens {
                write_token(*t, &code.as_ref(), section);
            }
            section.zero_pad_to_byte();
        }

        writer.write(1, 0);
        writer.zero_pad_to_byte();
        for s in &sections {
            write_toc_entry(s.bits_written() / 8, writer);
        }
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(&mut sections);
        writer.zero_pad_to_byte();
    }
}
