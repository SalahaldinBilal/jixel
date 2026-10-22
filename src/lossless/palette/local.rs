/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

//! Per-group palettes and multi-group palette frame encoding.

use super::super::groups::{
    DeepLzScratchPool, SLOW_DEEP_LZ_MAX_THREADS, group_lz_threads, write_lz_groups_with_header,
};
use super::super::header::{
    write_frame_header_modular_kind, write_modular_transforms, write_toc_entry,
};
use super::super::lz77::{
    LzToken, RunLzWriter, build_lz_pixel_code, lz77_compress_channels_for_speed_with_depth,
    lz77_compress_for_speed_with_depth, write_local_tree_lz77, write_lz_section,
};
use super::super::predictor::{
    GradPackInteriorFn, GradientScratch, SqueezePredictorCost, TokenSink, channel_to_context,
    choose_predictor_for_rows, fixed_predictor, tokenize_channels_with_wp, tokenize_plane_rows,
    tokenize_runs_with_wp, tokenize_sample_rows,
};
use super::super::{GROUP_DIM, GroupLayout};
use super::global::ColorMap;
use super::{visit_palette_colors, write_palette_transform};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// A sparse set of local-palette groups forces Slow mode to build a complete
/// alternate frame while most groups lose the learned-tree model. Require a
/// useful frame footprint before paying for that exact candidate.
/// color cap of a per-group palette. libjxl's own per-group cap is about
/// 640 for a 256x256 group, and screen content lives there: a 1280x720 UI
/// screen has 9 of its 15 groups between 221 and 870 colors.
pub(in crate::lossless) const LOCAL_PALETTE_MAX_COLORS: usize = 1024;

/// The flat local-palette frame keeps the original 256-color cap: it codes
/// every group without the learned tree, so wider palettes only buy it a
/// losing candidate (a 1000-color synthetic: +21% encode, no bytes).
const FLAT_LOCAL_PALETTE_MAX_COLORS: usize = 256;

const LOCAL_PALETTE_MIN_COVERAGE_NUM: usize = 1;

const LOCAL_PALETTE_MIN_COVERAGE_DEN: usize = 4;

#[inline]
pub(in crate::lossless) fn local_palette_coverage_is_sufficient(
    palette_pixels: usize,
    total_pixels: usize,
) -> bool {
    palette_pixels.saturating_mul(LOCAL_PALETTE_MIN_COVERAGE_DEN)
        >= total_pixels.saturating_mul(LOCAL_PALETTE_MIN_COVERAGE_NUM)
}

pub(in crate::lossless) struct LocalPaletteGroup {
    pub(in crate::lossless) palette: Vec<i32>,
    pub(in crate::lossless) indices: Vec<u16>,
    pub(in crate::lossless) nb_colors: usize,
    pub(in crate::lossless) w: usize,
    pub(in crate::lossless) h: usize,
}

impl LocalPaletteGroup {
    pub(super) fn tokenize(
        &self,
        num_contexts: usize,
        predictors: &[u32],
        grad_pack_fn: GradPackInteriorFn,
        scratch: &mut GradientScratch,
        out: &mut impl TokenSink,
    ) {
        tokenize_plane_rows(
            channel_to_context(0, num_contexts),
            |y| &self.palette[y * self.nb_colors..][..self.nb_colors],
            self.nb_colors,
            self.palette.len() / self.nb_colors,
            predictors[0],
            grad_pack_fn,
            scratch,
            out,
        );
        tokenize_sample_rows(
            channel_to_context(1, num_contexts),
            |y| &self.indices[y * self.w..][..self.w],
            self.w,
            self.h,
            predictors[1],
            grad_pack_fn,
            scratch,
            out,
        );
    }
}

fn add_alpha_predictor_cost(
    cost: &mut SqueezePredictorCost,
    alpha: &AlphaPlane,
    stride: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    use_wp: bool,
) {
    match alpha {
        AlphaPlane::U8(data) => {
            cost.add_rows(|y| &data[(y0 + y) * stride + x0..][..w], w, h, use_wp)
        }
        AlphaPlane::U16 { data, .. } => {
            cost.add_rows(|y| &data[(y0 + y) * stride + x0..][..w], w, h, use_wp)
        }
        AlphaPlane::F32(data) => {
            cost.add_rows(|y| &data[(y0 + y) * stride + x0..][..w], w, h, use_wp)
        }
    }
}

fn estimated_local_lz_stream_bits(
    lz: &[LzToken],
    predictors: &[u32],
    num_contexts: usize,
    min_symbol: u32,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> usize {
    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy,
        std::iter::once(lz),
        num_contexts,
        min_symbol,
        speed.effort().lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let mut writer = BitWriter::new();
    write_local_tree_lz77(
        predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut writer,
    );
    write_lz_section(lz, num_contexts as u32, &code, min_symbol, &mut writer);
    writer.bits_written()
}

#[allow(clippy::too_many_arguments)]
fn local_palette_is_better(
    palette: &LocalPaletteGroup,
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    use_wp: bool,
    scratch: &mut CoderScratch,
) -> (bool, SqueezePredictorCost) {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let mut index_cost = SqueezePredictorCost::default();
    let palette_predictors = if speed.effort().predictor_search {
        [
            choose_predictor_for_rows(
                |y| &palette.palette[y * palette.nb_colors..][..palette.nb_colors],
                palette.nb_colors,
                nb_chans,
                use_wp,
            ),
            {
                index_cost.add_rows(
                    |y| &palette.indices[y * palette.w..][..palette.w],
                    palette.w,
                    palette.h,
                    use_wp,
                );
                index_cost.predictor(use_wp)
            },
        ]
    } else {
        [fixed_predictor(use_wp); 2]
    };
    let mut palette_tokens = RunLzWriter::with_capacity(
        (nb_chans * palette.nb_colors + palette.w * palette.h).min(16 * 1024),
    );
    palette.tokenize(
        2,
        &palette_predictors,
        grad_pack_fn,
        &mut scratch.gradient,
        &mut palette_tokens,
    );
    let mut palette_transform = BitWriter::new();
    write_palette_transform(
        nb_chans as u32,
        palette.nb_colors as u32,
        &mut palette_transform,
    );
    let palette_bits = {
        let palette_lz = palette_tokens.finish();
        estimated_local_lz_stream_bits(
            &palette_lz,
            &palette_predictors,
            2,
            min_symbol,
            speed,
            scratch,
        ) + palette_transform.bits_written()
    };

    let plain_predictors: Vec<u32> = if speed.effort().predictor_search {
        (0..nb_chans)
            .map(|chan| {
                if chan < 3 {
                    let plane = linear.plane_data(chan);
                    choose_predictor_for_rows(
                        |y| &plane[(y0 + y) * xsize + x0..][..palette.w],
                        palette.w,
                        palette.h,
                        use_wp,
                    )
                } else {
                    let alpha = alpha.expect("alpha channel must exist");
                    let mut cost = SqueezePredictorCost::default();
                    add_alpha_predictor_cost(
                        &mut cost, alpha, xsize, x0, y0, palette.w, palette.h, use_wp,
                    );
                    cost.predictor(use_wp)
                }
            })
            .collect()
    } else {
        vec![fixed_predictor(use_wp); nb_chans]
    };
    let plain_lz = tokenize_runs_with_wp(
        linear,
        alpha,
        xsize,
        x0,
        y0,
        palette.w,
        palette.h,
        3,
        &plain_predictors,
        grad_pack_fn,
        scratch,
        WpParams::DEFAULT,
        None,
        None,
    );
    let plain_bits = estimated_local_lz_stream_bits(
        &plain_lz,
        &plain_predictors,
        nb_chans,
        min_symbol,
        speed,
        scratch,
    ) + 2; // zero-transform count

    (palette_bits < plain_bits, index_cost)
}

/// The flat path's builder: the original 256-color cap.
fn build_local_palette_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Option<LocalPaletteGroup> {
    build_local_palette_group_limited(
        linear,
        alpha,
        xsize,
        x0,
        y0,
        w,
        h,
        FLAT_LOCAL_PALETTE_MAX_COLORS,
    )
}

pub(in crate::lossless) fn build_local_palette_group_limited(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    max_colors: usize,
) -> Option<LocalPaletteGroup> {
    let num_c = 3 + usize::from(alpha.is_some());
    let num_pixels = w * h;
    // Most photographic groups exceed the palette limit almost immediately.
    // Probe only the distinct set first, avoiding a full-group tuple allocation
    // on that overwhelmingly common rejection path.
    // Palette indices come from the sorted colors below, independently of
    // hash-table iteration order. Reuse the global palette's cheap hasher.
    let mut seen = ColorMap::<()>::with_capacity_and_hasher(max_colors + 1, Default::default());
    let mut previous = None;
    if !visit_palette_colors::<false>(linear, alpha, xsize, x0, y0, w, h, |color| {
        if previous != Some(color) {
            seen.entry(color).or_insert(());
            previous = Some(color);
        }
        seen.len() <= max_colors
    }) {
        return None;
    }

    let nb_colors = seen.len();
    // Palette coding replaces `num_c * num_pixels` samples with one index per
    // pixel plus `num_c * nb_colors` palette samples. Leave a small margin for
    // the transform header and altered entropy statistics.
    let palette_samples = num_pixels + num_c * nb_colors + 16;
    if nb_colors == 0 || palette_samples >= num_c * num_pixels {
        return None;
    }

    let mut colors: Vec<[i32; 4]> = seen.keys().copied().collect();
    colors.sort_unstable();
    let mut index_of = ColorMap::<i32>::with_capacity_and_hasher(nb_colors, Default::default());
    let mut palette = vec![0i32; num_c * nb_colors];
    for (index, color) in colors.iter().enumerate() {
        index_of.insert(*color, index as i32);
    }
    for (c, row) in palette.chunks_exact_mut(nb_colors).enumerate() {
        for (value, color) in row.iter_mut().zip(&colors) {
            *value = color[c];
        }
    }
    let mut indices = Vec::with_capacity(num_pixels);
    let mut previous = ([0; 4], -1);
    visit_palette_colors::<false>(linear, alpha, xsize, x0, y0, w, h, |color| {
        if previous.1 < 0 || previous.0 != color {
            previous = (color, index_of[&color]);
        }
        indices.push(previous.1 as u16);
        true
    });

    Some(LocalPaletteGroup {
        palette,
        indices,
        nb_colors,
        w,
        h,
    })
}

/// Group step of the local-palette frame's plain-channel predictor costs
/// under `local_palette_predictors` alone.
const PLAIN_PREDICTOR_GROUP_STEP: usize = 4;

#[allow(clippy::too_many_arguments)]
pub(in crate::lossless) fn try_encode_local_palette_multi_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    xsize_groups: usize,
    ysize_groups: usize,
    num_dc_groups: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    use_wp: bool,
    frame_kind: ModularFrameKind<'_>,
    layout: GroupLayout,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let num_ac_groups = xsize_groups * ysize_groups;
    // Index samples are bytes, so these residual histograms stay small.
    // Merge only accepted palettes while their group costs are still available;
    // do not retain one set of histograms per group or serialize group scoring.
    let pooled_index_cost = std::sync::Mutex::new(SqueezePredictorCost::default());
    let palettes = if speed.effort().local_palette_select {
        let mut palettes = pool.steal_map(scratch, num_ac_groups, |group_index, _scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let w = GROUP_DIM.min(xsize - x0);
            let h = GROUP_DIM.min(ysize - y0);
            build_local_palette_group(linear, alpha, xsize, x0, y0, w, h)
        });
        let palette_pixels: usize = palettes
            .iter()
            .flatten()
            .map(|palette| palette.w * palette.h)
            .sum();
        if !local_palette_coverage_is_sufficient(palette_pixels, xsize * ysize) {
            return false;
        }

        let palette_is_better = pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
            let Some(palette) = &palettes[group_index] else {
                return false;
            };
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let (keep, index_cost) = local_palette_is_better(
                palette,
                linear,
                alpha,
                xsize,
                gx * GROUP_DIM,
                gy * GROUP_DIM,
                min_symbol,
                grad_pack_fn,
                speed,
                use_wp,
                scratch,
            );
            if keep {
                pooled_index_cost
                    .lock()
                    .unwrap()
                    .costs
                    .merge(&index_cost.costs);
            }
            keep
        });
        for (palette, keep) in palettes.iter_mut().zip(palette_is_better) {
            if !keep {
                *palette = None;
            }
        }
        palettes
    } else {
        pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let w = GROUP_DIM.min(xsize - x0);
            let h = GROUP_DIM.min(ysize - y0);
            let palette = build_local_palette_group(linear, alpha, xsize, x0, y0, w, h)?;
            local_palette_is_better(
                &palette,
                linear,
                alpha,
                xsize,
                x0,
                y0,
                min_symbol,
                grad_pack_fn,
                speed,
                use_wp,
                scratch,
            )
            .0
            .then_some(palette)
        })
    };
    if !palettes.iter().any(Option::is_some) {
        return false;
    }

    // The global MA tree sees group-local channel slots. Pool predictor costs
    // for palette/index channels and ordinary YCoCg(A) channels by those slots.
    let predictors: Vec<u32> = if speed.effort().predictor_search
        || speed.effort().local_palette_predictors
    {
        // Without `predictor_search` the plain channels are costed on a
        // diagonal of every `PLAIN_PREDICTOR_GROUP_STEP`-th group.
        let plain_step = if speed.effort().predictor_search {
            1
        } else {
            PLAIN_PREDICTOR_GROUP_STEP
        };
        // Split the remaining palette-meta and plain-channel work by group.
        // Each crop still resets WP; integer histogram merging leaves the
        // final costs and predictor tie-breaking unchanged.
        let tasks = pool
            .num_threads()
            .min(SLOW_DEEP_LZ_MAX_THREADS)
            .min(num_ac_groups)
            .max(1);
        let groups_per_task = num_ac_groups.div_ceil(tasks);
        let parts = pool.steal_map(scratch, tasks, |task, _scratch| {
            let mut costs: [SqueezePredictorCost; 4] =
                std::array::from_fn(|_| SqueezePredictorCost::default());
            let begin = task * groups_per_task;
            let end = (begin + groups_per_task).min(num_ac_groups);
            for group_index in begin..end {
                let palette = &palettes[group_index];
                for (slot, cost) in costs[..nb_chans].iter_mut().enumerate() {
                    if let Some(palette) = palette {
                        if slot == 0 {
                            cost.add_rows(
                                |y| &palette.palette[y * palette.nb_colors..][..palette.nb_colors],
                                palette.nb_colors,
                                nb_chans,
                                use_wp,
                            );
                        }
                        // Slot 1 already includes accepted palette indices.
                        continue;
                    }
                    if !(group_index % xsize_groups + group_index / xsize_groups)
                        .is_multiple_of(plain_step)
                    {
                        continue;
                    }

                    let gx = group_index % xsize_groups;
                    let gy = group_index / xsize_groups;
                    let x0 = gx * GROUP_DIM;
                    let y0 = gy * GROUP_DIM;
                    let w = GROUP_DIM.min(xsize - x0);
                    let h = GROUP_DIM.min(ysize - y0);
                    if slot < 3 {
                        let plane = linear.plane_data(slot);
                        cost.add_rows(|y| &plane[(y0 + y) * xsize + x0..][..w], w, h, use_wp);
                    } else {
                        let alpha = alpha.expect("alpha slot requires alpha channel");
                        add_alpha_predictor_cost(cost, alpha, xsize, x0, y0, w, h, use_wp);
                    }
                }
            }
            costs
        });
        let mut costs: [SqueezePredictorCost; 4] =
            std::array::from_fn(|_| SqueezePredictorCost::default());
        costs[1] = pooled_index_cost.into_inner().unwrap();
        for part in parts {
            for (cost, part) in costs.iter_mut().zip(part) {
                cost.costs.merge(&part.costs);
            }
        }
        costs[..nb_chans]
            .iter()
            .map(|cost| cost.predictor(use_wp))
            .collect()
    } else {
        vec![fixed_predictor(use_wp); nb_chans]
    };

    let distance_ctx = nb_chans as u32;
    let group_lz_tokens: Vec<Vec<LzToken>> = {
        let deep_lz = speed
            .effort()
            .lz_deep
            .then(|| DeepLzScratchPool::new(group_lz_threads(speed, pool)));
        pool.steal_map_with_threads(
            scratch,
            num_ac_groups,
            group_lz_threads(speed, pool),
            |group_index, scratch| {
                if let Some(palette) = &palettes[group_index] {
                    let capacity = nb_chans * palette.nb_colors + palette.w * palette.h;
                    if let Some(deep_lz) = &deep_lz {
                        let mut tokens = Vec::with_capacity(capacity);
                        palette.tokenize(
                            nb_chans,
                            &predictors,
                            grad_pack_fn,
                            &mut scratch.gradient,
                            &mut tokens,
                        );
                        deep_lz.with_depth(|depth| {
                            lz77_compress_for_speed_with_depth(
                                &tokens,
                                distance_ctx,
                                speed,
                                depth,
                                None,
                                scratch,
                            )
                        })
                    } else {
                        let mut runs = RunLzWriter::with_capacity(capacity.min(16 * 1024));
                        palette.tokenize(
                            nb_chans,
                            &predictors,
                            grad_pack_fn,
                            &mut scratch.gradient,
                            &mut runs,
                        );
                        runs.finish()
                    }
                } else {
                    let gx = group_index % xsize_groups;
                    let gy = group_index / xsize_groups;
                    let x0 = gx * GROUP_DIM;
                    let y0 = gy * GROUP_DIM;
                    let w = GROUP_DIM.min(xsize - x0);
                    let h = GROUP_DIM.min(ysize - y0);
                    if speed.effort().lz_deep {
                        let channel_tokens = tokenize_channels_with_wp(
                            linear,
                            alpha,
                            xsize,
                            ysize,
                            x0,
                            y0,
                            w,
                            h,
                            3,
                            &predictors,
                            grad_pack_fn,
                            pool,
                            scratch,
                            WpParams::DEFAULT,
                            None,
                            None,
                        );
                        deep_lz.as_ref().unwrap().with_depth(|depth| {
                            lz77_compress_channels_for_speed_with_depth(
                                channel_tokens,
                                distance_ctx,
                                speed,
                                depth,
                                scratch,
                            )
                        })
                    } else {
                        tokenize_runs_with_wp(
                            linear,
                            alpha,
                            xsize,
                            x0,
                            y0,
                            w,
                            h,
                            3,
                            &predictors,
                            grad_pack_fn,
                            scratch,
                            WpParams::DEFAULT,
                            None,
                            None,
                        )
                    }
                }
            },
        )
    };
    // Only transform metadata is needed after tokenization. Release palette
    // planes before allocating entropy tables and serialized group sections.
    let palette_sizes: Vec<_> = palettes
        .into_iter()
        .map(|palette| palette.map(|palette| palette.nb_colors))
        .collect();

    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy,
        group_lz_tokens.iter().map(Vec::as_slice),
        nb_chans,
        min_symbol,
        speed.effort().lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    write_frame_header_modular_kind(alpha.is_some(), frame_kind, layout, writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default
    sections[0].write(1, 1); // has global tree
    write_local_tree_lz77(
        &predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    sections[0].write(1, 1); // wp_default
    write_modular_transforms(nb_chans, 6, &mut sections[0]);
    // An empty global stream still carries its ANS state: jxl-oxide reads it.
    write_lz_section::<LzToken>(&[], distance_ctx, &code, min_symbol, &mut sections[0]);
    sections[0].zero_pad_to_byte();

    for section in sections[1..num_dc_groups + 1].iter_mut() {
        section.write(1, 1);
        section.write(1, 1);
        section.write(2, 0);
        section.zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    write_lz_groups_with_header(
        &group_lz_tokens,
        &[],
        &code,
        distance_ctx,
        min_symbol,
        pool,
        &mut sections[2 + num_dc_groups..],
        |group_index, section| {
            section.write(1, 1); // use_global_tree
            section.write(1, 1); // wp_default
            if let Some(nb_colors) = palette_sizes[group_index] {
                write_palette_transform(nb_chans as u32, nb_colors as u32, section);
            } else {
                section.write(2, 0); // no local transforms
            }
        },
    );

    writer.write(1, 0); // no TOC permutation
    writer.zero_pad_to_byte();
    for section in &sections {
        write_toc_entry(section.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}
