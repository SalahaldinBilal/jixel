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

//! Displayed, reference, and region lossless frame orchestration.

use super::encoder::encode_frame_lossless_core;
use super::header::write_patch_dictionary;
use super::ma_learning::rank_coarse;
use super::ma_source::{MA_FULL_SAMPLING, PATCH_MIN_COVERAGE, rgb_ma_source};
use super::{GroupLayout, constant_alpha_value, lossless_min_symbol};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::patches::{LosslessPatches, ModularFrameKind, find_lossless_patches};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

fn zero_alpha_like(alpha: Option<&AlphaPlane>, pixels: usize) -> Option<AlphaPlane> {
    match alpha {
        Some(AlphaPlane::U8(_)) => Some(AlphaPlane::U8(vec![0; pixels])),
        Some(AlphaPlane::U16 { bits, .. }) => Some(AlphaPlane::U16 {
            data: vec![0; pixels],
            bits: *bits,
        }),
        Some(AlphaPlane::F32(_)) => Some(AlphaPlane::F32(vec![0; pixels])),
        None => None,
    }
}

pub(super) fn keep_smaller_writer(best: &mut Option<BitWriter>, candidate: BitWriter) {
    if best
        .as_ref()
        .is_none_or(|current| candidate.bits_written() < current.bits_written())
    {
        *best = Some(candidate);
    }
}

pub(crate) fn encode_frame_lossless(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    patches: bool,
    num_color: usize,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    near_lossless: bool,
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let pool = ThreadPool::new_lossless(num_threads);
    let mut scratch = Box::new(CoderScratch::lossless());
    encode_frame_lossless_with_pool(
        linear,
        alpha,
        max_bits,
        progressive,
        patches,
        num_color,
        speed,
        decoding_speed,
        near_lossless,
        &pool,
        &mut scratch,
        writer,
    );
}

/// The displayed frame, or, for a screenshot with gradient or photo-like
/// rectangles (see `region_frames`), the base saved to reference slot 0
/// followed by one cropped kReplace frame per rectangle, whichever is
/// smaller.
#[allow(clippy::too_many_arguments)]
fn encode_frame_lossless_with_pool(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    patches: bool,
    num_color: usize,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    near_lossless: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    // The whole frame's glyph plan, shared with its encode below.
    let mut whole_glyphs: Option<Option<LosslessPatches>> = None;
    if speed.effort().region_frames
        && !progressive
        && num_color == 3
        && max_bits == 8
        && let Some(mut regions) = crate::region_frames::plan(linear, alpha)
    {
        let min_symbol = lossless_min_symbol(max_bits);
        let use_wp = decoding_speed.use_weighted_predictor();
        let mut estimate = |image: &Image3Si, alpha: Option<&AlphaPlane>| {
            coarse_frame_bits(image, alpha, min_symbol, use_wp, pool, scratch)
        };
        let mut crops: Vec<_> = regions
            .iter()
            .map(|r| crate::region_frames::crop(linear, alpha, r))
            .collect();
        let mut own: Vec<Option<f64>> =
            crops.iter().map(|(c, a)| estimate(c, a.as_ref())).collect();
        // A rectangle of text (a nested screenshot) codes worse as its own
        // frame than inside the whole, whose glyph patches cover it: it
        // stays in the base when the patches save a share of its estimate.
        if patches {
            let glyphs = whole_glyphs.insert(crate::patches::find_lossless_glyph_patches(linear));
            if let Some(glyphs) = glyphs {
                let mut keep = Vec::with_capacity(regions.len());
                for (i, r) in regions.iter().enumerate() {
                    let (patched, patched_alpha) =
                        crate::region_frames::crop(&glyphs.base, alpha, r);
                    let text_like = match (own[i], estimate(&patched, patched_alpha.as_ref())) {
                        (Some(own), Some(patched)) => patched < own * (1.0 - REGION_GLYPH_SAVING),
                        _ => false,
                    };
                    keep.push(!text_like);
                }
                let mut kept = keep.iter();
                regions.retain(|_| *kept.next().unwrap_or(&true));
                let mut kept = keep.iter();
                crops.retain(|_| *kept.next().unwrap_or(&true));
                let mut kept = keep.iter();
                own.retain(|_| *kept.next().unwrap_or(&true));
            }
        }
        let (base, base_alpha) = crate::region_frames::cut(linear, alpha, &regions);
        let whole = estimate(linear, alpha);
        let base_bits = estimate(&base, base_alpha.as_ref());
        let split = own
            .iter()
            .copied()
            .chain([base_bits])
            .sum::<Option<f64>>()
            .map(|bits| bits + REGION_FRAME_OVERHEAD_BITS * regions.len() as f64);
        if !regions.is_empty()
            && let (Some(whole), Some(split)) = (whole, split)
            && split < whole * REGION_SPLIT_MARGIN
        {
            let mut split_writer = BitWriter::new();
            encode_displayed_frame(
                &base,
                base_alpha.as_ref(),
                max_bits,
                progressive,
                patches,
                num_color,
                speed,
                decoding_speed,
                near_lossless,
                pool,
                scratch,
                true,
                None,
                &mut split_writer,
            );
            for (i, ((crop, crop_alpha), r)) in crops.iter().zip(&regions).enumerate() {
                encode_frame_lossless_core(
                    crop,
                    crop_alpha.as_ref(),
                    max_bits,
                    false,
                    num_color,
                    speed,
                    decoding_speed,
                    near_lossless,
                    pool,
                    scratch,
                    ModularFrameKind::RgbCrop {
                        x0: r.x0,
                        y0: r.y0,
                        width: r.width,
                        height: r.height,
                        last: i + 1 == regions.len(),
                    },
                    true,
                    &mut split_writer,
                );
            }
            // The estimate ranks the two within a few percent and misses
            // what the cut does to the base's palette and tree, so the
            // whole frame competes on complete size.
            let mut whole_writer = BitWriter::new();
            encode_displayed_frame(
                linear,
                alpha,
                max_bits,
                progressive,
                patches,
                num_color,
                speed,
                decoding_speed,
                near_lossless,
                pool,
                scratch,
                false,
                whole_glyphs.take(),
                &mut whole_writer,
            );
            if whole_writer.bits_written() < split_writer.bits_written() {
                writer.append(&whole_writer);
            } else {
                writer.append(&split_writer);
            }
            return;
        }
    }
    encode_displayed_frame(
        linear,
        alpha,
        max_bits,
        progressive,
        patches,
        num_color,
        speed,
        decoding_speed,
        near_lossless,
        pool,
        scratch,
        false,
        whole_glyphs,
        writer,
    );
}

/// Header, TOC and tree of one region frame.
const REGION_FRAME_OVERHEAD_BITS: f64 = 1600.0;

/// A rectangle stays in the base when the whole frame's glyph patches save
/// this share of its own-frame estimate (a nested screenshot 15%, a chat
/// header block 19%; an image thumbnail 2%, art and avatars 0).
const REGION_GLYPH_SAVING: f64 = 0.1;

/// The region split is encoded when its first-stage estimate is under this
/// share of the whole frame's: estimated gains of 1.8% and more won 2-20%,
/// one of 0.6% lost 3.7%.
const REGION_SPLIT_MARGIN: f64 = 0.99;

#[allow(clippy::too_many_arguments)]
fn encode_displayed_frame(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    patches: bool,
    num_color: usize,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    near_lossless: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    saved_first: bool,
    glyph_plan: Option<Option<LosslessPatches>>,
    writer: &mut BitWriter,
) {
    let regular = if saved_first {
        ModularFrameKind::RgbSavedFirst
    } else {
        ModularFrameKind::Regular
    };
    // Glyph plans (arbitrary-position repeated shapes: text, UI) come first
    // and always compete on complete frame size; tile plans keep the old
    // decisive-tree skip, having never beaten one. The caller may have
    // found the plan already.
    let glyph_plan = match glyph_plan {
        Some(plan) => plan,
        None if patches && num_color == 3 => crate::patches::find_lossless_glyph_patches(linear),
        None => None,
    };
    let is_glyph_plan = glyph_plan.is_some();
    if patches
        && num_color == 3
        && let Some(plan) = glyph_plan.or_else(|| find_lossless_patches(linear, pool, scratch))
    {
        // Patches change prediction, LZ77, entropy histograms, and add both a
        // reference-only frame and a dictionary. Their true cost cannot be
        // inferred reliably from covered pixel area, so both complete
        // alternatives are compared by their final byte-aligned bit count.
        let mut regular_writer = BitWriter::new();
        if !is_glyph_plan {
            // Tile plans: the regular frame first; a decisive learned tree
            // (never beaten by a tile plan) or a plan covering little of the
            // image (cannot pay for its dictionary and reference frame)
            // skips the patched encode.
            let decisive_tree = encode_frame_lossless_core(
                linear,
                alpha,
                max_bits,
                progressive,
                num_color,
                speed,
                decoding_speed,
                near_lossless,
                pool,
                scratch,
                regular,
                true,
                &mut regular_writer,
            );
            let covered: usize = plan
                .references
                .iter()
                .map(|r| r.positions.len() * r.width * r.height)
                .sum::<usize>();
            let coverage = covered as f64 / (linear.xsize() * linear.ysize()) as f64;
            if decisive_tree || coverage < PATCH_MIN_COVERAGE {
                writer.append(&regular_writer);
                return;
            }
        }

        let mut patched_writer = BitWriter::new();
        let atlas_alpha =
            zero_alpha_like(alpha, plan.atlas.xsize().saturating_mul(plan.atlas.ysize()));
        encode_frame_lossless_core(
            &plan.atlas,
            atlas_alpha.as_ref(),
            max_bits,
            false,
            num_color,
            speed,
            decoding_speed,
            near_lossless,
            pool,
            scratch,
            ModularFrameKind::ReferenceOnly {
                width: plan.atlas.xsize(),
                height: plan.atlas.ysize(),
            },
            true,
            &mut patched_writer,
        );
        let atlas_bits = patched_writer.bits_written();
        encode_frame_lossless_core(
            &plan.base,
            alpha,
            max_bits,
            false,
            num_color,
            speed,
            decoding_speed,
            near_lossless,
            pool,
            scratch,
            ModularFrameKind::Patched {
                references: &plan.references,
                saved_first,
            },
            true,
            &mut patched_writer,
        );

        if is_glyph_plan {
            // Glyph plans win whenever the plan is any good, so the regular
            // frame (a complete second encode) is written only when the
            // first-stage estimates leave it a chance; the dense local-tree
            // learn is spent on the patched frame only.
            let min_symbol = lossless_min_symbol(max_bits);
            let use_wp = decoding_speed.use_weighted_predictor();
            let regular_est = coarse_frame_bits(linear, alpha, min_symbol, use_wp, pool, scratch);
            let base_est = coarse_frame_bits(&plan.base, alpha, min_symbol, use_wp, pool, scratch);
            let competitive = match (regular_est, base_est) {
                (Some(regular), Some(base)) => {
                    let mut dictionary = BitWriter::new();
                    write_patch_dictionary(
                        &plan.references,
                        alpha.is_some(),
                        scratch,
                        &mut dictionary,
                    );
                    let patched = base + (atlas_bits + dictionary.bits_written()) as f64;
                    regular <= patched * GLYPH_REGULAR_COARSE_MARGIN
                }
                _ => true,
            };
            if competitive {
                encode_frame_lossless_core(
                    linear,
                    alpha,
                    max_bits,
                    progressive,
                    num_color,
                    speed,
                    decoding_speed,
                    near_lossless,
                    pool,
                    scratch,
                    regular,
                    false,
                    &mut regular_writer,
                );
            }
        }

        if regular_writer.bits_written() > 0
            && regular_writer.bits_written() < patched_writer.bits_written()
        {
            writer.append(&regular_writer);
        } else {
            writer.append(&patched_writer);
        }
        return;
    }
    encode_frame_lossless_core(
        linear,
        alpha,
        max_bits,
        progressive,
        num_color,
        speed,
        decoding_speed,
        near_lossless,
        pool,
        scratch,
        regular,
        true,
        writer,
    );
}

/// First-stage learned-tree estimate of a frame's pixel bits: the ranking
/// stage's coarse learn (32 leaves, YCoCg planes, default WP preset, 1024-px
/// layout), or `None` when no tree beats the flat coder there. Frames of one
/// image relate to their complete encodes by nearly the same factor (a UI
/// screenshot and its glyph-patched base: 0.77 and 0.80), so it can rank two
/// competing frames before either is fully encoded.
fn coarse_frame_bits(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    min_symbol: u32,
    use_wp: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<f64> {
    let (xsize, ysize) = (linear.xsize(), linear.ysize());
    let source = rgb_ma_source(
        linear,
        alpha,
        xsize,
        6,
        alpha.and_then(constant_alpha_value),
        3,
    );
    let stage = rank_coarse(
        &source,
        xsize,
        ysize,
        min_symbol,
        WpParams::DEFAULT,
        use_wp,
        GroupLayout::LARGE,
        MA_FULL_SAMPLING,
        pool,
        scratch,
    )?;
    Some(stage.candidate.est_real.min(stage.candidate.flat_real))
}

/// A glyph plan's regular frame is encoded only when its first-stage
/// estimate is within this factor of the patched frame's (base estimate plus
/// the atlas frame and dictionary): the coarse ratio between the two frames
/// tracks the final one within a few percent (1.25 coarse vs 1.19 final on a
/// UI screenshot), and a clear loser's complete encode was a third of the
/// total time.
const GLYPH_REGULAR_COARSE_MARGIN: f64 = 1.10;
