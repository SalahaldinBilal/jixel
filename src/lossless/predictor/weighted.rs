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

//! Weighted-predictor context buckets and row tokenization.

use super::{
    GradPackInteriorFn, GradientScratch, PREDICTOR_AVERAGE4, PREDICTOR_GRADIENT, PREDICTOR_LEFT,
    PREDICTOR_SELECT, PREDICTOR_TOP, PREDICTOR_WEIGHTED, TokenSink, pack_gradient_row,
    predictor_neighbors, predictor_value, push_wp_token, reset_wp_scratch, visit_predictor_row,
};
use crate::entropy::{Token, pack_signed};
use crate::weighted_predictor::{WpNeighbors, WpParams, WpState};

/// Cut-offs on the WP error property (libjxl property 15) of the fixed flat
/// context tree: every other entry of libjxl's `kWPFixedDC` table.
pub(in crate::lossless) const WP_CONTEXT_CUTOFFS: [i32; 17] = [
    -500, -255, -127, -63, -31, -15, -7, -3, 0, 3, 7, 15, 31, 63, 127, 255, 500,
];

/// Buckets of `WP_CONTEXT_CUTOFFS`: bucket `i` holds the values above exactly
/// `i` cut-offs.
pub(in crate::lossless) const WP_CONTEXT_BUCKETS: usize = WP_CONTEXT_CUTOFFS.len() + 1;

const WP_BUCKET_LIMIT: i64 = 501;

static WP_BUCKET_OF: [u8; 2 * WP_BUCKET_LIMIT as usize + 1] = {
    let mut table = [0u8; 2 * WP_BUCKET_LIMIT as usize + 1];
    let mut i = 0;
    while i < table.len() {
        let value = i as i32 - WP_BUCKET_LIMIT as i32;
        let mut bucket = 0;
        while bucket < WP_CONTEXT_CUTOFFS.len() && value > WP_CONTEXT_CUTOFFS[bucket] {
            bucket += 1;
        }
        table[i] = bucket as u8;
        i += 1;
    }
    table
};

/// Context of a WP-predicted pixel from its WP error property.
pub(in crate::lossless) trait WpContext: Copy {
    fn context(self, wp_prop: i64) -> u32;
}

impl WpContext for u32 {
    #[inline(always)]
    fn context(self, _wp_prop: i64) -> u32 {
        self
    }
}

/// One channel under the fixed WP context tree: the context of every bucket
/// of `wp_prop >> shift`.
pub(in crate::lossless) struct WpChannelContexts {
    pub(in crate::lossless) contexts: [u32; WP_CONTEXT_BUCKETS],
    pub(in crate::lossless) shift: u32,
}

#[derive(Clone, Copy)]
pub(super) struct WpBucketContexts<'a>(pub(super) &'a WpChannelContexts);

impl WpContext for WpBucketContexts<'_> {
    #[inline(always)]
    fn context(self, wp_prop: i64) -> u32 {
        let value = wp_prop >> self.0.shift;
        let index = value.clamp(-WP_BUCKET_LIMIT, WP_BUCKET_LIMIT) + WP_BUCKET_LIMIT;
        self.0.contexts[WP_BUCKET_OF[index as usize] as usize]
    }
}

/// The first, second, and interior row kernels deliberately have separate
/// border handling. Besides keeping branches out of the interior loop, this is
/// the dispatch seam for a future vector implementation of the row arithmetic.
fn tokenize_wp_first_row<T: Copy>(
    ctx: impl WpContext,
    current: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    let row = wp.row_offsets(0);
    let mut left = 0i64;
    for (x, &value) in current.iter().enumerate() {
        let value = i64::from(value);
        let prediction = wp.predict_and_update(
            value,
            x,
            row,
            WpNeighbors {
                north: left,
                west: left,
                north_east: left,
                north_west: left,
                north_north: left,
            },
        );
        push_wp_token(ctx.context(wp.wp_prop), value, prediction, out);
        left = value;
    }
}

fn tokenize_wp_second_row<T: Copy>(
    ctx: impl WpContext,
    current: &[T],
    north: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    tokenize_wp_interior_row(ctx, 1, current, north, north, wp, out);
}

fn tokenize_wp_interior_row<T: Copy>(
    ctx: impl WpContext,
    y: usize,
    current: &[T],
    north: &[T],
    north_north: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    let row = wp.row_offsets(y);
    visit_predictor_row(current, north, north_north, false, |x, value, n| {
        let prediction = wp.predict_and_update(
            value,
            x,
            row,
            WpNeighbors {
                north: n.top,
                west: n.left,
                north_east: n.top_right,
                north_west: n.top_left,
                north_north: n.top_top,
            },
        );
        push_wp_token(ctx.context(wp.wp_prop), value, prediction, out);
    });
}

pub(super) fn tokenize_wp_row_slices<'a, T: Copy + 'a>(
    ctx: impl WpContext,
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    out: &mut impl TokenSink,
    wp_params: WpParams,
    wp_scratch: &mut Option<Box<WpState>>,
) where
    i64: From<T>,
{
    if gw == 0 || gh == 0 {
        return;
    }
    let wp = reset_wp_scratch(wp_scratch, gw, wp_params);
    let first = get_row(0);
    debug_assert_eq!(first.len(), gw);
    tokenize_wp_first_row(ctx, first, wp, out);
    if gh == 1 {
        return;
    }
    let second = get_row(1);
    debug_assert_eq!(second.len(), gw);
    tokenize_wp_second_row(ctx, second, first, wp, out);
    let mut north_north = first;
    let mut north = second;
    for y in 2..gh {
        let current = get_row(y);
        debug_assert_eq!(current.len(), gw);
        tokenize_wp_interior_row(ctx, y, current, north, north_north, wp, out);
        north_north = north;
        north = current;
    }
}

fn tokenize_wp_rows_from_get(
    ctx: impl WpContext,
    get: &impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    if gw == 0 || gh == 0 {
        return;
    }
    if scratch.cur.len() < gw {
        scratch.cur.resize(gw, 0);
    }
    if scratch.prev.len() < gw {
        scratch.prev.resize(gw, 0);
    }
    if scratch.prev_prev.len() < gw {
        scratch.prev_prev.resize(gw, 0);
    }

    let mut current = &mut scratch.cur[..gw];
    let mut north = &mut scratch.prev[..gw];
    let mut north_north = &mut scratch.prev_prev[..gw];
    let wp = reset_wp_scratch(&mut scratch.wp, gw, wp_params);
    for y in 0..gh {
        if y != 0 {
            std::mem::swap(&mut north_north, &mut north);
            std::mem::swap(&mut north, &mut current);
        }
        for (x, value) in current.iter_mut().enumerate() {
            *value = get(x, y);
        }
        match y {
            0 => tokenize_wp_first_row(ctx, current, wp, out),
            1 => tokenize_wp_second_row(ctx, current, north, wp, out),
            _ => tokenize_wp_interior_row(ctx, y, current, north, north_north, wp, out),
        }
    }
}

pub(super) fn tokenize_plane_with_wp(
    ctx: u32,
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    if gw == 0 || gh == 0 {
        return;
    }
    if pred_id == PREDICTOR_WEIGHTED {
        tokenize_wp_rows_from_get(ctx, &get, gw, gh, scratch, out, wp_params);
    } else if pred_id == PREDICTOR_GRADIENT {
        // Gradient (ClampedGradient): per-pixel independent, pure integer ->
        // vectorized over the interior of each row.
        if scratch.buf.len() < gw {
            scratch.buf.resize(gw, 0);
        }
        if scratch.cur.len() < gw {
            scratch.cur.resize(gw, 0);
        }
        if scratch.prev.len() < gw {
            scratch.prev.resize(gw, 0);
        }
        let mut cur = &mut scratch.cur[..gw];
        let mut prev = &mut scratch.prev[..gw];
        let buf = &mut scratch.buf[..gw];
        for gy in 0..gh {
            std::mem::swap(&mut cur, &mut prev); // prev = last row's cur
            for (gx, c) in cur.iter_mut().enumerate() {
                *c = get(gx, gy);
            }
            pack_gradient_row(cur, (gy != 0).then_some(&*prev), buf, grad_pack_fn);
            out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
        }
    } else {
        debug_assert!(matches!(
            pred_id,
            PREDICTOR_AVERAGE4 | PREDICTOR_SELECT | PREDICTOR_LEFT | PREDICTOR_TOP
        ));
        for gy in 0..gh {
            for gx in 0..gw {
                let value = get(gx, gy) as i64;
                let neighbors = predictor_neighbors(&get, gx, gy, gw);
                let pred = predictor_value(pred_id, neighbors, 0);
                out.push_token(Token::new(ctx, pack_signed((value - pred) as i32)));
            }
        }
    }
}
