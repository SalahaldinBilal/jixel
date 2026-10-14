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
//! Predictor-aware chroma DC rounding on the existing quantization lattice.

use super::{
    DC_PREDICTOR_WEIGHTED, DcPlanePrice, DcPredictorChoice, DistanceParams, K_GRAD_RANGE_MAX,
    K_GRAD_RANGE_MID, K_GRAD_RANGE_MIN, choose_dc_predictors, collect_dc_tokens_with_gradient,
};
use crate::coder_scratch::DcPredictorScratch;
use crate::color_correlation::{CflFrame, dc_cfl_factor, dc_cfl_factor_x};
use crate::dc_group_data::DcGroupData;
use crate::dc_ops::BdcCompensation;
use crate::dct::fmla;
use crate::entropy::{ALPHABET_SIZE, Token, f_log2, pack_signed, uint_encode};
use crate::quant_weights::INV_DC_QUANT;
use crate::static_entropy_codes::{K_GRADIENT_CONTEXT_LUT, K_NUM_DC_CONTEXTS};
use crate::util::{EncodeError, try_vec};
use crate::weighted_predictor::WpState;

/// Rate penalty in squared DC-step units per bit, ramped linearly between
/// the two distances.
static DC_RDO_LAMBDA: [f32; 2] = [0.04, 0.15];
static DC_RDO_LAMBDA_DISTANCE: [f32; 2] = [1.0, 2.0];
/// X rounds toward its prediction more readily than B.
const DC_RDO_X_LAMBDA_SCALE: f32 = 1.5;
/// Bound the error of a newly chosen level, retaining larger incumbent errors.
const DC_RDO_MAX_ERROR: f32 = 0.65;
/// Require enough estimated payload gain to absorb small coding-model changes.
const DC_RDO_MIN_GAIN_BITS: f64 = 8.0;

struct Prices {
    bits: Vec<[f32; ALPHABET_SIZE]>,
}

impl Prices {
    fn new(
        groups: &[DcGroupData],
        scratch: &mut DcPredictorScratch,
    ) -> Result<(Self, DcPredictorChoice), EncodeError> {
        let mut wp = Vec::with_capacity(groups.len());
        let mut grad = Vec::with_capacity(groups.len());
        for dc in groups {
            let (w, g) = collect_dc_tokens_with_gradient(
                dc,
                &DC_PREDICTOR_WEIGHTED,
                &mut Vec::new(),
                false,
                true,
            );
            wp.push(w);
            grad.push(g);
        }
        let predictors = choose_dc_predictors(&wp, &grad, scratch);
        let mut counts = try_vec![[0u32; ALPHABET_SIZE]; 3 * K_NUM_DC_CONTEXTS]?;
        for ((dc, wp), grad) in groups.iter().zip(&wp).zip(&grad) {
            let mut offset = 0;
            for (mchan, c) in [1, 0, 2].into_iter().enumerate() {
                let end = offset + dc.quant_dc.plane(c).as_slice().len();
                for (w, g) in wp[offset..end].iter().zip(&grad[offset..end]) {
                    let token = if predictors[w.context as usize] { g } else { w };
                    let (symbol, _, _) = uint_encode(token.value);
                    counts[mchan * K_NUM_DC_CONTEXTS + token.context as usize][symbol as usize] +=
                        1;
                }
                offset = end;
            }
        }
        let mut bits = try_vec![[0.0; ALPHABET_SIZE]; counts.len()]?;
        for (counts, bits) in counts.iter().zip(&mut bits) {
            // A half-count keeps newly visited contexts and symbols finite.
            let total =
                counts.iter().map(|&n| f64::from(n)).sum::<f64>() + 0.5 * ALPHABET_SIZE as f64;
            for (&count, bits) in counts.iter().zip(bits) {
                *bits = f_log2(total / (f64::from(count) + 0.5)) as f32;
            }
        }
        Ok((Self { bits }, predictors))
    }

    #[inline]
    fn residual_bits(&self, mchan: usize, context: usize, residual: i32) -> f32 {
        let (symbol, extra, _) = uint_encode(pack_signed(residual));
        self.bits[mchan * K_NUM_DC_CONTEXTS + context][symbol as usize] + extra as f32
    }
}

/// Prices are frozen before changing the planes. Final tree learning and
/// entropy coding still run on the chosen levels, including any predictor
/// flips. The frozen static contexts are only a quantization rate estimate.
pub(super) fn refine_dc_rounding(
    groups: &mut [DcGroupData],
    distp: &DistanceParams,
    ytob_dc: i32,
    cfl: CflFrame,
    compensate_b: bool,
    scratch: &mut DcPredictorScratch,
) -> Result<(), EncodeError> {
    if groups
        .iter()
        .any(|dc| dc.source_dc_y.is_none() || dc.source_dc_x.is_none() || dc.source_dc_b.is_none())
    {
        return Ok(());
    }
    let (prices, predictors) = Prices::new(groups, scratch)?;
    let inv: [f32; 3] =
        std::array::from_fn(|c| INV_DC_QUANT[c] * distp.scale_dc / distp.dc_step[c]);
    let factors = [
        dc_cfl_factor_x(distp.dc_step, cfl.base_x),
        0.0,
        dc_cfl_factor(distp.dc_step, ytob_dc, cfl),
    ];
    let compensation = BdcCompensation::new(distp);
    let [d0, d1] = DC_RDO_LAMBDA_DISTANCE;
    let t = ((distp.distance - d0) / (d1 - d0)).clamp(0.0, 1.0);
    let lambda = DC_RDO_LAMBDA[0] + (DC_RDO_LAMBDA[1] - DC_RDO_LAMBDA[0]) * t;
    // Preserve luma: changing it also changes both chroma reconstruction
    // targets. Chroma channels can be raced independently under the final Y.
    for (mchan, c, lambda) in [(1, 0, lambda * DC_RDO_X_LAMBDA_SCALE), (2, 2, lambda)] {
        let previous: Vec<Vec<i16>> = groups
            .iter()
            .map(|dc| dc.quant_dc.plane(c).as_slice().to_vec())
            .collect();
        let (mut base_price, mut candidate_price) =
            (DcPlanePrice::default(), DcPlanePrice::default());
        let (mut base_error, mut candidate_error) = (0.0f64, 0.0f64);
        for (dc, previous) in groups.iter_mut().zip(&previous) {
            let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
            let mut targets = try_vec![0.0f32; w * h]?;
            let source = match c {
                0 => dc.source_dc_x.as_ref().unwrap(),
                1 => dc.source_dc_y.as_ref().unwrap(),
                _ => dc.source_dc_b.as_ref().unwrap(),
            };
            for (((target, &source), &yq), &sy) in targets
                .iter_mut()
                .zip(source.as_slice())
                .zip(dc.quant_dc.plane(1).as_slice())
                .zip(dc.source_dc_y.as_ref().unwrap().as_slice())
            {
                *target = fmla(source, inv[c], -f32::from(yq) * factors[c]);
                if c == 2 && compensate_b {
                    *target = compensation.adjust(*target, sy, yq);
                }
            }
            base_price.add(previous, w, h, w);
            base_error += targets
                .iter()
                .zip(previous)
                .map(|(&t, &q)| f64::from(t - f32::from(q)).powi(2))
                .sum::<f64>();
            refine_channel(
                dc.quant_dc.plane_mut(c).as_mut_slice(),
                &targets,
                w,
                h,
                mchan,
                &predictors,
                &prices,
                lambda,
                |_| {},
            );
            let levels = dc.quant_dc.plane(c).as_slice();
            candidate_price.add(levels, w, h, w);
            candidate_error += targets
                .iter()
                .zip(levels)
                .map(|(&t, &q)| f64::from(t - f32::from(q)).powi(2))
                .sum::<f64>();
        }
        // Local choices change later predictions. Reprice the complete plane
        // with pooled histograms before keeping any of its changes, including
        // its distortion under the same source targets and quantization step.
        let (base_bits, candidate_bits) = (base_price.bits(), candidate_price.bits());
        let lambda = f64::from(lambda);
        if candidate_bits + DC_RDO_MIN_GAIN_BITS >= base_bits
            || candidate_error + lambda * candidate_bits >= base_error + lambda * base_bits
        {
            for (dc, previous) in groups.iter_mut().zip(previous) {
                dc.quant_dc
                    .plane_mut(c)
                    .as_mut_slice()
                    .copy_from_slice(&previous);
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn dc_rounding_bounds<const NATIVE_ROUND: bool>(target: f32) -> [i64; 2] {
    if NATIVE_ROUND {
        [target.floor() as i64, target.ceil() as i64]
    } else {
        dc_bounds_from_truncation(target)
    }
}

#[inline]
fn dc_bounds_from_truncation(target: f32) -> [i64; 2] {
    // Every candidate is clamped to i16 below. Clamp before converting so a
    // single i32 truncation suffices, including targets outside the DC range.
    // Baseline x86 can do this without floorf/ceilf or SSE4.1.
    let bounded = target.clamp(f32::from(i16::MIN), f32::from(i16::MAX));
    let truncated = bounded as i32;
    let back = truncated as f32;
    [
        i64::from(truncated - i32::from(bounded < back)),
        i64::from(truncated + i32::from(bounded > back)),
    ]
}

#[allow(clippy::too_many_arguments)]
fn refine_channel(
    levels: &mut [i16],
    targets: &[f32],
    w: usize,
    h: usize,
    mchan: usize,
    predictors: &DcPredictorChoice,
    prices: &Prices,
    lambda: f32,
    emit: impl FnMut(Token),
) {
    // Dispatch once for the complete causal pass, outside the sample loop.
    #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
    if std::is_x86_feature_detected!("sse4.1") {
        // SAFETY: SSE4.1 support was checked above.
        unsafe {
            refine_channel_sse41(
                levels, targets, w, h, mchan, predictors, prices, lambda, emit,
            );
        }
        return;
    }
    const NATIVE_ROUND: bool =
        !cfg!(any(target_arch = "x86", target_arch = "x86_64")) || cfg!(target_feature = "sse4.1");
    refine_channel_core::<NATIVE_ROUND>(
        levels, targets, w, h, mchan, predictors, prices, lambda, emit,
    );
}

#[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn refine_channel_sse41(
    levels: &mut [i16],
    targets: &[f32],
    w: usize,
    h: usize,
    mchan: usize,
    predictors: &DcPredictorChoice,
    prices: &Prices,
    lambda: f32,
    emit: impl FnMut(Token),
) {
    refine_channel_core::<true>(
        levels, targets, w, h, mchan, predictors, prices, lambda, emit,
    );
}

// Inline the loop into the feature-specific entry point so floor/ceil inherit
// its SSE4.1 target features and lower to rounding instructions.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn refine_channel_core<const NATIVE_ROUND: bool>(
    levels: &mut [i16],
    targets: &[f32],
    w: usize,
    h: usize,
    mchan: usize,
    predictors: &DcPredictorChoice,
    prices: &Prices,
    lambda: f32,
    mut emit: impl FnMut(Token),
) {
    let mut wp = WpState::new(w);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let get = |x: usize, y: usize| i64::from(levels[y * w + x]);
            // The DC tokenizer replicates W at the top edge, N at the left
            // edge, and N for NE at the right edge. NN is N on the second row.
            let west = if x > 0 { get(x - 1, y) } else { 0 };
            let north = if y > 0 { get(x, y - 1) } else { west };
            let west = if x == 0 { north } else { west };
            let ne = if y > 0 {
                get((x + 1).min(w - 1), y - 1)
            } else {
                west
            };
            let nw = if y > 0 && x > 0 {
                get(x - 1, y - 1)
            } else {
                north
            };
            let nn = if y > 1 { get(x, y - 2) } else { north };
            let prediction = wp.predict(x, y, north, west, ne, nw, nn);
            let prop =
                (K_GRAD_RANGE_MID + wp.wp_prop).clamp(K_GRAD_RANGE_MIN, K_GRAD_RANGE_MAX) as usize;
            let context = K_GRADIENT_CONTEXT_LUT[prop] as usize;
            let prediction = if predictors[context] {
                (north + west - nw).clamp(north.min(west), north.max(west))
            } else {
                prediction
            };
            let target = targets[i];
            let current = i64::from(levels[i]);
            let cost = |level: i64| {
                let error = target - level as f32;
                error * error
                    + lambda * prices.residual_bits(mchan, context, (level - prediction) as i32)
            };
            let mut best = current;
            let mut best_cost = cost(current);
            let max_error = (target - current as f32).abs().max(DC_RDO_MAX_ERROR);
            if target.is_finite() {
                let [floor, ceil] = dc_rounding_bounds::<NATIVE_ROUND>(target);
                for candidate in [floor, ceil, current - 1, current + 1, prediction] {
                    let candidate = candidate.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
                    if (target - candidate as f32).abs() > max_error {
                        continue;
                    }
                    let candidate_cost = cost(candidate);
                    if candidate_cost < best_cost {
                        best = candidate;
                        best_cost = candidate_cost;
                    }
                }
            }
            levels[i] = best as i16;
            emit(Token::new(
                context as u32,
                pack_signed((best - prediction) as i32),
            ));
            wp.update(best, x, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prices() -> Prices {
        let mut bits = vec![[5.0; ALPHABET_SIZE]; 3 * K_NUM_DC_CONTEXTS];
        for row in &mut bits {
            row[0] = 0.0;
        }
        Prices { bits }
    }

    #[test]
    fn truncated_bounds_match_clamped_floor_and_ceil() {
        let check = |target: f32| {
            let clamp = |v: i64| v.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
            assert_eq!(
                dc_bounds_from_truncation(target),
                [clamp(target.floor() as i64), clamp(target.ceil() as i64)],
                "target={target:?}, bits={:08x}",
                target.to_bits(),
            );
        };
        for integer in -32770..=32770 {
            for fraction in [-0.75, -0.25, 0.0, 0.25, 0.75] {
                check(integer as f32 + fraction);
            }
        }
        for value in [
            f32::MIN,
            f32::MAX,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(1),
            -f32::from_bits(1),
            -0.0,
            i64::MIN as f32,
            i64::MAX as f32,
        ] {
            check(value);
        }
        let mut state = 0x85eb_ca6bu32;
        for _ in 0..100_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            check(f32::from_bits(state));
        }
    }

    #[test]
    fn rounding_dispatch_matches_portable_path() {
        for w in [1, 2, 3, 17] {
            for h in [1, 2, 4, 16] {
                for predictors in [DC_PREDICTOR_WEIGHTED, std::array::from_fn(|i| i % 2 == 0)] {
                    let original: Vec<i16> = (0..w * h).map(|i| (i * 7919) as i16).collect();
                    let targets: Vec<f32> = original
                        .iter()
                        .enumerate()
                        .map(|(i, &q)| match i % 19 {
                            0 => f32::from(i16::MIN) - 0.75,
                            1 => f32::from(i16::MAX) + 0.75,
                            2 => 1e20,
                            3 => -1e20,
                            4 => f32::NAN,
                            5 => f32::INFINITY,
                            6 => f32::NEG_INFINITY,
                            7 => f32::from_bits(1),
                            8 => -f32::from_bits(1),
                            9 => -0.0,
                            _ => f32::from(q) + [-0.51, -0.49, 0.0, 0.49, 0.51][i % 5],
                        })
                        .collect();
                    for mchan in [1, 2] {
                        for lambda in [0.04, 0.225] {
                            let mut portable = original.clone();
                            let mut dispatched = original.clone();
                            let mut portable_tokens = Vec::new();
                            let mut dispatched_tokens = Vec::new();
                            let prices = prices();
                            refine_channel_core::<false>(
                                &mut portable,
                                &targets,
                                w,
                                h,
                                mchan,
                                &predictors,
                                &prices,
                                lambda,
                                |t| portable_tokens.push((t.context, t.value)),
                            );
                            refine_channel(
                                &mut dispatched,
                                &targets,
                                w,
                                h,
                                mchan,
                                &predictors,
                                &prices,
                                lambda,
                                |t| dispatched_tokens.push((t.context, t.value)),
                            );
                            assert_eq!(portable, dispatched, "w={w}, h={h}");
                            assert_eq!(portable_tokens, dispatched_tokens, "w={w}, h={h}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rounding_favors_prediction_near_a_quantization_boundary() {
        let mut levels = [20, 21, 21];
        refine_channel(
            &mut levels,
            &[20.0, 20.51, 20.8],
            3,
            1,
            0,
            &DC_PREDICTOR_WEIGHTED,
            &prices(),
            0.02,
            |_| {},
        );
        assert_eq!(levels, [20, 20, 21]);
    }

    #[test]
    fn chosen_residuals_match_the_dc_tokenizer_at_every_edge() {
        for w in [1, 2, 3, 17] {
            for h in [1, 2, 4] {
                for predictors in [DC_PREDICTOR_WEIGHTED, std::array::from_fn(|i| i % 2 == 0)] {
                    let mut levels: Vec<i16> = (0..w * h)
                        .map(|i| ((i * 19 + i / w * 7) % 101) as i16 - 50)
                        .collect();
                    let targets: Vec<f32> = levels.iter().map(|&q| f32::from(q) - 0.49).collect();
                    let mut tokens = Vec::new();
                    refine_channel(
                        &mut levels,
                        &targets,
                        w,
                        h,
                        0,
                        &predictors,
                        &prices(),
                        0.04,
                        |t| tokens.push(t),
                    );
                    let mut expected = Vec::new();
                    super::super::walk_dc_channel(
                        &levels,
                        w,
                        h,
                        w,
                        0,
                        &predictors,
                        &mut Vec::new(),
                        false,
                        false,
                        &mut expected,
                        &mut Vec::new(),
                    );
                    assert_eq!(tokens.len(), expected.len());
                    for (a, b) in tokens.iter().zip(&expected) {
                        assert_eq!((a.context, a.value), (b.context, b.value), "w={w}, h={h}");
                    }
                }
            }
        }
    }

    #[test]
    fn rounding_handles_saturated_and_nonfinite_targets() {
        let mut levels = [i16::MAX, i16::MIN, 42, -7];
        refine_channel(
            &mut levels,
            &[1e20, -1e20, f32::NAN, f32::INFINITY],
            2,
            2,
            0,
            &DC_PREDICTOR_WEIGHTED,
            &prices(),
            0.04,
            |_| {},
        );
        assert_eq!(levels, [i16::MAX, i16::MIN, 42, -7]);
    }

    #[test]
    fn missing_sources_leave_every_group_unchanged() {
        let mut groups = [DcGroupData::new(3, 2).unwrap()];
        groups[0].quant_dc.plane_mut(1).as_mut_slice().fill(123);
        refine_dc_rounding(
            &mut groups,
            &super::super::compute_distance_params(4.0),
            0,
            CflFrame::XYB,
            false,
            &mut DcPredictorScratch::default(),
        )
        .unwrap();
        assert!(
            groups[0]
                .quant_dc
                .plane(1)
                .as_slice()
                .iter()
                .all(|&q| q == 123)
        );
    }

    #[test]
    fn constant_chroma_keeps_its_levels_without_a_rate_gain() {
        use crate::image::Plane;
        let params = super::super::compute_distance_params(4.0);
        let mut dc = DcGroupData::new(16, 8).unwrap();
        dc.quant_dc.plane_mut(0).as_mut_slice().fill(10);
        dc.source_dc_x = Some(Plane::new_fill(
            16,
            8,
            10.75 * params.dc_step[0] / (INV_DC_QUANT[0] * params.scale_dc),
        ));
        dc.source_dc_y = Some(Plane::new(16, 8));
        dc.source_dc_b = Some(Plane::new(16, 8));
        let mut groups = [dc];
        refine_dc_rounding(
            &mut groups,
            &params,
            0,
            CflFrame::XYB,
            false,
            &mut DcPredictorScratch::default(),
        )
        .unwrap();
        assert!(
            groups[0]
                .quant_dc
                .plane(0)
                .as_slice()
                .iter()
                .all(|&q| q == 10)
        );
        assert!(
            groups[0]
                .quant_dc
                .plane(1)
                .as_slice()
                .iter()
                .all(|&q| q == 0)
        );
    }
}
