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

//! ColorVideoVDP arbiter of the closed loop under [`crate::Arbiter::Cvvdp`].
//! The loop's reconstruction carries only the AC quantization error, so each
//! 8x8 block's DC is replaced by its decoded value before scoring. Whole-frame
//! arms are scored in full; per-candidate moves update an incremental state
//! that follows the accepted frame.

use crate::image::Image3F;
use crate::ssimulacra2::Rect;
use cvvdp_lite::{CvvdpLite, Display, IncrementalState, PreparedReference, RgbImage, Workspace};

/// A 500 nit laptop panel: bright enough that dark-region color errors stay
/// visible, unlike the 200 nit standard displays.
fn display() -> Display {
    Display::macbook_pro_16()
}

/// What a candidate update overwrote.
#[derive(Default)]
pub(crate) struct CvvdpCheckpoint {
    rect: Option<Rect>,
    linear: Vec<f32>,
    state: cvvdp_lite::Checkpoint,
}

pub(crate) struct CvvdpArbiter {
    reference: PreparedReference,
    workspace: Workspace,
    /// Incremental state of the accepted frame, once `reset`.
    state: Option<IncrementalState>,
    /// Linear RGB of the accepted frame (DC included).
    linear: [Vec<f32>; 3],
    /// Source mean of every 8x8 block (edge blocks over their visible pixels).
    source_dc: Image3F,
    width: usize,
    height: usize,
}

impl CvvdpArbiter {
    pub(crate) fn new(source: &Image3F) -> Option<Self> {
        let (width, height) = (source.xsize(), source.ysize());
        let linear = crate::ssimulacra2::linear_of_xyb(source);
        let planes = [&linear[0][..], &linear[1][..], &linear[2][..]];
        let metric = CvvdpLite::new(display()).ok()?;
        let mut workspace = Workspace::new();
        workspace.reserve(width, height).ok()?;
        let image = RgbImage::linear_rgb(planes, width, height, width).ok()?;
        let reference = metric.prepare_reference_with(image, &mut workspace).ok()?;
        let (xb, yb) = (width.div_ceil(8), height.div_ceil(8));
        let mut source_dc = Image3F::new(xb, yb);
        for c in 0..3 {
            for by in 0..yb {
                let rows = by * 8..(by * 8 + 8).min(height);
                let n_rows = rows.len() as f32;
                let out = source_dc.plane_row_mut(c, by);
                for (bx, v) in out[..xb].iter_mut().enumerate() {
                    let cols = bx * 8..(bx * 8 + 8).min(width);
                    let n = n_rows * cols.len() as f32;
                    let sum: f32 = rows
                        .clone()
                        .map(|y| source.plane_row(c, y)[cols.clone()].iter().sum::<f32>())
                        .sum();
                    *v = sum / n;
                }
            }
        }
        Some(Self {
            reference,
            workspace,
            state: None,
            linear: [
                vec![0.0; width * height],
                vec![0.0; width * height],
                vec![0.0; width * height],
            ],
            source_dc,
            width,
            height,
        })
    }

    /// Linear RGB of `recon` with every block's DC replaced by `decoded_dc`,
    /// written into `out` (frame-sized planes) over `r`.
    fn write_linear(
        &self,
        recon: &Image3F,
        decoded_dc: &Image3F,
        r: Rect,
        out: &mut [Vec<f32>; 3],
    ) {
        let w = self.width;
        let n = r.x1 - r.x0;
        let mut rows = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        let [r_out, g_out, b_out] = out;
        for y in r.y0..r.y1 {
            for (c, row) in rows.iter_mut().enumerate() {
                let dc = decoded_dc.plane_row(c, y / 8);
                let source = self.source_dc.plane_row(c, y / 8);
                let src = &recon.plane_row(c, y)[r.x0..r.x1];
                for ((v, &s), x) in row.iter_mut().zip(src).zip(r.x0..) {
                    *v = s + (dc[x / 8] - source[x / 8]);
                }
            }
            let span = y * w + r.x0..y * w + r.x1;
            crate::xyb::xyb_to_rgb_rows(
                &crate::xyb::XybMatrix::SPEC,
                [&rows[0][..], &rows[1][..], &rows[2][..]],
                [
                    &mut r_out[span.clone()],
                    &mut g_out[span.clone()],
                    &mut b_out[span],
                ],
            );
        }
    }

    fn full(&self) -> Rect {
        Rect {
            x0: 0,
            y0: 0,
            x1: self.width,
            y1: self.height,
        }
    }

    fn image<'a>(&self, linear: &'a [Vec<f32>; 3]) -> RgbImage<'a> {
        let planes = [&linear[0][..], &linear[1][..], &linear[2][..]];
        RgbImage::linear_rgb(planes, self.width, self.height, self.width).expect("decoded planes")
    }

    /// Pooled distortion (smaller is better) of `recon` with `decoded_dc`,
    /// independent of the accepted frame.
    pub(crate) fn deficit_of(&mut self, recon: &Image3F, decoded_dc: &Image3F) -> f32 {
        let (w, h) = (self.width, self.height);
        let mut linear = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
        self.write_linear(recon, decoded_dc, self.full(), &mut linear);
        let image = self.image(&linear);
        self.workspace
            .compare(&self.reference, image)
            .expect("prepared reference")
            .distortion as f32
    }

    /// Make `recon` with `decoded_dc` the accepted frame that updates follow.
    pub(crate) fn reset(&mut self, recon: &Image3F, decoded_dc: &Image3F) {
        let mut linear = std::mem::take(&mut self.linear);
        self.write_linear(recon, decoded_dc, self.full(), &mut linear);
        self.linear = linear;
        let image = self.image(&self.linear);
        self.state =
            Some(IncrementalState::new(&self.reference, image).expect("prepared reference"));
    }

    /// Pooled distortion of the accepted frame.
    pub(crate) fn deficit(&self) -> f32 {
        self.state.as_ref().expect("reset").score().distortion as f32
    }

    /// Score the accepted frame with `recon` (and `decoded_dc`) inside `r`,
    /// keeping it until `rollback(cp)`.
    pub(crate) fn update_checkpointed(
        &mut self,
        recon: &Image3F,
        decoded_dc: &Image3F,
        r: Rect,
        cp: &mut CvvdpCheckpoint,
    ) -> f32 {
        let w = self.width;
        cp.rect = Some(r);
        cp.linear.clear();
        for plane in &self.linear {
            for y in r.y0..r.y1 {
                cp.linear
                    .extend_from_slice(&plane[y * w + r.x0..y * w + r.x1]);
            }
        }
        let mut linear = std::mem::take(&mut self.linear);
        self.write_linear(recon, decoded_dc, r, &mut linear);
        self.linear = linear;
        let image = self.image(&self.linear);
        let rect = cvvdp_lite::Rect::new(r.x0, r.y0, r.x1, r.y1);
        self.state
            .as_mut()
            .expect("reset")
            .update_checkpointed(&self.reference, image, rect, &mut cp.state)
            .expect("prepared reference")
            .distortion as f32
    }

    /// Undo the `update_checkpointed` that filled `cp`.
    pub(crate) fn rollback(&mut self, cp: &CvvdpCheckpoint) {
        let Some(r) = cp.rect else {
            return;
        };
        let w = self.width;
        let n = r.x1 - r.x0;
        let mut at = 0;
        for plane in &mut self.linear {
            for y in r.y0..r.y1 {
                plane[y * w + r.x0..y * w + r.x1].copy_from_slice(&cp.linear[at..at + n]);
                at += n;
            }
        }
        self.state.as_mut().expect("reset").rollback(&cp.state);
    }
}
