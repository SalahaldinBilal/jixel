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

//! The exact metric the closed loop arbitrates candidates with: an
//! incremental state following the accepted frame, on the arbiter's metric.

use crate::cvvdp_arbiter::{CvvdpCheckpoint, CvvdpArbiter};
use crate::image::Image3F;
use crate::ssimulacra2::{Rect, Ssimulacra2Checkpoint, Ssimulacra2State};

pub(super) enum LoopJudge {
    Ssimulacra2(Box<Ssimulacra2State>),
    Cvvdp(Box<CvvdpArbiter>),
}

pub(super) enum LoopCheckpoint {
    Ssimulacra2(Ssimulacra2Checkpoint),
    Cvvdp(CvvdpCheckpoint),
}

impl LoopJudge {
    /// Deficit of the accepted frame (smaller is better).
    pub(super) fn deficit(&self) -> f32 {
        match self {
            Self::Ssimulacra2(s) => s.deficit(),
            Self::Cvvdp(j) => j.deficit(),
        }
    }

    /// Whether candidates must supply their decoded DC.
    pub(super) fn needs_dc(&self) -> bool {
        matches!(self, Self::Cvvdp(_))
    }

    pub(super) fn checkpoint(&self) -> LoopCheckpoint {
        match self {
            Self::Ssimulacra2(_) => LoopCheckpoint::Ssimulacra2(Default::default()),
            Self::Cvvdp(_) => LoopCheckpoint::Cvvdp(Default::default()),
        }
    }

    /// Deficit with `recon` (and its decoded DC) inside `r`, kept until
    /// `rollback(cp)`.
    pub(super) fn update_checkpointed(
        &mut self,
        recon: &Image3F,
        decoded_dc: Option<&Image3F>,
        r: Rect,
        cp: &mut LoopCheckpoint,
    ) -> f32 {
        match (self, cp) {
            (Self::Ssimulacra2(s), LoopCheckpoint::Ssimulacra2(cp)) => {
                s.update_checkpointed(recon, r, cp);
                s.deficit()
            }
            (Self::Cvvdp(j), LoopCheckpoint::Cvvdp(cp)) => {
                j.update_checkpointed(recon, decoded_dc.expect("decoded DC"), r, cp)
            }
            _ => unreachable!("checkpoint of another judge"),
        }
    }

    pub(super) fn rollback(&mut self, cp: &LoopCheckpoint) {
        match (self, cp) {
            (Self::Ssimulacra2(s), LoopCheckpoint::Ssimulacra2(cp)) => s.rollback(cp),
            (Self::Cvvdp(j), LoopCheckpoint::Cvvdp(cp)) => j.rollback(cp),
            _ => unreachable!("checkpoint of another judge"),
        }
    }

    /// The SSIMULACRA2 per-pixel map (empty for other metrics).
    pub(super) fn map(&self) -> Vec<f32> {
        match self {
            Self::Ssimulacra2(s) => s.map(),
            Self::Cvvdp(_) => Vec::new(),
        }
    }

    pub(super) fn cvvdp(&mut self) -> Option<&mut CvvdpArbiter> {
        match self {
            Self::Cvvdp(j) => Some(j),
            Self::Ssimulacra2(_) => None,
        }
    }
}
