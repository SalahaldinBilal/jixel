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

//! Per-arbiter tuning of the lossy encoder. Every decision whose best setting
//! depends on the target metric reads one field here, so an [`Arbiter`] is
//! just a row of this table.

use crate::Arbiter;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tuning {
    /// Base of the AC trellis rate price, in units of `RD_LAMBDA / 4`.
    pub(crate) rdoq_lambda: f32,
    /// Distance from which the finer 16/32 and coarser DCT8 B tables apply.
    pub(crate) qm_ss2_distance: f32,
    /// Let the reconstruction rerank split merged transforms back into DCT8s.
    pub(crate) rerank_downgrades: bool,
}

const SSIMULACRA2: Tuning = Tuning {
    rdoq_lambda: 1.5,
    qm_ss2_distance: crate::quant_weights::QM_SS2_MIN_DISTANCE,
    rerank_downgrades: true,
};

const CVVDP: Tuning = Tuning {
    rdoq_lambda: 1.25,
    qm_ss2_distance: 1.75,
    rerank_downgrades: false,
};

impl Arbiter {
    pub(crate) const fn tuning(self) -> &'static Tuning {
        match self {
            Arbiter::Ssimulacra2 => &SSIMULACRA2,
            Arbiter::Cvvdp => &CVVDP,
        }
    }
}
