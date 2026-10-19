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

//! Tests for lz77.

use super::{
    LZ77_DIST_VALUE, LZ77_NUM_SPECIAL_DISTANCES, LZ77_WINDOW, lz77_compress_with_depth_into_limit,
};
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;

#[test]
fn deep_matcher_never_reaches_past_the_decoder_window() {
    // A repeated 64-token motif separated by more than a window of
    // unique filler: the second copy may not reference the first.
    let motif: Vec<Token> = (0..64).map(|i| Token::new(0, 1000 + i)).collect();
    let mut tokens = motif.clone();
    tokens.extend((0..LZ77_WINDOW as u32 + 100).map(|i| Token::new(0, 2000 + (i % 50_000))));
    tokens.extend(motif.iter().copied());
    let mut scratch = CoderScratch::default();
    let mut out = Vec::new();
    assert!(lz77_compress_with_depth_into_limit(
        &tokens,
        8,
        &mut scratch.lz_depth,
        &mut out,
        usize::MAX
    ));
    for token in &out {
        if token.is_lz77() && token.distance != LZ77_DIST_VALUE {
            let distance = (token.distance - LZ77_NUM_SPECIAL_DISTANCES + 1) as usize;
            assert!(
                distance <= LZ77_WINDOW,
                "distance {distance} exceeds the window"
            );
        }
    }
}
