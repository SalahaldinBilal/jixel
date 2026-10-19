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

use super::constant_alpha_value;
use super::context::{build_context_tree, emit_ct_tree};
use super::lz77::ConstantTail;
use super::ma_learning::finish_ma_candidate;
use super::ma_lookup::MaLookup;
use super::tree::{
    TREE_CTX_OFFSET, build_balanced_tree_tokens, build_balanced_tree_tokens_offsets,
};
use crate::encode_image::AlphaPlane;
use crate::entropy::pack_signed;
use crate::ma_tree::{LearnedTree, MaNode, NUM_MA_PROPS};

#[test]
fn constant_alpha_value_detects_solid_planes() {
    assert_eq!(
        constant_alpha_value(&AlphaPlane::U8(vec![255; 10])),
        Some(255)
    );
    assert_eq!(constant_alpha_value(&AlphaPlane::U8(vec![255, 254])), None);
    assert_eq!(constant_alpha_value(&AlphaPlane::U8(Vec::new())), None);
    assert_eq!(
        constant_alpha_value(&AlphaPlane::U16 {
            data: vec![65535; 3],
            bits: 16
        }),
        Some(65535)
    );
    assert_eq!(
        constant_alpha_value(&AlphaPlane::F32(vec![-7, -7])),
        Some(-7)
    );
}

#[test]
fn with_constant_channel_routes_the_channel_to_a_zero_leaf() {
    let tree = LearnedTree {
        nodes: vec![
            MaNode::Split {
                prop: 2,
                val: 10,
                gt: 1,
                le: 2,
            },
            MaNode::Leaf { pred: 5, offset: 0 },
            MaNode::Leaf { pred: 6, offset: 0 },
        ],
        est_bits: 100.0,
        flat_bits: 200.0,
    };
    let mut cand = finish_ma_candidate(tree, 1.0, None);
    cand.leaf_offset = vec![-3, 4];
    let wrapped = cand.with_constant_channel(3, 255);
    assert_eq!(wrapped.num_ctx, 3);
    assert_eq!(wrapped.constant_ctx, Some(0));
    assert_eq!(wrapped.leaf_offset, vec![255, -3, 4]);
    assert_eq!(wrapped.est_real, cand.est_real);
    let offsets: Vec<u32> = wrapped
        .tree_tokens
        .iter()
        .filter(|token| token.context == TREE_CTX_OFFSET)
        .map(|token| token.value)
        .collect();
    assert_eq!(
        offsets,
        vec![pack_signed(255), pack_signed(-3), pack_signed(4)]
    );
    assert_eq!(
        wrapped.constant_tail(4, 5),
        ConstantTail {
            context: 0,
            count: 20
        }
    );

    let mut props = [0i32; NUM_MA_PROPS];
    let alpha = MaLookup::new(&wrapped.tree, &wrapped.leaf_ctx, 3, 0);
    assert!(!alpha.needs_wp());
    props[2] = 20;
    assert_eq!(alpha.lookup(&props), (0, 0));
    props[2] = 0;
    assert_eq!(alpha.lookup(&props), (0, 0));
    for chan in 0..3 {
        let color = MaLookup::new(&wrapped.tree, &wrapped.leaf_ctx, chan, 0);
        props[2] = 20;
        assert_eq!(color.lookup(&props), (1, 5));
        props[2] = 0;
        assert_eq!(color.lookup(&props), (2, 6));
    }
}

#[test]
fn balanced_tree_offsets_follow_the_bfs_leaf_order() {
    let tokens = build_balanced_tree_tokens_offsets(&[5, 5, 5, 0], &[1, 2, 3, 255]);
    let offsets: Vec<u32> = tokens
        .iter()
        .filter(|token| token.context == TREE_CTX_OFFSET)
        .map(|token| token.value)
        .collect();
    // Leaves are emitted chan 3, 2, 1, 0.
    assert_eq!(
        offsets,
        vec![
            pack_signed(255),
            pack_signed(3),
            pack_signed(2),
            pack_signed(1)
        ]
    );
    let plain = build_balanced_tree_tokens(&[5, 5, 5, 0]);
    assert_eq!(tokens.len(), plain.len());
}

#[test]
fn context_tree_constant_leaf_shares_one_context() {
    let tree = build_context_tree(4, &[5, 5, 5, 0], &[8, 8, 8, 8], Some(255));
    let mut tokens = Vec::new();
    let map = emit_ct_tree(&tree, &mut tokens);
    assert_eq!(map[&9], map[&10]);
    assert_eq!(map[&10], map[&11]);
    let num_ctx = map.values().max().map_or(0, |&m| m + 1);
    assert_eq!(num_ctx, 10);
    assert!(
        tokens
            .iter()
            .any(|token| token.context == TREE_CTX_OFFSET && token.value == pack_signed(255))
    );
    let plain = build_context_tree(4, &[5, 5, 5, 0], &[8, 8, 8, 8], None);
    let mut plain_tokens = Vec::new();
    let plain_map = emit_ct_tree(&plain, &mut plain_tokens);
    assert_eq!(plain_map.values().max(), Some(&11));
}
