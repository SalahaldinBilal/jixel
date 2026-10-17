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

//! Ink layer: the dominant single-color thin strokes of an image (the ink of
//! cel and line art) lifted into a second frame. The base frame is free under
//! the ink, so VarDCT no longer pays for the dark lines; the ink frame is one
//! constant color plus a few-level coverage alpha, blended with kBlend by the
//! decoder in the output color space.

/// Half the side of the median window that defines a thin feature.
const MEDIAN_RADIUS: usize = 5;
/// A pixel is thin when its luma sits this far from the window median.
const THIN_DEV: i32 = 24;
/// Max per-channel distance from the ink color for a pixel to join a stroke;
/// lighter antialiasing pixels become partial coverage instead.
const COLOR_TOL: i32 = 20;
/// Strokes smaller than this are left to VarDCT.
const MIN_COMPONENT: usize = 16;
/// How far a stroke may grow from its thin seeds into same-colored pixels.
const MAX_GROW: u8 = 3;
/// The dominant color must hold this share of the thin pixels.
const MIN_DOMINANT_SHARE: f32 = 0.3;
/// The ink must cover this share of the image.
const MIN_CORE_SHARE: f32 = 0.005;
/// Mean per-pixel distance (max over channels) of the strokes from the ink
/// color above which the lines are painted in several hues, not inked.
const MAX_INK_DEVIATION: f32 = 8.0;
/// Coverage steps above zero: alpha = step * 255 / COVERAGE_STEPS.
const COVERAGE_STEPS: u32 = 7;
const KMEANS_K: usize = 8;
const KMEANS_SAMPLES: usize = 40_000;
const DIFFUSION_ITERS: usize = 60;

pub(crate) struct InkSplit {
    /// Interleaved RGB of the base frame: the ink core is a smooth fill, the
    /// antialiasing ring is un-premultiplied.
    pub(crate) base: Vec<u8>,
    /// Coverage alpha of the ink frame, on `COVERAGE_STEPS + 1` levels.
    pub(crate) coverage: Vec<u8>,
    pub(crate) ink: [u8; 3],
}

const UNVISITED: u8 = 0;
const CORE: u8 = 1;
const REJECTED: u8 = 2;
const NEAR_CORE: u8 = 3;
const UNGROWN: u8 = MAX_GROW + 1;

pub(crate) fn split(rgb: &[u8], width: usize, height: usize) -> Option<InkSplit> {
    let n = width * height;
    if n < 64 * 64 {
        return None;
    }
    let rgb = rgb.as_chunks::<3>().0;
    let (mut state, thin_count) = thin_pixels(rgb, width, height);
    if thin_count < MIN_COMPONENT {
        return None;
    }
    let (ink, share) = dominant_color(rgb, &state, thin_count)?;
    if share < MIN_DOMINANT_SHARE {
        return None;
    }
    // Reuse the thin mask for growth depth, component state, then fill state.
    // Classify colors once, including neighbors reached by several seeds.
    let mut queue = Vec::with_capacity(thin_count);
    for (i, (s, p)) in state.iter_mut().zip(rgb).enumerate() {
        let near_ink = p
            .iter()
            .zip(&ink)
            .all(|(&v, &c)| (v as i32 - c as i32).abs() < COLOR_TOL);
        *s = if !near_ink {
            u8::MAX
        } else if *s != 0 {
            queue.push(i);
            0
        } else {
            UNGROWN
        };
    }
    // Breadth first from every thin seed, 4-connected. A Vec with a read
    // cursor also serves as the component worklist below.
    let mut head = 0;
    while let Some(&i) = queue.get(head) {
        head += 1;
        let d = state[i];
        if d >= MAX_GROW {
            continue;
        }
        let x = i % width;
        let mut visit = |j: usize| {
            if state[j] == UNGROWN {
                state[j] = d + 1;
                queue.push(j);
            }
        };
        if x > 0 {
            visit(i - 1);
        }
        if x + 1 < width {
            visit(i + 1);
        }
        if i >= width {
            visit(i - width);
        }
        if i + width < n {
            visit(i + width);
        }
    }
    for s in &mut state {
        *s = if *s <= MAX_GROW { UNVISITED } else { REJECTED };
    }
    // Drop specks and accumulate the accepted cores' statistics in the same
    // traversal, without allocating a second mask or component buffer.
    let mut core_count = 0;
    let mut deviation = 0u64;
    let mut seed = 0;
    while let Some(offset) = state[seed..].iter().position(|&s| s == UNVISITED) {
        seed += offset;
        queue.clear();
        queue.push(seed);
        state[seed] = CORE;
        let mut component_deviation = 0u64;
        let mut head = 0;
        while let Some(&i) = queue.get(head) {
            head += 1;
            component_deviation += rgb[i]
                .iter()
                .zip(&ink)
                .map(|(&v, &c)| v.abs_diff(c) as u64)
                .max()
                .unwrap();
            let x = i % width;
            let mut visit = |j: usize| {
                if state[j] == UNVISITED {
                    state[j] = CORE;
                    queue.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < width {
                visit(i + 1);
            }
            if i >= width {
                visit(i - width);
            }
            if i + width < n {
                visit(i + width);
            }
        }
        if queue.len() < MIN_COMPONENT {
            for &i in &queue {
                state[i] = REJECTED;
            }
        } else {
            core_count += queue.len();
            deviation += component_deviation;
        }
        seed += 1;
    }
    if (core_count as f32) < MIN_CORE_SHARE * n as f32 {
        return None;
    }
    // One flat color must reproduce the strokes: painted lines of several
    // hues stay in the base.
    if deviation as f32 > MAX_INK_DEVIATION * core_count as f32 {
        return None;
    }
    // Cache the 3x3 core dilation in the same mask. The 5x5 fill windows
    // below can now test adjacency with a single load per pixel.
    for i in 0..n {
        if state[i] != CORE {
            continue;
        }
        let (x, y) = (i % width, i / width);
        let x0 = x.saturating_sub(1);
        let x1 = (x + 2).min(width);
        let y0 = y.saturating_sub(1);
        let y1 = (y + 2).min(height);
        for row in state[y0 * width..y1 * width].chunks_exact_mut(width) {
            for s in &mut row[x0..x1] {
                if *s == REJECTED {
                    *s = NEAR_CORE;
                }
            }
        }
    }
    let inkf = ink.map(f32::from);
    let mut coverage = vec![0u8; n];
    let mut base: Vec<[f32; 3]> = rgb.iter().map(|p| p.map(f32::from)).collect();
    for (i, (((p, base), alpha), &s)) in rgb
        .iter()
        .zip(&mut base)
        .zip(&mut coverage)
        .zip(&state)
        .enumerate()
    {
        if s == CORE {
            *alpha = 255;
            continue;
        }
        if s != NEAR_CORE {
            continue;
        }
        let (x, y) = (i % width, i / width);
        // Local fill color: the pixels of the 5x5 window that are neither
        // ink nor next to it.
        let mut fill = [0f32; 3];
        let mut count = 0usize;
        for (pixels, states) in neighborhood(rgb, width, height, x, y, 2)
            .zip(neighborhood(&state, width, height, x, y, 2))
        {
            for (p, &s) in pixels.iter().zip(states) {
                if s != REJECTED {
                    continue;
                }
                for (f, &v) in fill.iter_mut().zip(p) {
                    *f += v as f32;
                }
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        for f in &mut fill {
            *f /= count as f32;
        }
        let d: [f32; 3] = std::array::from_fn(|c| fill[c] - inkf[c]);
        let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if dd < 100.0 {
            continue;
        }
        let num = fill
            .iter()
            .zip(p)
            .zip(&d)
            .map(|((&f, &v), &d)| (f - v as f32) * d)
            .sum::<f32>();
        let c = (num / dd).clamp(0.0, 1.0);
        let step = (c * COVERAGE_STEPS as f32).round() as u32;
        if step == 0 {
            continue;
        }
        *alpha = ((step * 255 + COVERAGE_STEPS / 2) / COVERAGE_STEPS) as u8;
        if c < 0.5 {
            // Mostly fill: un-premultiply with the measured coverage so the
            // base carries the fill color under the antialiasing.
            for ((b, &v), &ink) in base.iter_mut().zip(p).zip(&inkf) {
                *b = ((v as f32 - ink * c) / (1.0 - c)).clamp(0.0, 255.0);
            }
        }
    }
    // With seven coverage steps, alpha >= 128 exactly identifies c >= 0.5,
    // including the opaque core. Everything else is fixed during the fill.
    for (s, &alpha) in state.iter_mut().zip(&coverage) {
        *s = u8::from(alpha < 128);
    }
    fill_dont_care(&mut base, &mut state, width, height, &mut queue);
    Some(InkSplit {
        base: base
            .as_flattened()
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
        coverage,
        ink,
    })
}

/// Rows of a clipped square neighborhood, with bounds checked per row
/// instead of per channel of every pixel.
fn neighborhood<T>(
    pixels: &[T],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
) -> impl Iterator<Item = &[T]> {
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(height);
    pixels[y0 * width..y1 * width]
        .chunks_exact(width)
        .map(move |row| &row[x0..x1])
}

/// Thin features from an 11x11 sliding luma histogram. A coarse histogram
/// skips groups of 16 bins during median selection; emit the mask directly.
fn thin_pixels(rgb: &[[u8; 3]], width: usize, height: usize) -> (Vec<u8>, usize) {
    let luma: Vec<u8> = rgb
        .iter()
        .map(|&[r, g, b]| ((77 * r as u32 + 150 * g as u32 + 29 * b as u32 + 128) >> 8) as u8)
        .collect();
    let mut thin = vec![0u8; width * height];
    let mut thin_count = 0;
    let r = MEDIAN_RADIUS;
    for (y, (out, center)) in thin
        .chunks_exact_mut(width)
        .zip(luma.chunks_exact(width))
        .enumerate()
    {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(height);
        let rows = &luma[y0 * width..y1 * width];
        // At most 121 pixels in the window.
        let mut hist = [0u16; 256];
        let mut coarse = [0u16; 16];
        for row in rows.chunks_exact(width) {
            for &v in &row[..(r + 1).min(width)] {
                hist[v as usize] += 1;
                coarse[v as usize / 16] += 1;
            }
        }
        let mut count = ((r + 1).min(width) * (y1 - y0)) as u16;
        for (x, (out, &l)) in out.iter_mut().zip(center).enumerate() {
            if x > 0 {
                if x + r < width {
                    for row in rows.chunks_exact(width) {
                        let v = row[x + r] as usize;
                        hist[v] += 1;
                        coarse[v / 16] += 1;
                    }
                    count += (y1 - y0) as u16;
                }
                if x > r {
                    for row in rows.chunks_exact(width) {
                        let v = row[x - r - 1] as usize;
                        hist[v] -= 1;
                        coarse[v / 16] -= 1;
                    }
                    count -= (y1 - y0) as u16;
                }
            }
            let mut target = count.div_ceil(2);
            let mut select = |&bin: &u16| {
                if target <= bin {
                    true
                } else {
                    target -= bin;
                    false
                }
            };
            let group = coarse.iter().position(&mut select).unwrap();
            let bin = hist[group * 16..][..16].iter().position(select).unwrap();
            let median = (group * 16 + bin) as i32;
            *out = u8::from((l as i32 - median).abs() > THIN_DEV);
            thin_count += *out as usize;
        }
    }
    (thin, thin_count)
}

/// k-means over the thin pixels; returns the largest cluster's color and
/// its share of the samples.
fn dominant_color(rgb: &[[u8; 3]], thin: &[u8], thin_count: usize) -> Option<([u8; 3], f32)> {
    let stride = (thin_count / KMEANS_SAMPLES).max(1);
    let mut samples = Vec::with_capacity(thin_count.div_ceil(stride));
    samples.extend(
        rgb.iter()
            .zip(thin)
            .filter(|&(_, &t)| t != 0)
            .step_by(stride)
            .map(|(p, _)| p.map(f32::from)),
    );
    if samples.len() < KMEANS_K {
        return None;
    }
    let mut centroids: [[f32; 3]; KMEANS_K] =
        std::array::from_fn(|k| samples[k * samples.len() / KMEANS_K]);
    let mut counts = [0usize; KMEANS_K];
    for _ in 0..20 {
        let mut sums = [[0f32; 3]; KMEANS_K];
        counts.fill(0);
        // Assignment and accumulation use the same sample order, so no
        // assignment buffer or second pass over the samples is needed.
        for s in &samples {
            let mut best = (f32::MAX, 0);
            for (k, c) in centroids.iter().enumerate() {
                let d = s
                    .iter()
                    .zip(c)
                    .map(|(&v, &c)| (v - c) * (v - c))
                    .sum::<f32>();
                if d < best.0 {
                    best = (d, k);
                }
            }
            for (sum, &v) in sums[best.1].iter_mut().zip(s) {
                *sum += v;
            }
            counts[best.1] += 1;
        }
        let mut changed = false;
        for ((centroid, sum), &count) in centroids.iter_mut().zip(&sums).zip(&counts) {
            if count > 0 {
                let next = sum.map(|v| v / count as f32);
                changed |= *centroid != next;
                *centroid = next;
            }
        }
        if !changed {
            break;
        }
    }
    let (k, &count) = counts.iter().enumerate().max_by_key(|&(_, &c)| c)?;
    Some((
        centroids[k].map(|v| v.round() as u8),
        count as f32 / samples.len() as f32,
    ))
}

/// Replace the free pixels of `base` by a smooth continuation of their
/// surroundings: a boundary-inward fill followed by box diffusion.
/// `filled` starts at zero for free pixels and one for fixed pixels.
fn fill_dont_care(
    base: &mut [[f32; 3]],
    filled: &mut [u8],
    width: usize,
    height: usize,
    dont_care: &mut Vec<usize>,
) {
    dont_care.clear();
    dont_care.extend(
        filled
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| (s == 0).then_some(i)),
    );
    if dont_care.is_empty() {
        return;
    }
    let neighbors = |i: usize| {
        let (x, y) = (i % width, i / width);
        (y.saturating_sub(1)..(y + 2).min(height)).flat_map(move |yy| {
            (x.saturating_sub(1)..(x + 2).min(width)).map(move |xx| yy * width + xx)
        })
    };
    let mut frontier: Vec<usize> = dont_care
        .iter()
        .copied()
        .filter(|&i| neighbors(i).any(|j| filled[j] == 1))
        .collect();
    // Inward fill in rings: each ring takes the mean of its filled neighbors.
    // State 2 claims a pixel for the next ring without making its value visible.
    let mut values: Vec<[f32; 3]> = Vec::with_capacity(frontier.len());
    let mut next = Vec::new();
    while !frontier.is_empty() {
        values.clear();
        for &i in &frontier {
            let mut sum = [0f32; 3];
            let mut count = 0usize;
            let (x, y) = (i % width, i / width);
            for (pixels, states) in neighborhood(base, width, height, x, y, 1)
                .zip(neighborhood(filled, width, height, x, y, 1))
            {
                for (p, &s) in pixels.iter().zip(states) {
                    if s == 1 {
                        for (sum, &v) in sum.iter_mut().zip(p) {
                            *sum += v;
                        }
                        count += 1;
                    }
                }
            }
            values.push(sum.map(|v| v / count.max(1) as f32));
        }
        for (&i, &v) in frontier.iter().zip(&values) {
            base[i] = v;
            filled[i] = 1;
        }
        next.clear();
        for &i in &frontier {
            for j in neighbors(i) {
                if filled[j] == 0 {
                    filled[j] = 2;
                    next.push(j);
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next);
    }
    drop(frontier);
    drop(next);
    // Reuse the ring values for Jacobi diffusion; pixels outside the free
    // set stay fixed and each iteration reads only the preceding iteration.
    values.resize(dont_care.len(), [0f32; 3]);
    for _ in 0..DIFFUSION_ITERS {
        for (slot, &i) in values.iter_mut().zip(dont_care.iter()) {
            let mut sum = [0f32; 3];
            let mut count = 0usize;
            let (x, y) = (i % width, i / width);
            for row in neighborhood(base, width, height, x, y, 1) {
                for p in row {
                    for (sum, &v) in sum.iter_mut().zip(p) {
                        *sum += v;
                    }
                }
                count += row.len();
            }
            *slot = sum.map(|v| v / count as f32);
        }
        for (&v, &i) in values.iter().zip(dont_care.iter()) {
            base[i] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_pixels_match_sorted_windows_at_edges() {
        for (width, height) in [(1, 19), (19, 1), (2, 7), (7, 2), (11, 13), (23, 17)] {
            let rgb: Vec<_> = (0..width * height)
                .map(|i| [((i * 73 + i / width * 31) % 256) as u8; 3])
                .collect();
            let expected: Vec<_> = rgb
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let (x, y) = (i % width, i / width);
                    let mut window = Vec::new();
                    for yy in y.saturating_sub(MEDIAN_RADIUS)..(y + MEDIAN_RADIUS + 1).min(height) {
                        for xx in
                            x.saturating_sub(MEDIAN_RADIUS)..(x + MEDIAN_RADIUS + 1).min(width)
                        {
                            window.push(rgb[yy * width + xx][0]);
                        }
                    }
                    window.sort_unstable();
                    let median = window[(window.len() - 1) / 2];
                    u8::from((p[0] as i32 - median as i32).abs() > THIN_DEV)
                })
                .collect();
            let (thin, count) = thin_pixels(&rgb, width, height);
            assert_eq!(thin, expected, "{width}x{height}");
            assert_eq!(count, expected.iter().map(|&v| v as usize).sum());
        }
    }

    #[test]
    fn inward_fill_reaches_free_pixels_and_preserves_fixed_pixels() {
        for (width, height) in [(1, 17), (17, 1), (9, 7)] {
            let color = [17.0, 83.0, 201.0];
            let mut base = vec![[0.0; 3]; width * height];
            base[0] = color;
            let mut filled = vec![0; base.len()];
            filled[0] = 1;
            let mut work = Vec::new();
            fill_dont_care(&mut base, &mut filled, width, height, &mut work);
            assert!(base.iter().all(|&p| p == color));

            // The empty free set and a free set without a boundary are valid.
            filled.fill(1);
            fill_dont_care(&mut base, &mut filled, width, height, &mut work);
            assert!(work.is_empty());
            filled.fill(0);
            fill_dont_care(&mut base, &mut filled, width, height, &mut work);
            assert!(base.iter().all(|&p| p == color));
        }
    }

    #[test]
    fn line_art_split_preserves_base_and_coverage() {
        let (width, height) = (79, 83);
        let rgb: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let background = [180 + (x % 61) as u8, 160 + (y % 81) as u8, 210];
                    let ink = [12, 18, 24];
                    let d = (x + y / 3) % 31;
                    if d < 3 || y % 47 < 2 || (x % 101 < 5 && y % 113 < 5) {
                        ink
                    } else if d == 3 || y % 47 == 2 {
                        std::array::from_fn(|c| ((background[c] as u16 + ink[c] as u16) / 2) as u8)
                    } else {
                        background
                    }
                })
            })
            .collect();
        let split = split(&rgb, width, height).unwrap();
        let hash = |bytes: &[u8]| {
            bytes.iter().fold(0xcbf29ce484222325u64, |hash, &v| {
                (hash ^ v as u64).wrapping_mul(0x100000001b3)
            })
        };
        assert_eq!(split.ink, [12, 18, 24]);
        // Captured from the original implementation; includes clipped strokes,
        // antialiasing, and the synchronous diffusion result.
        assert_eq!(hash(&split.coverage), 0xbab21cc039d73c55);
        assert_eq!(hash(&split.base), 0x5d12c4e53c402f1b);
    }

    #[test]
    fn split_rejects_small_images_flat_images_and_specks() {
        assert!(split(&vec![200; 32 * 32 * 3], 32, 32).is_none());
        assert!(split(&vec![200; 64 * 64 * 3], 64, 64).is_none());
        let rgb: Vec<u8> = (0..64)
            .flat_map(|y| {
                (0..64).flat_map(move |x| [if x % 8 == 0 && y % 8 == 0 { 0 } else { 200 }; 3])
            })
            .collect();
        assert!(split(&rgb, 64, 64).is_none());
    }
}
