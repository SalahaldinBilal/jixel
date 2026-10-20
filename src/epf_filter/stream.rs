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

//! Bounded row rings for all three EPF stages. Later stages update the output
//! in place; only the rows crossing worker boundaries need a snapshot.
use super::kernels::{FilterRow, Kernels, Neighbor};
use super::{LoopFilter, Stage, mirror};
use crate::coder_scratch::CoderScratch;
use crate::image::{Image3F, Plane};
use crate::thread_pool::ThreadPool;

fn grow(data: &mut Vec<f32>, len: usize) {
    if data.len() < len {
        data.resize(len, 0.0);
    }
}
fn slot(row: isize, rows: usize, stride: usize) -> usize {
    row.rem_euclid(rows as isize) as usize * stride
}

#[derive(Default)]
pub(super) struct Scratch {
    pixels: [Vec<f32>; 3],
    difference: [Vec<f32>; 6],
    sad: [Vec<f32>; 6],
    halo: [Vec<f32>; 3],
}

impl Scratch {
    fn prepare(&mut self, stage: &Stage, w: usize) {
        let stride = w + 2 * stage.border;
        for p in &mut self.pixels {
            grow(p, (2 * stage.border + 1) * stride);
        }
        for (i, &(_, dy)) in stage.dirs.iter().enumerate() {
            if stage.patch {
                grow(&mut self.difference[i], 3 * stride);
            }
            grow(&mut self.sad[i], (dy as usize + 1) * stride);
        }
    }
    fn snapshot(&mut self, image: &Image3F, first: usize, rows: usize, b: usize) {
        let w = image.xsize();
        for c in 0..3 {
            grow(&mut self.halo[c], 2 * b * w);
            for i in 0..2 * b {
                let y = if i < b {
                    first as isize + i as isize - b as isize
                } else {
                    (first + rows + i - b) as isize
                };
                self.halo[c][i * w..(i + 1) * w]
                    .copy_from_slice(image.plane_row(c, mirror(y, image.ysize() as isize)));
            }
        }
    }
    fn load_row(
        &mut self,
        row: isize,
        source: Option<&Image3F>,
        band: &[&mut [f32]; 3],
        first: usize,
        rows: usize,
        w: usize,
        b: usize,
    ) {
        let stride = w + 2 * b;
        let start = slot(row, 2 * b + 1, stride);
        for c in 0..3 {
            let src = if let Some(image) = source {
                image.plane_row(c, mirror(first as isize + row, image.ysize() as isize))
            } else if row < 0 {
                let i = (row + b as isize) as usize * w;
                &self.halo[c][i..i + w]
            } else if row >= rows as isize {
                let i = (b + row as usize - rows) * w;
                &self.halo[c][i..i + w]
            } else {
                let i = row as usize * w;
                &band[c][i..i + w]
            };
            let dst = &mut self.pixels[c][start..start + stride];
            dst[b..b + w].copy_from_slice(src);
            for x in 0..b {
                dst[b - 1 - x] = src[mirror(-1 - x as isize, w as isize)];
                dst[b + w + x] = src[mirror((w + x) as isize, w as isize)];
            }
        }
    }
    fn difference(
        &mut self,
        stage: &Stage,
        dir: usize,
        row: isize,
        w: usize,
        lf: &LoopFilter,
        k: Kernels,
    ) {
        let (dx, dy) = stage.dirs[dir];
        let stride = w + 2 * stage.border;
        let lo = (-dx).max(0) as usize;
        let hi = stride - dx.max(0) as usize;
        let a = slot(row, 2 * stage.border + 1, stride) + lo;
        let b = slot(row + dy, 2 * stage.border + 1, stride) + (lo as isize + dx) as usize;
        let n = hi - lo;
        let pixels = &self.pixels;
        let dst = if stage.patch {
            &mut self.difference[dir]
        } else {
            &mut self.sad[dir]
        };
        let start = slot(row, if stage.patch { 3 } else { dy as usize + 1 }, stride) + lo;
        (k.difference)(
            pixels.each_ref().map(|p| &p[a..a + n]),
            pixels.each_ref().map(|p| &p[b..b + n]),
            lf.channel_scale,
            &mut dst[start..start + n],
        );
    }
    fn patch(&mut self, stage: &Stage, dir: usize, row: isize, w: usize, k: Kernels) {
        let (dx, dy) = stage.dirs[dir];
        let stride = w + 2 * stage.border;
        let lo = (-dx).max(0) as usize + 1;
        let hi = stride - dx.max(0) as usize - 1;
        let src = &self.difference[dir];
        let m = slot(row, 3, stride);
        let u = slot(row - 1, 3, stride);
        let d = slot(row + 1, 3, stride);
        let start = slot(row, dy as usize + 1, stride);
        (k.patch)(
            [
                &src[m + lo..m + hi],
                &src[m + lo - 1..m + hi - 1],
                &src[m + lo + 1..m + hi + 1],
                &src[u + lo..u + hi],
                &src[d + lo..d + hi],
            ],
            &mut self.sad[dir][start + lo..start + hi],
        );
    }
    fn render(
        &mut self,
        source: Option<&Image3F>,
        band: &mut [&mut [f32]; 3],
        first: usize,
        w: usize,
        stage: &Stage,
        lf: &LoopFilter,
        sigma: &[f32],
        blocks_w: usize,
        sm: f32,
        k: Kernels,
    ) {
        self.prepare(stage, w);
        let b = stage.border;
        let rows = band[0].len() / w;
        let stride = w + 2 * b;
        for y in -(b as isize)..=b as isize {
            self.load_row(y, source, band, first, rows, w, b);
        }
        for (dir, &(_, dy)) in stage.dirs.iter().enumerate() {
            if stage.patch {
                self.difference(stage, dir, -dy - 1, w, lf, k);
                self.difference(stage, dir, -dy, w, lf, k);
                for y in -dy..=0 {
                    self.difference(stage, dir, y + 1, w, lf, k);
                    self.patch(stage, dir, y, w, k);
                }
            } else {
                for y in -dy..=0 {
                    self.difference(stage, dir, y, w, lf, k);
                }
            }
        }
        for r in 0..rows {
            let y = r as isize;
            if r > 0 {
                self.load_row(y + b as isize, source, band, first, rows, w, b);
                for dir in 0..stage.dirs.len() {
                    self.difference(stage, dir, if stage.patch { y + 1 } else { y }, w, lf, k);
                    if stage.patch {
                        self.patch(stage, dir, y, w, k);
                    }
                }
            }
            let center = slot(y, 2 * b + 1, stride) + b;
            let neighbors: [Neighbor<'_>; 12] = std::array::from_fn(|i| {
                if i >= stage.neighbors.len() {
                    return Neighbor {
                        pixels: [&[]; 3],
                        sad: &[],
                    };
                }
                let (dir, along) = stage.neighbors[i];
                let (dx, dy) = stage.dirs[dir];
                let (ny, nx, sy, sx) = if along {
                    (y + dy, b as isize + dx, y, b)
                } else {
                    (y - dy, b as isize - dx, y - dy, (b as isize - dx) as usize)
                };
                let at = slot(ny, 2 * b + 1, stride) + nx as usize;
                let sad = slot(sy, dy as usize + 1, stride) + sx;
                Neighbor {
                    pixels: self.pixels.each_ref().map(|p| &p[at..at + w]),
                    sad: &self.sad[dir][sad..sad + w],
                }
            });
            let global_y = first + r;
            let row = FilterRow {
                center: self.pixels.each_ref().map(|p| &p[center..center + w]),
                neighbors: &neighbors[..stage.neighbors.len()],
                sigma: &sigma[(global_y / 8) * blocks_w..(global_y / 8 + 1) * blocks_w],
                border: global_y.is_multiple_of(8) || global_y % 8 == 7,
                sm,
                bsm: sm * lf.border_sad_mul,
            };
            (k.filter)(row, band.each_mut().map(|p| &mut p[r * w..(r + 1) * w]));
        }
    }
}

pub(super) fn render(
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    source: Option<&Image3F>,
    image: &mut Image3F,
    stage: &Stage,
    lf: &LoopFilter,
    sigma: &[f32],
    blocks_w: usize,
    sm: f32,
    k: Kernels,
    work: &mut [Scratch],
) {
    let (w, h) = (image.xsize(), image.ysize());
    let band_rows = h.div_ceil(work.len());
    if source.is_none() {
        // Snapshot all boundaries before any worker can overwrite its band.
        for (i, work) in work.iter_mut().enumerate() {
            let first = i * band_rows;
            if first < h {
                work.snapshot(image, first, band_rows.min(h - first), stage.border);
            }
        }
    }
    let [a, b, c] = image.all_planes_mut().map(Plane::as_mut_slice);
    let mut bands: Vec<_> = a
        .chunks_mut(band_rows * w)
        .zip(b.chunks_mut(band_rows * w))
        .zip(c.chunks_mut(band_rows * w))
        .zip(work)
        .enumerate()
        .map(|(i, (((a, b), c), work))| (i * band_rows, [a, b, c], work))
        .collect();
    pool.steal_for_each_mut(scratch, &mut bands, |_, (first, band, work), _| {
        work.render(source, band, *first, w, stage, lf, sigma, blocks_w, sm, k);
    });
}

#[cfg(test)]
impl Scratch {
    pub(super) fn floats(&self) -> usize {
        self.pixels
            .iter()
            .chain(&self.difference)
            .chain(&self.sad)
            .chain(&self.halo)
            .map(Vec::capacity)
            .sum()
    }
}
