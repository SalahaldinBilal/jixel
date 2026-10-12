//! Separable five-tap Gaussian reduction and sparse expansion. Reduction
//! reflects including edge samples; expansion replicates coarse edge samples,
//! matching the original's zero interleaving and explicit padding.

pub(crate) fn reduce(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, tmp: &mut Vec<f32>) {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    ensure_len(tmp, w * nh);
    ensure_len(dst, nw * nh);
    for y in 0..nh {
        let rows =
            std::array::from_fn::<_, 5, _>(|k| reflect((2 * y) as isize + k as isize - 2, h) * w);
        for x in 0..w {
            tmp[y * w + x] = 0.05 * (src[rows[0] + x] + src[rows[4] + x])
                + 0.25 * (src[rows[1] + x] + src[rows[3] + x])
                + 0.4 * src[rows[2] + x];
        }
    }
    for y in 0..nh {
        let row = &tmp[y * w..(y + 1) * w];
        for x in 0..nw {
            let xx = (2 * x) as isize;
            dst[y * nw + x] = 0.05 * (row[reflect(xx - 2, w)] + row[reflect(xx + 2, w)])
                + 0.25 * (row[reflect(xx - 1, w)] + row[reflect(xx + 1, w)])
                + 0.4 * row[2 * x];
        }
    }
}

pub(crate) fn expand(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, tmp: &mut Vec<f32>) {
    let (sw, sh) = (w.div_ceil(2), h.div_ceil(2));
    ensure_len(tmp, sw * h);
    ensure_len(dst, w * h);
    for y in 0..h {
        let sy = y / 2;
        let before = sy.saturating_sub(1) * sw;
        let center = sy * sw;
        let after = (sy + 1).min(sh - 1) * sw;
        for x in 0..sw {
            tmp[y * sw + x] = if y % 2 == 0 {
                0.1 * (src[before + x] + src[after + x]) + 0.8 * src[center + x]
            } else {
                0.5 * (src[center + x] + src[after + x])
            };
        }
    }
    for y in 0..h {
        let row = &tmp[y * sw..(y + 1) * sw];
        for x in 0..w {
            let sx = x / 2;
            dst[y * w + x] = if x % 2 == 0 {
                0.1 * (row[sx.saturating_sub(1)] + row[(sx + 1).min(sw - 1)]) + 0.8 * row[sx]
            } else {
                0.5 * (row[sx] + row[(sx + 1).min(sw - 1)])
            };
        }
    }
}

// Torchvision GaussianBlur(13, 3) uses reflect padding excluding the edge.
// The model deliberately skips this blur when either dimension is <= 6.
pub(crate) fn mask_blur(
    mask: &mut [f32],
    w: usize,
    h: usize,
    tmp: &mut Vec<f32>,
    kernel: &[f32; 13],
    padded: &mut Vec<f32>,
    backend: crate::Backend,
) {
    if w <= 6 || h <= 6 {
        return;
    }
    ensure_len(tmp, w * h);
    ensure_len(padded, w + 12);
    for y in 0..h {
        padded[6..w + 6].copy_from_slice(&mask[y * w..(y + 1) * w]);
        for k in 0..6 {
            padded[k] = mask[y * w + reflect_exclusive(k as isize - 6, w)];
            padded[w + 6 + k] = mask[y * w + reflect_exclusive((w + k) as isize, w)];
        }
        let rows = std::array::from_fn(|k| &padded[k..k + w]);
        crate::kernels::blur(backend, rows, &mut tmp[y * w..(y + 1) * w], kernel);
    }
    for y in 0..h {
        let rows = std::array::from_fn(|k| {
            let offset = reflect_exclusive(y as isize + k as isize - 6, h) * w;
            &tmp[offset..offset + w]
        });
        crate::kernels::blur(backend, rows, &mut mask[y * w..(y + 1) * w], kernel);
    }
}

fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    if i < 0 {
        (-i - 1) as usize
    } else if i >= n as isize {
        (2 * n) - i as usize - 1
    } else {
        i as usize
    }
}

fn reflect_exclusive(i: isize, n: usize) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= n as isize {
        (2 * n) - i as usize - 2
    } else {
        i as usize
    }
}

pub(crate) fn ensure_len(buffer: &mut Vec<f32>, length: usize) {
    if buffer.len() < length {
        buffer.resize(length, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_odd_even_and_degenerate_pyramids() {
        for (w, h) in [(1, 1), (1, 9), (10, 1), (3, 4), (4, 3), (63, 64), (64, 63)] {
            let src = vec![1.0; w * h];
            let (mut reduced, mut expanded, mut tmp) = (Vec::new(), Vec::new(), Vec::new());
            reduce(&src, w, h, &mut reduced, &mut tmp);
            expand(&reduced, w, h, &mut expanded, &mut tmp);
            assert!(
                reduced
                    .iter()
                    .chain(&expanded)
                    .all(|v| (*v - 1.0).abs() < 1e-6)
            );
        }
    }

    #[test]
    fn symmetric_reduce_duplicates_right_edge() {
        let mut reduced = Vec::new();
        reduce(
            &[0.0, 0.0, 0.0, 1.0, 1.0],
            5,
            1,
            &mut reduced,
            &mut Vec::new(),
        );
        assert!((reduced[2] - 0.95).abs() < 1e-6);
    }
}
