//! SSIMULACRA 2 on XYB images, for the closed-loop quant field.
//!
//! Follows the reference (libjxl `tools/ssimulacra2.cc`): six scales, X/Y/B
//! planes with the "positive XYB" offsets, SSIM map without the luminance
//! denominator, artifact / detail-loss edge maps, 1- and 4-norm pooling, the
//! 108-weight sum and the final nonlinearity. Scales are box-downsampled in
//! linear RGB and re-converted to XYB. The reference's recursive Gaussian
//! kernel is applied with the encoder's separable 11-tap SIMD FIR.
//!
//! Besides the score it returns a per-pixel map of the linearized loss: the
//! pooled sums are additive over pixels before the final nonlinearity, so a
//! region's map sum is that region's share of `100 - score`.
use crate::{aq_blur::GaussianBlur, aq_loop::downsample_rows, image::Image3F};
use std::sync::OnceLock;

const C2: f32 = 0.0009;
const NUM_SCALES: usize = 6;

/// Reference weights, indexed `((plane * 6 + scale) * 2 + norm) * 3 + map`
/// with map 0 = SSIM, 1 = artifact, 2 = detail loss.
#[rustfmt::skip]
static WEIGHTS: [f64; 108] = [
    0.0, 0.0007376606707406586, 0.0, 0.0, 0.0007793481682867309, 0.0, 0.0, 0.0004371155730107379, 0.0,
    1.1041726426657346, 0.00066284834129271, 0.00015231632783718752, 0.0, 0.0016406437456599754, 0.0,
    1.8422455520539298, 11.441172603757666, 0.0, 0.0007989109436015163, 0.000176816438078653, 0.0,
    1.8787594979546387, 10.94906990605142, 0.0, 0.0007289346991508072, 0.9677937080626833, 0.0,
    0.00014003424285435884, 0.9981766977854967, 0.00031949755934435053, 0.0004550992113792063, 0.0, 0.0,
    0.0013648766163243398, 0.0, 0.0, 0.0, 0.0, 0.0, 7.466890328078848, 0.0, 17.445833984131262,
    0.0006235601634041466, 0.0, 0.0, 6.683678146179332, 0.00037724407979611296, 1.027889937768264,
    225.20515300849274, 0.0, 0.0, 19.213238186143016, 0.0011401524586618361, 0.001237755635509985,
    176.39317598450694, 0.0, 0.0, 24.43300999870476, 0.28520802612117757, 0.0004485436923833408, 0.0, 0.0,
    0.0, 34.77906344483772, 44.835625328877896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0008680556573291698, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005313191874358747, 0.0, 0.00016533814161379112,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0004179171803251336, 0.0017290828234722833, 0.0, 0.0020827005846636437,
    0.0, 0.0, 8.826982764996862, 23.19243343998926, 0.0, 95.1080498811086, 0.9863978034400682,
    0.9834382792465353, 0.0012286405048278493, 171.2667255897307, 0.9807858872435379, 0.0, 0.0, 0.0,
    0.0005130064588990679, 0.0, 0.00010854057858411537,
];

/// A map is needed if either of its pooling norms has a nonzero weight.
fn active_maps(plane: usize, scale: usize) -> [bool; 3] {
    let base = (plane * NUM_SCALES + scale) * 6;
    std::array::from_fn(|m| WEIGHTS[base + m] != 0.0 || WEIGHTS[base + 3 + m] != 0.0)
}

/// Score from the weighted pooled sum.
fn score_from_sum(s: f64) -> f64 {
    let s = s * 0.9562382616834844;
    let s =
        2.326765642916932 * s - 0.020884521182843837 * s * s + 6.248496625763138e-05 * s * s * s;
    if s > 0.0 {
        100.0 - 10.0 * s.powf(0.6276336467831387)
    } else {
        100.0
    }
}

/// d(100 - score)/d(sum) at `s`: how many score points one unit of the
/// weighted sum costs.
fn deficit_slope(s: f64) -> f64 {
    let u = s * 0.9562382616834844;
    let g =
        2.326765642916932 * u - 0.020884521182843837 * u * u + 6.248496625763138e-05 * u * u * u;
    if g <= 0.0 {
        return 0.0;
    }
    let dg = (2.326765642916932 - 2.0 * 0.020884521182843837 * u
        + 3.0 * 6.248496625763138e-05 * u * u)
        * 0.9562382616834844;
    10.0 * 0.6276336467831387 * g.powf(0.6276336467831387 - 1.0) * dg
}

/// 2×2 box downsample with edge replication (ceil dimensions).
fn downsample(a: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0f32; nw * nh];
    for y in 0..nh {
        let (y0, y1) = ((2 * y).min(h - 1), (2 * y + 1).min(h - 1));
        downsample_rows(
            &a[y0 * w..(y0 + 1) * w],
            &a[y1 * w..(y1 + 1) * w],
            &mut out[y * nw..(y + 1) * nw],
        );
    }
    (out, nw, nh)
}

/// Reference blur: libjxl's recursive Gaussian (Charalampidis 2016) is a
/// truncated sum of three cosines with radius `round(3.2795σ + 0.2546)` = 5 at
/// σ = 1.5 and zero padding outside the image; this is the same kernel applied
/// as a separable FIR.
const RADIUS: usize = crate::aq_blur::RADIUS;
const TAPS: usize = crate::aq_blur::WINDOW;

fn kernel() -> &'static [f32; TAPS] {
    static K: OnceLock<[f32; TAPS]> = OnceLock::new();
    K.get_or_init(|| {
        let sigma = 1.5f64;
        let radius = RADIUS as f64;
        let pi_div_2r = std::f64::consts::PI / (2.0 * radius);
        let omega = [pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r];
        let p = [
            1.0 / (0.5 * omega[0]).tan(),
            -1.0 / (0.5 * omega[1]).tan(),
            1.0 / (0.5 * omega[2]).tan(),
        ];
        let r = [
            p[0] * p[0] / omega[0].sin(),
            -p[1] * p[1] / omega[1].sin(),
            p[2] * p[2] / omega[2].sin(),
        ];
        let rho: Vec<f64> = omega
            .iter()
            .map(|w| (-0.5 * sigma * sigma * w * w).exp() / radius)
            .collect();
        let d13 = p[0] * r[1] - r[0] * p[1];
        let d35 = p[1] * r[2] - r[1] * p[2];
        let d51 = p[2] * r[0] - r[2] * p[0];
        let zeta15 = d35 / d13;
        let zeta35 = d51 / d13;
        // solve A·beta = gamma with A = [[p], [r], [zeta15, zeta35, 1]]
        let a = [
            [p[0], p[1], p[2]],
            [r[0], r[1], r[2]],
            [zeta15, zeta35, 1.0],
        ];
        let g = [
            1.0,
            radius * radius - sigma * sigma,
            zeta15 * rho[0] + zeta35 * rho[1] + rho[2],
        ];
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        let solve = |col: usize| {
            let mut m = a;
            for row in 0..3 {
                m[row][col] = g[row];
            }
            (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                / det
        };
        let beta = [solve(0), solve(1), solve(2)];
        let mut k = [0f32; TAPS];
        for (i, slot) in k.iter_mut().enumerate() {
            let x = i as f64 - radius;
            *slot = (0..3).map(|j| beta[j] * (omega[j] * x).cos()).sum::<f64>() as f32;
        }
        k
    })
}

#[derive(Default)]
struct PlaneMaps {
    ssim: Vec<f32>,
    artifact: Vec<f32>,
    detail: Vec<f32>,
}

/// Moment and map buffers are overwritten for each plane and retained across
/// scales, together with the blur's row ring and padded product input row.
struct PlaneMapScratch {
    blur: GaussianBlur,
    moments: [Vec<f32>; 5],
    maps: PlaneMaps,
}

impl PlaneMapScratch {
    fn new(w: usize) -> Self {
        Self {
            blur: GaussianBlur::with_kernel(w, *kernel(), true),
            moments: std::array::from_fn(|_| Vec::new()),
            maps: PlaneMaps::default(),
        }
    }

    fn resize(&mut self, w: usize, h: usize) {
        self.blur.set_width(w);
        for buffer in self.moments.iter_mut().chain([
            &mut self.maps.ssim,
            &mut self.maps.artifact,
            &mut self.maps.detail,
        ]) {
            buffer.resize(w * h, 0.0);
        }
    }

    fn plane_maps(&mut self, src: &[f32], rec: &[f32], h: usize, need_ssim: bool) -> &PlaneMaps {
        let [mu1, mu2, s11, s22, s12] = &mut self.moments;
        self.blur.blur(src, h, mu1);
        self.blur.blur(rec, h, mu2);
        // The finest X/B scales only weight edge artifacts. Their two means
        // suffice; the three variance/covariance blurs cannot affect the loss.
        if need_ssim {
            self.blur.blur_product([src, src], h, s11);
            self.blur.blur_product([rec, rec], h, s22);
            self.blur.blur_product([src, rec], h, s12);
            for (((((out, &m1), &m2), &v11), &v22), &v12) in self
                .maps
                .ssim
                .iter_mut()
                .zip(mu1.iter())
                .zip(mu2.iter())
                .zip(s11.iter())
                .zip(s22.iter())
                .zip(s12.iter())
            {
                let num_m = 1.0 - (m1 - m2) * (m1 - m2);
                let num_s = 2.0 * (v12 - m1 * m2) + C2;
                let denom_s = (v11 - m1 * m1) + (v22 - m2 * m2) + C2;
                *out = (1.0 - num_m * num_s / denom_s).max(0.0);
            }
        }
        for (((((artifact, detail), &a), &b), &m1), &m2) in self
            .maps
            .artifact
            .iter_mut()
            .zip(&mut self.maps.detail)
            .zip(src)
            .zip(rec)
            .zip(mu1.iter())
            .zip(mu2.iter())
        {
            let d1 = (1.0 + (b - m2).abs()) / (1.0 + (a - m1).abs()) - 1.0;
            *artifact = d1.max(0.0);
            *detail = (-d1).max(0.0);
        }
        &self.maps
    }
}

/// Pooled 1-norm, 4-norm and the per-pixel linearized contribution of one map
/// to the weighted sum (`w1·x/N + w4·x⁴/(4·N·M4^(3/4))`), accumulated into `acc`.
fn pool(map: &[f32], w1: f64, w4: f64, acc: &mut [f32]) -> f64 {
    if w1 == 0.0 && w4 == 0.0 {
        return 0.0;
    }
    let n = map.len() as f64;
    let (s1, s4) = map_sums(map);
    let m1 = s1 / n;
    let m4 = s4 / n;
    let n4 = m4.sqrt().sqrt();
    add_map_contribution(map, w1, w4, n4, acc);
    w1 * m1 + w4 * n4
}

fn map_sums(map: &[f32]) -> (f64, f64) {
    let mut s1 = [0.0f64; SUM_LANES];
    let mut s4 = [0.0f64; SUM_LANES];
    let (chunks, tail) = map.as_chunks::<SUM_LANES>();
    for chunk in chunks {
        for lane in 0..SUM_LANES {
            let v = chunk[lane] as f64;
            s1[lane] += v;
            let v_sqr = v * v;
            s4[lane] += v_sqr * v_sqr;
        }
    }
    for (lane, &v) in tail.iter().enumerate() {
        let v = v as f64;
        s1[lane] += v;
        let v_sqr = v * v;
        s4[lane] += v_sqr * v_sqr;
    }
    (s1.iter().sum(), s4.iter().sum())
}

/// Compute the pixel contribution in f32. Normalizing by the fourth norm
/// keeps the fourth powers bounded by the pixel count, even for HDR maps.
/// The pooled totals and final score stay f64, including incremental deltas.
fn add_map_contribution(map: &[f32], w1: f64, w4: f64, n4: f64, acc: &mut [f32]) {
    let n = map.len() as f64;
    let g1 = w1 / n;
    if w4 == 0.0 || n4 == 0.0 {
        contribute(
            map,
            MapContribution {
                linear: g1 as f32,
                inv_norm: 0.0,
                norm_weight: 0.0,
            },
            acc,
        );
        return;
    }
    let inv_n4 = (1.0 / n4) as f32;
    let norm_weight = (w4 * n4 / (4.0 * n)) as f32;
    if inv_n4.is_normal() && norm_weight.is_normal() {
        contribute(
            map,
            MapContribution {
                linear: g1 as f32,
                inv_norm: inv_n4,
                norm_weight,
            },
            acc,
        );
        return;
    }
    // Tiny or large norms may have unrepresentable f32 factors.
    let g4 = w4 / (4.0 * n * n4 * n4 * n4);
    for (a, &v) in acc.iter_mut().zip(map) {
        let v = v as f64;
        let v_sqr = v * v;
        *a += (g1 * v + g4 * v_sqr * v_sqr) as f32;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MapContribution {
    pub(crate) linear: f32,
    pub(crate) inv_norm: f32,
    pub(crate) norm_weight: f32,
}

type ContributionFn = unsafe fn(&[f32], MapContribution, &mut [f32]) -> usize;
type ExpandRowFn = unsafe fn(&[f32], &mut [f32], f32) -> usize;

struct MapKernels {
    contribution: ContributionFn,
    expand: [ExpandRowFn; NUM_SCALES],
}

fn map_kernels() -> &'static MapKernels {
    static KERNELS: OnceLock<MapKernels> = OnceLock::new();
    KERNELS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return MapKernels {
                contribution: crate::avx::ss2_contribution_avx2,
                expand: [
                    crate::avx::ss2_expand_row_avx2::<1>,
                    crate::avx::ss2_expand_row_avx2::<2>,
                    crate::avx::ss2_expand_row_avx2::<4>,
                    crate::avx::ss2_expand_row_avx2::<8>,
                    crate::avx::ss2_expand_row_avx2::<16>,
                    crate::avx::ss2_expand_row_avx2::<32>,
                ],
            };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        return MapKernels {
            contribution: crate::neon::ss2_contribution_neon,
            expand: [
                crate::neon::ss2_expand_row_neon::<1>,
                crate::neon::ss2_expand_row_neon::<2>,
                crate::neon::ss2_expand_row_neon::<4>,
                crate::neon::ss2_expand_row_neon::<8>,
                crate::neon::ss2_expand_row_neon::<16>,
                crate::neon::ss2_expand_row_neon::<32>,
            ],
        };
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        MapKernels {
            contribution: contribution_scalar,
            expand: [
                expand_row_scalar::<1>,
                expand_row_scalar::<2>,
                expand_row_scalar::<4>,
                expand_row_scalar::<8>,
                expand_row_scalar::<16>,
                expand_row_scalar::<32>,
            ],
        }
    })
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn contribution_scalar(map: &[f32], p: MapContribution, acc: &mut [f32]) -> usize {
    if p.norm_weight == 0.0 {
        for (a, &v) in acc.iter_mut().zip(map) {
            *a += p.linear * v;
        }
    } else {
        for (a, &v) in acc.iter_mut().zip(map) {
            let normalized = v * p.inv_norm;
            let square = normalized * normalized;
            *a += p.linear * v + p.norm_weight * square * square;
        }
    }
    acc.len()
}

fn contribute(map: &[f32], p: MapContribution, acc: &mut [f32]) {
    assert_eq!(map.len(), acc.len());
    // SAFETY: dispatch checked the CPU features and both slices have equal
    // lengths. The kernel reads/writes complete vectors and returns its prefix.
    let done = unsafe { (map_kernels().contribution)(map, p, acc) };
    // SIMD kernels use FMA; keep their short tails fused as well. The portable
    // kernel processes the whole slice without requiring hardware FMA.
    for (a, &v) in acc[done..].iter_mut().zip(&map[done..]) {
        if p.norm_weight == 0.0 {
            *a = p.linear.mul_add(v, *a);
        } else {
            let normalized = v * p.inv_norm;
            let square = normalized * normalized;
            *a += (p.norm_weight * square).mul_add(square, p.linear * v);
        }
    }
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn expand_row_scalar<const STEP: usize>(src: &[f32], dst: &mut [f32], share: f32) -> usize {
    for (pixels, &v) in dst.chunks_mut(STEP).zip(src) {
        let contribution = v * share;
        for out in pixels {
            *out += contribution;
        }
    }
    dst.len()
}

/// Spread a scale's contribution in contiguous runs. Each source pixel covers
/// STEP×STEP pixels, with the last run cropped at odd image edges.
fn accumulate_scale_map(acc: &[f32], w0: usize, h0: usize, scale: usize, map: &mut [f32]) {
    let step = 1usize << scale;
    let w = w0.div_ceil(step);
    let h = h0.div_ceil(step);
    assert_eq!(acc.len(), w * h);
    assert_eq!(map.len(), w0 * h0);
    if map.is_empty() {
        return;
    }
    let share = 1.0 / (step * step) as f32;
    let expand = map_kernels().expand[scale];
    for (src, rows) in acc.chunks_exact(w).zip(map.chunks_mut(w0 * step)) {
        for row in rows.chunks_exact_mut(w0) {
            // SAFETY: dispatch checked CPU features. `src` has ceil(row/STEP)
            // pixels and the specialized kernel handles only complete runs.
            let done = unsafe { expand(src, row, share) };
            for (pixels, &v) in row[done..].chunks_mut(step).zip(&src[done / step..]) {
                for out in pixels {
                    *out = v.mul_add(share, *out);
                }
            }
        }
    }
}

/// Independent lanes break the pooled sums' per-pixel dependency chain and
/// let LLVM vectorize the map calculation together with the f64 reductions.
const SUM_LANES: usize = 8;

fn replace_map<const INPUTS: usize>(
    map: &mut [f32],
    inputs: [&[f32]; INPUTS],
    sums: &mut (f64, f64),
    need_fourth: bool,
    value: impl Fn([f32; INPUTS]) -> f32,
) {
    // Select once per row, keeping the map math and independent f64 lanes
    // vectorizable without a norm-weight branch inside the pixel loop.
    if need_fourth {
        replace_map_with_norm::<INPUTS, true>(map, inputs, sums, value);
    } else {
        replace_map_with_norm::<INPUTS, false>(map, inputs, sums, value);
    }
}

fn replace_map_with_norm<const INPUTS: usize, const FOURTH: bool>(
    map: &mut [f32],
    inputs: [&[f32]; INPUTS],
    sums: &mut (f64, f64),
    value: impl Fn([f32; INPUTS]) -> f32,
) {
    let n = map.len();
    assert!(inputs.iter().all(|input| input.len() == n));
    let mut delta1 = [0.0f64; SUM_LANES];
    let mut delta4 = [0.0f64; SUM_LANES];
    let (chunks, tail) = map.as_chunks_mut::<SUM_LANES>();
    let input_chunks = inputs.map(|input| input.as_chunks::<SUM_LANES>().0);
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let input = input_chunks.map(|chunks| &chunks[i]);
        for lane in 0..SUM_LANES {
            let new = value(input.map(|input| input[lane]));
            let old = chunk[lane] as f64;
            let v = new as f64;
            delta1[lane] += v - old;
            if FOURTH {
                let v_sqr = v * v;
                let old_sqr = old * old;
                delta4[lane] += v_sqr * v_sqr - old_sqr * old_sqr;
            }
            chunk[lane] = new;
        }
    }
    let offset = n - tail.len();
    for (i, out) in tail.iter_mut().enumerate() {
        let new = value(inputs.map(|input| input[offset + i]));
        let old = *out as f64;
        let v = new as f64;
        delta1[i] += v - old;
        if FOURTH {
            let v_sqr = v * v;
            let old_sqr = old * old;
            delta4[i] += v_sqr * v_sqr - old_sqr * old_sqr;
        }
        *out = new;
    }
    sums.0 += delta1.iter().sum::<f64>();
    if FOURTH {
        sums.1 += delta4.iter().sum::<f64>();
    }
}

/// Positive XYB planes of linear-RGB planes (the per-scale conversion of the reference).
fn positive_xyb_of_linear(lin: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
    let n = lin[0].len();
    let mut out = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    if n > 0 {
        let [x, y, b] = &mut out;
        positive_xyb_into([&lin[0], &lin[1], &lin[2]], [x, y, b]);
    }
    out
}

fn positive_xyb_into(lin: [&[f32]; 3], [x, y, b]: [&mut [f32]; 3]) {
    let m = crate::xyb::XybMatrix::SPEC;
    let n = x.len();
    assert!(n > 0);
    assert!(lin.iter().all(|p| p.len() == n));
    assert_eq!(y.len(), n);
    assert_eq!(b.len(), n);
    let f = crate::xyb::selected_to_xyb_band_fn();
    // SAFETY: the dispatcher picked a kernel for a detected feature set; all
    // input and output slices have `n` elements and the band width is nonzero.
    unsafe { f(&m, lin, [x, y, b], n) };
    for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
        *x = *x * 14.0 + 0.42;
        *b = (*b - *y) + 0.55;
        *y += 0.01;
    }
}

/// Linear RGB planes of an XYB image (the encoder's inverse opsin transform).
pub(crate) fn linear_of_xyb(img: &Image3F) -> [Vec<f32>; 3] {
    let (w, h) = (img.xsize(), img.ysize());
    let m = crate::xyb::XybMatrix::SPEC;
    let mut out = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
    let [r, g, b] = &mut out;
    for y in 0..h {
        let span = y * w..(y + 1) * w;
        crate::xyb::xyb_to_rgb_rows(
            &m,
            [
                img.plane_row(0, y),
                img.plane_row(1, y),
                img.plane_row(2, y),
            ],
            [&mut r[span.clone()], &mut g[span.clone()], &mut b[span]],
        );
    }
    out
}

/// Scores two images given as linear-RGB planes: scales are box-downsampled in
/// linear RGB and converted to positive XYB, as in the reference.
fn ssimulacra2_of_linear(
    mut la: [Vec<f32>; 3],
    mut lb: [Vec<f32>; 3],
    w0: usize,
    h0: usize,
) -> (f32, Vec<f32>) {
    let mut src = positive_xyb_of_linear(&la);
    let mut rec = positive_xyb_of_linear(&lb);
    let mut next =
        |src: &mut [Vec<f32>; 3], rec: &mut [Vec<f32>; 3], w: usize, h: usize| -> (usize, usize) {
            let (mut nw, mut nh) = (w, h);
            for c in 0..3 {
                let (na, a, b) = downsample(&la[c], w, h);
                let (nb, _, _) = downsample(&lb[c], w, h);
                la[c] = na;
                lb[c] = nb;
                nw = a;
                nh = b;
            }
            *src = positive_xyb_of_linear(&la);
            *rec = positive_xyb_of_linear(&lb);
            (nw, nh)
        };
    ssimulacra2_scales(w0, h0, &mut src, &mut rec, &mut next)
}

/// `100 - SSIMULACRA2(source, recon)` of two XYB images and the per-pixel
/// linearized map (a region's map mean is its share of the deficit).
pub(crate) fn ssimulacra2_deficit_and_map(source: &Image3F, recon: &Image3F) -> (f32, Vec<f32>) {
    ssimulacra2_of_linear(
        linear_of_xyb(source),
        linear_of_xyb(recon),
        source.xsize(),
        source.ysize(),
    )
}

/// Produces the next scale's planes in place and returns the new size.
type NextScale<'a> =
    dyn FnMut(&mut [Vec<f32>; 3], &mut [Vec<f32>; 3], usize, usize) -> (usize, usize) + 'a;

/// Shared scale loop: `next` replaces the planes with the next scale and returns its size.
fn ssimulacra2_scales(
    w0: usize,
    h0: usize,
    src: &mut [Vec<f32>; 3],
    rec: &mut [Vec<f32>; 3],
    next: &mut NextScale<'_>,
) -> (f32, Vec<f32>) {
    let (mut w, mut h) = (w0, h0);
    let mut sum = 0f64;
    let mut map = vec![0f32; w0 * h0];
    let mut scratch = PlaneMapScratch::new(w0);
    let mut acc = Vec::new();
    for scale in 0..NUM_SCALES {
        if w < 8 || h < 8 {
            break;
        }
        scratch.resize(w, h);
        acc.resize(w * h, 0.0);
        acc.fill(0.0);
        for c in 0..3 {
            let base = (c * 6 + scale) * 6;
            if WEIGHTS[base..base + 6].iter().all(|&w| w == 0.0) {
                continue;
            }
            let need_ssim = WEIGHTS[base] != 0.0 || WEIGHTS[base + 3] != 0.0;
            let m = scratch.plane_maps(&src[c], &rec[c], h, need_ssim);
            // norm 0 (1-norm) and norm 1 (4-norm) weights per map
            let wt = |norm: usize, map: usize| WEIGHTS[base + norm * 3 + map];
            sum += pool(&m.ssim, wt(0, 0), wt(1, 0), &mut acc);
            sum += pool(&m.artifact, wt(0, 1), wt(1, 1), &mut acc);
            sum += pool(&m.detail, wt(0, 2), wt(1, 2), &mut acc);
        }
        accumulate_scale_map(&acc, w0, h0, scale, &mut map);
        if scale + 1 < NUM_SCALES {
            let (nw, nh) = next(src, rec, w, h);
            w = nw;
            h = nh;
        }
    }
    let deficit = 100.0 - score_from_sum(sum);
    // per-pixel values whose MEAN over the image is the (linearized) deficit,
    // like the MS-SSIM map the loop consumes
    let scale = deficit_slope(sum) as f32 * (w0 * h0) as f32;
    for v in &mut map {
        *v *= scale;
    }
    (deficit as f32, map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_kernels() -> [(&'static str, MapKernels, bool); 2] {
        let selected = map_kernels();
        let fused = cfg!(all(target_arch = "aarch64", feature = "neon")) || {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
            {
                false
            }
        };
        [
            (
                "portable",
                MapKernels {
                    contribution: contribution_scalar,
                    expand: [
                        expand_row_scalar::<1>,
                        expand_row_scalar::<2>,
                        expand_row_scalar::<4>,
                        expand_row_scalar::<8>,
                        expand_row_scalar::<16>,
                        expand_row_scalar::<32>,
                    ],
                },
                false,
            ),
            (
                "dispatch",
                MapKernels {
                    contribution: selected.contribution,
                    expand: selected.expand,
                },
                fused,
            ),
        ]
    }

    #[test]
    fn contribution_kernels_cover_unaligned_vectors_and_tails() {
        const GUARD: f32 = -12345.0;
        for (name, kernels, fused) in test_kernels() {
            for n in (0..=65).chain([127, 128, 129, 255, 257]) {
                for offset in 0..4 {
                    let mut input = vec![GUARD; n + 8];
                    let map = &mut input[offset..offset + n];
                    for (i, v) in map.iter_mut().enumerate() {
                        *v = (i % 31 + 1) as f32 * 0.173829;
                    }
                    let input_before = input.clone();
                    for norm_weight in [0.0, 0.0009876543] {
                        let p = MapContribution {
                            linear: 0.0001234567,
                            inv_norm: 0.7123456,
                            norm_weight,
                        };
                        let mut output = vec![GUARD; n + 8];
                        let initial: Vec<_> =
                            (0..n).map(|i| (i % 17) as f32 * 0.00001876543).collect();
                        output[offset..offset + n].copy_from_slice(&initial);
                        // SAFETY: the selected kernel's CPU requirements were
                        // detected; the offset slices have matching lengths.
                        let done = unsafe {
                            (kernels.contribution)(
                                &input[offset..offset + n],
                                p,
                                &mut output[offset..offset + n],
                            )
                        };
                        assert!(done <= n);
                        for i in 0..n {
                            let v = input[offset + i];
                            let expected = if i >= done {
                                initial[i]
                            } else if norm_weight == 0.0 {
                                if fused {
                                    p.linear.mul_add(v, initial[i])
                                } else {
                                    initial[i] + p.linear * v
                                }
                            } else {
                                let normalized = v * p.inv_norm;
                                let square = normalized * normalized;
                                initial[i]
                                    + if fused {
                                        (norm_weight * square).mul_add(square, p.linear * v)
                                    } else {
                                        p.linear * v + norm_weight * square * square
                                    }
                            };
                            assert_eq!(
                                output[offset + i].to_bits(),
                                expected.to_bits(),
                                "{name}, n={n}, offset={offset}, pixel={i}"
                            );
                        }
                        assert!(output[..offset].iter().all(|&v| v == GUARD));
                        assert!(output[offset + n..].iter().all(|&v| v == GUARD));
                        assert_eq!(input, input_before);

                        if name == "dispatch" {
                            let mut actual = initial.clone();
                            contribute(&input[offset..offset + n], p, &mut actual);
                            for i in done..n {
                                let v = input[offset + i];
                                let normalized = v * p.inv_norm;
                                let square = normalized * normalized;
                                let expected = if norm_weight == 0.0 {
                                    p.linear.mul_add(v, initial[i])
                                } else {
                                    initial[i]
                                        + (norm_weight * square).mul_add(square, p.linear * v)
                                };
                                assert_eq!(actual[i].to_bits(), expected.to_bits());
                            }
                            assert_eq!(actual[..done], output[offset..offset + done]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scale_expansion_covers_all_scales_and_cropped_edges() {
        const GUARD: f32 = -12345.0;
        for (name, kernels, _) in test_kernels() {
            for scale in 0..NUM_SCALES {
                let step = 1 << scale;
                let share = 1.0 / (step * step) as f32;
                for width in (0usize..=65).chain([127, 129, 255, 257]) {
                    let n = width.div_ceil(step);
                    for offset in 0..4 {
                        let mut input = vec![GUARD; n + 8];
                        for (i, v) in input[offset..offset + n].iter_mut().enumerate() {
                            *v = ((i * 19) % 127) as f32 * 0.125;
                        }
                        let before = input.clone();
                        let mut output = vec![GUARD; width + 8];
                        let initial: Vec<_> = (0..width)
                            .map(|i| ((i * 11) % 23) as f32 * 0.0625)
                            .collect();
                        output[offset..offset + width].copy_from_slice(&initial);
                        // SAFETY: detected CPU features and ceil(width/STEP)
                        // source pixels cover every complete destination run.
                        let done = unsafe {
                            (kernels.expand[scale])(
                                &input[offset..offset + n],
                                &mut output[offset..offset + width],
                                share,
                            )
                        };
                        assert!(done <= width && (done == width || done.is_multiple_of(step)));
                        for i in 0..width {
                            let expected = initial[i]
                                + if i < done {
                                    input[offset + i / step] * share
                                } else {
                                    0.0
                                };
                            assert_eq!(
                                output[offset + i].to_bits(),
                                expected.to_bits(),
                                "{name}, scale={scale}, width={width}, offset={offset}, pixel={i}"
                            );
                        }
                        assert!(output[..offset].iter().all(|&v| v == GUARD));
                        assert!(output[offset + width..].iter().all(|&v| v == GUARD));
                        assert_eq!(input, before);
                    }
                    for height in [0usize, 1, 3, 17, 33] {
                        let h = height.div_ceil(step);
                        let acc: Vec<_> = (0..n * h)
                            .map(|i| ((i * 19) % 127) as f32 * 0.125)
                            .collect();
                        let mut actual: Vec<_> = (0..width * height)
                            .map(|i| ((i * 11) % 23) as f32 * 0.0625)
                            .collect();
                        let mut expected = actual.clone();
                        for y in 0..height {
                            for x in 0..width {
                                expected[y * width + x] += acc[(y / step) * n + x / step] * share;
                            }
                        }
                        accumulate_scale_map(&acc, width, height, scale, &mut actual);
                        assert_eq!(actual, expected, "scale={scale}, {width}x{height}");
                    }
                }
            }
        }
    }

    #[test]
    fn replacement_tails_update_fourth_powers_and_revert() {
        let totals = |values: &[f32]| {
            values.iter().fold((0.0, 0.0), |(s1, s4), &v| {
                let v = v as f64;
                (s1 + v, s4 + v.powi(4))
            })
        };
        for n in (0..=65).chain([127, 129, 257]) {
            let old: Vec<_> = (0..n).map(|i| (i % 23) as f32 / 16.0).collect();
            let new: Vec<_> = (0..n).map(|i| (i % 29) as f32 / 32.0).collect();
            for fourth in [false, true] {
                let expected = |values: &[f32]| {
                    let (s1, s4) = totals(values);
                    (s1, if fourth { s4 } else { 0.0 })
                };
                let mut map = old.clone();
                let mut sums = expected(&old);
                replace_map(&mut map, [&new], &mut sums, fourth, |[v]| v);
                assert_eq!(map, new);
                assert_eq!(sums, expected(&new), "n={n}, fourth={fourth}");
                replace_map(&mut map, [&old], &mut sums, fourth, |[v]| v);
                assert_eq!(map, old);
                assert_eq!(sums, expected(&old), "n={n}, fourth={fourth}");
            }
        }
    }

    #[test]
    fn first_norm_reduction_matches_full_reduction_across_tails_and_hdr_values() {
        for n in (0..=65).chain([127, 129, 4099, 307_200]) {
            for magnitude in [0.0, 1e-35, 1e-7, 1.0, 1e20, 1e37] {
                let map: Vec<_> = (0..n)
                    .map(|i| magnitude * ((i % 17 + 1) as f32 / 17.0))
                    .collect();
                let mut actual = vec![0.0; n];
                let mut first = (0.0, 0.0);
                replace_map(&mut actual, [&map], &mut first, false, |[v]| v);
                let both = map_sums(&map);
                assert_eq!(actual, map);
                assert_eq!(first.0.to_bits(), both.0.to_bits());
                assert_eq!(first.1, 0.0);
            }
        }
    }

    // Original f64 per-pixel pooling, kept as an independent precision oracle.
    fn reference_pool(map: &[f32], w1: f64, w4: f64, acc: &mut [f32]) -> f64 {
        if w1 == 0.0 && w4 == 0.0 {
            return 0.0;
        }
        let n = map.len() as f64;
        let (mut s1, mut s4) = (0.0f64, 0.0f64);
        for &v in map {
            let v = v as f64;
            s1 += v;
            s4 += v * v * v * v;
        }
        let n4 = (s4 / n).sqrt().sqrt();
        let g1 = w1 / n;
        let g4 = if n4 > 0.0 {
            w4 / (4.0 * n * n4 * n4 * n4)
        } else {
            0.0
        };
        for (a, &v) in acc.iter_mut().zip(map) {
            let v = v as f64;
            *a += (g1 * v + g4 * v * v * v * v) as f32;
        }
        w1 * s1 / n + w4 * n4
    }

    #[test]
    fn pooling_matches_f64_for_sparse_maps_tails_and_extreme_norms() {
        let mut max_relative_error = 0.0f32;
        for n in (1..=17).chain([31, 257, 4099, 307_200]) {
            for magnitude in [0.0, 1e-44, 1e-32, 1e-20, 1e-7, 1.0, 1e5, 1e20, 1e37] {
                for sparse in [false, true] {
                    let map: Vec<_> = (0..n)
                        .map(|i| {
                            if sparse && i != n / 2 {
                                0.0
                            } else {
                                magnitude * ((i % 17 + 1) as f32 / 17.0)
                            }
                        })
                        .collect();
                    let mut actual = vec![0.0; n];
                    let mut expected = vec![0.0; n];
                    let (weight_sets, _) = WEIGHTS.as_chunks::<6>();
                    for weights in weight_sets.iter().step_by(if n > 4099 { 6 } else { 1 }) {
                        for m in 0..3 {
                            actual.fill(0.0);
                            expected.fill(0.0);
                            let sum = pool(&map, weights[m], weights[m + 3], &mut actual);
                            let reference =
                                reference_pool(&map, weights[m], weights[m + 3], &mut expected);
                            assert!(
                                (sum - reference).abs() <= 2e-12 * reference.abs(),
                                "{n}, {magnitude}, {sparse}: {sum} vs {reference}"
                            );
                            for (&a, &b) in actual.iter().zip(&expected) {
                                if b.is_infinite() {
                                    assert_eq!(a, b);
                                    continue;
                                }
                                let error = (a - b).abs() / b.abs().max(f32::MIN_POSITIVE);
                                max_relative_error = max_relative_error.max(error);
                                assert!(
                                    error <= 2e-6,
                                    "{n}, {magnitude}, {sparse}: {a} vs {b}, error {error}"
                                );
                            }
                        }
                    }
                }
            }
        }
        eprintln!("f32 contribution max relative error: {max_relative_error}");
    }

    #[test]
    fn kernel_is_a_normalized_gaussian_like_window() {
        let k = kernel();
        let sum: f32 = k.iter().sum();
        eprintln!("kernel {k:?} sum {sum}");
        assert!((sum - 1.0).abs() < 1e-3, "{sum}");
        assert!(k[RADIUS] > k[RADIUS - 1] && k[RADIUS - 1] > k[RADIUS - 2]);
        // interior of a constant plane stays constant
        let (w, h) = (32, 20);
        let src = vec![0.7f32; w * h];
        let mut out = vec![0f32; w * h];
        GaussianBlur::with_kernel(w, *kernel(), true).blur(&src, h, &mut out);
        assert!(
            (out[10 * w + 16] - 0.7).abs() < 1e-4,
            "{}",
            out[10 * w + 16]
        );
        assert!(out[0] < 0.7 * 0.9, "zero padding at the corner: {}", out[0]);
    }

    #[test]
    fn identical_images_score_100_with_zero_map() {
        let mut img = Image3F::new(64, 48);
        for y in 0..48 {
            for x in 0..64 {
                img.plane_row_mut(0, y)[x] = 0.01 * ((x * 7 + y * 3) % 11) as f32 - 0.05;
                img.plane_row_mut(1, y)[x] = 0.2 + 0.4 * ((x + y) % 5) as f32 / 5.0;
                img.plane_row_mut(2, y)[x] = 0.3 + 0.2 * ((x * y) % 3) as f32 / 3.0;
            }
        }
        let (d, map) = ssimulacra2_deficit_and_map(&img, &img);
        assert!(d.abs() < 1e-4, "{d}");
        assert!(map.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn map_sum_tracks_the_deficit_and_degradation_lowers_the_score() {
        let mut img = Image3F::new(96, 80);
        let mut rng = 12345u32;
        for c in 0..3 {
            for y in 0..80 {
                for x in 0..96 {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (rng >> 24) as f32 / 255.0;
                    img.plane_row_mut(c, y)[x] =
                        0.1 + 0.6 * ((x / 8 + y / 8) % 2) as f32 + 0.1 * noise;
                }
            }
        }
        let mut rec = img.clone();
        for y in 0..80 {
            for x in 0..96 {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                rec.plane_row_mut(1, y)[x] += 0.03 * ((rng >> 24) as f32 / 255.0 - 0.5);
            }
        }
        let (d, map) = ssimulacra2_deficit_and_map(&img, &rec);
        assert!(d > 0.5 && d < 60.0, "{d}");
        let s: f32 = map.iter().sum::<f32>() / map.len() as f32;
        // the map is the local derivative of a concave mapping: positive and of the deficit's order
        assert!(s > 0.1 * d && s < 2.0 * d, "map sum {s} vs deficit {d}");
        let mut worse = rec.clone();
        for y in 0..80 {
            for x in 0..96 {
                worse.plane_row_mut(1, y)[x] = rec.plane_row(1, y)[x] * 1.5 - 0.1;
            }
        }
        let (d2, _) = ssimulacra2_deficit_and_map(&img, &worse);
        assert!(d2 > d, "{d2} <= {d}");
    }
}

/// What one `update_checkpointed` overwrote.
#[derive(Default)]
pub(crate) struct Ssimulacra2Checkpoint {
    rects: Vec<(Rect, Rect)>,
    data: Vec<f32>,
    sums: Vec<[[(f64, f64); 3]; 3]>,
}

fn save_rows(buf: &[f32], w: usize, r: Rect, out: &mut Vec<f32>) {
    if buf.is_empty() {
        return;
    }
    for y in r.y0..r.y1 {
        out.extend_from_slice(&buf[y * w + r.x0..y * w + r.x1]);
    }
}

fn load_rows(buf: &mut [f32], w: usize, r: Rect, data: &[f32], mut at: usize) -> usize {
    if buf.is_empty() {
        return at;
    }
    let n = r.x1 - r.x0;
    for y in r.y0..r.y1 {
        buf[y * w + r.x0..y * w + r.x1].copy_from_slice(&data[at..at + n]);
        at += n;
    }
    at
}

/// Incremental SSIMULACRA2 of a fixed source against a mutable reconstruction:
/// after a localized change of the reconstruction only the affected pixels (plus
/// the blur halo) of every scale are recomputed and the pooled sums updated.
pub(crate) struct Ssimulacra2State {
    w0: usize,
    h0: usize,
    dims: Vec<(usize, usize)>,
    /// Reconstruction's linear RGB planes per scale: `[scale][channel]`.
    lin_rec: Vec<[Vec<f32>; 3]>,
    /// Positive XYB planes per scale.
    pos_src: Vec<[Vec<f32>; 3]>,
    pos_rec: Vec<[Vec<f32>; 3]>,
    /// Blurred moments per scale and plane. Unweighted planes are empty;
    /// squared/cross moments are empty unless the SSIM map is weighted.
    mu1: Vec<[Vec<f32>; 3]>,
    s11: Vec<[Vec<f32>; 3]>,
    mu2: Vec<[Vec<f32>; 3]>,
    s22: Vec<[Vec<f32>; 3]>,
    s12: Vec<[Vec<f32>; 3]>,
    /// Maps per scale, plane: ssim, artifact, detail. Unweighted maps are empty.
    maps: Vec<[[Vec<f32>; 3]; 3]>,
    /// Pooled sums per scale, plane, map: (Σx, Σx⁴). Unweighted fourth sums
    /// stay zero; first sums also detect exact-zero maps after local reverts.
    sums: Vec<[[(f64, f64); 3]; 3]>,
    /// Persistent row scratch, also used to form products for moment blurs.
    blur: GaussianBlur,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Rect {
    fn expand(self, r: usize, w: usize, h: usize) -> Rect {
        Rect {
            x0: self.x0.saturating_sub(r),
            y0: self.y0.saturating_sub(r),
            x1: (self.x1 + r).min(w),
            y1: (self.y1 + r).min(h),
        }
    }
    fn half(self, w: usize, h: usize) -> Rect {
        Rect {
            x0: self.x0 / 2,
            y0: self.y0 / 2,
            x1: self.x1.div_ceil(2).min(w),
            y1: self.y1.div_ceil(2).min(h),
        }
    }
    fn is_empty(self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
}

impl Ssimulacra2State {
    pub(crate) fn new(source: &Image3F, recon: &Image3F) -> Self {
        let (w0, h0) = (source.xsize(), source.ysize());
        let mut st = Ssimulacra2State {
            w0,
            h0,
            dims: Vec::new(),
            lin_rec: Vec::new(),
            pos_src: Vec::new(),
            pos_rec: Vec::new(),
            mu1: Vec::new(),
            s11: Vec::new(),
            mu2: Vec::new(),
            s22: Vec::new(),
            s12: Vec::new(),
            maps: Vec::new(),
            sums: Vec::new(),
            blur: GaussianBlur::with_kernel(w0, *kernel(), true),
        };
        let mut la = linear_of_xyb(source);
        let mut lb = linear_of_xyb(recon);
        let (mut w, mut h) = (w0, h0);
        for scale in 0..NUM_SCALES {
            if w < 8 || h < 8 {
                break;
            }
            st.dims.push((w, h));
            st.pos_src.push(positive_xyb_of_linear(&la));
            st.pos_rec.push(positive_xyb_of_linear(&lb));
            st.lin_rec.push(std::mem::take(&mut lb));
            let n = w * h;
            let active: [[bool; 3]; 3] = std::array::from_fn(|c| active_maps(c, scale));
            let buffer = |needed| if needed { vec![0.0; n] } else { Vec::new() };
            st.mu1
                .push(std::array::from_fn(|c| buffer(active[c].contains(&true))));
            st.mu2
                .push(std::array::from_fn(|c| buffer(active[c].contains(&true))));
            st.s11.push(std::array::from_fn(|c| buffer(active[c][0])));
            st.s22.push(std::array::from_fn(|c| buffer(active[c][0])));
            st.s12.push(std::array::from_fn(|c| buffer(active[c][0])));
            st.maps.push(std::array::from_fn(|c| {
                std::array::from_fn(|m| buffer(active[c][m]))
            }));
            st.sums.push([[(0.0, 0.0); 3]; 3]);
            let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
            if scale + 1 == NUM_SCALES || nw < 8 || nh < 8 {
                break;
            }
            for c in 0..3 {
                la[c] = downsample(&la[c], w, h).0;
                lb[c] = downsample(&st.lin_rec[scale][c], w, h).0;
            }
            (w, h) = (nw, nh);
        }
        for s in 0..st.dims.len() {
            let (w, h) = st.dims[s];
            st.blur.set_width(w);
            for c in 0..3 {
                let active = active_maps(c, s);
                if !active.contains(&true) {
                    continue;
                }
                let src = &st.pos_src[s][c];
                st.blur.blur(src, h, &mut st.mu1[s][c]);
                if active[0] {
                    st.blur.blur_product([src, src], h, &mut st.s11[s][c]);
                }
                // full-plane rec moments with the SIMD blur; maps follow below
                let rec = &st.pos_rec[s][c];
                st.blur.blur(rec, h, &mut st.mu2[s][c]);
                if active[0] {
                    st.blur.blur_product([rec, rec], h, &mut st.s22[s][c]);
                    st.blur.blur_product([src, rec], h, &mut st.s12[s][c]);
                }
            }
            st.refresh_maps(
                s,
                Rect {
                    x0: 0,
                    y0: 0,
                    x1: w,
                    y1: h,
                },
            );
        }
        st
    }

    /// Recompute the weighted maps and their required rec moments at scale `s` over
    /// `r` (whose inputs changed within `r` shrunk by the blur radius), updating the sums.
    fn refresh_rec_moments_and_maps(&mut self, s: usize, r: Rect) {
        let (w, h) = self.dims[s];
        if r.is_empty() {
            return;
        }
        self.blur.set_width(r.x1 - r.x0);
        let offset = r.y0 * w + r.x0;
        for c in 0..3 {
            let active = active_maps(c, s);
            if !active.contains(&true) {
                continue;
            }
            let rec = &self.pos_rec[s][c];
            let src = &self.pos_src[s][c];
            self.blur.blur_region(
                rec,
                w,
                h,
                [r.x0..r.x1, r.y0..r.y1],
                &mut self.mu2[s][c][offset..],
                w,
            );
            if !active[0] {
                continue;
            }
            self.blur.blur_product_region(
                [rec, rec],
                w,
                h,
                [r.x0..r.x1, r.y0..r.y1],
                &mut self.s22[s][c][offset..],
                w,
            );
            self.blur.blur_product_region(
                [src, rec],
                w,
                h,
                [r.x0..r.x1, r.y0..r.y1],
                &mut self.s12[s][c][offset..],
                w,
            );
        }
        self.refresh_maps(s, r);
    }

    /// Recompute the weighted maps over `r` from their current moments.
    fn refresh_maps(&mut self, s: usize, r: Rect) {
        let (w, _h) = self.dims[s];
        for c in 0..3 {
            let active = active_maps(c, s);
            if !active.contains(&true) {
                continue;
            }
            let base = (c * NUM_SCALES + s) * 6;
            let need_fourth: [bool; 3] = std::array::from_fn(|m| WEIGHTS[base + 3 + m] != 0.0);
            let rec = &self.pos_rec[s][c];
            let src = &self.pos_src[s][c];
            let mu1 = &self.mu1[s][c];
            let mu2 = &self.mu2[s][c];
            let s11 = &self.s11[s][c];
            let s22 = &self.s22[s][c];
            let s12 = &self.s12[s][c];
            let [ssim, artifact, detail] = &mut self.maps[s][c];
            for y in r.y0..r.y1 {
                let span = y * w + r.x0..y * w + r.x1;
                if active[0] {
                    replace_map(
                        &mut ssim[span.clone()],
                        [
                            &mu1[span.clone()],
                            &mu2[span.clone()],
                            &s11[span.clone()],
                            &s22[span.clone()],
                            &s12[span.clone()],
                        ],
                        &mut self.sums[s][c][0],
                        need_fourth[0],
                        |[m1, m2, v11, v22, v12]| {
                            let num_m = 1.0 - (m1 - m2) * (m1 - m2);
                            let num_s = 2.0 * (v12 - m1 * m2) + C2;
                            let denom_s = (v11 - m1 * m1) + (v22 - m2 * m2) + C2;
                            (1.0 - num_m * num_s / denom_s).max(0.0)
                        },
                    );
                }
                let inputs = [
                    &src[span.clone()],
                    &rec[span.clone()],
                    &mu1[span.clone()],
                    &mu2[span.clone()],
                ];
                if active[1] {
                    replace_map(
                        &mut artifact[span.clone()],
                        inputs,
                        &mut self.sums[s][c][1],
                        need_fourth[1],
                        |[a, b, m1, m2]| {
                            ((1.0 + (b - m2).abs()) / (1.0 + (a - m1).abs()) - 1.0).max(0.0)
                        },
                    );
                }
                if active[2] {
                    replace_map(
                        &mut detail[span],
                        inputs,
                        &mut self.sums[s][c][2],
                        need_fourth[2],
                        |[a, b, m1, m2]| {
                            (1.0 - (1.0 + (b - m2).abs()) / (1.0 + (a - m1).abs())).max(0.0)
                        },
                    );
                }
            }
            // Maps are nonnegative: an exact zero first sum implies a zero
            // fourth sum. Clear residual round-off after a candidate revert;
            // the fourth root would otherwise amplify it near a perfect match.
            for sums in &mut self.sums[s][c] {
                if sums.0 == 0.0 {
                    sums.1 = 0.0;
                }
            }
        }
    }

    /// Weighted pooled sum over all scales, planes and maps.
    fn weighted_sum(&self) -> f64 {
        let mut sum = 0f64;
        for s in 0..self.dims.len() {
            let n = (self.dims[s].0 * self.dims[s].1) as f64;
            for c in 0..3 {
                let base = (c * 6 + s) * 6;
                for m in 0..3 {
                    let (s1, s4) = self.sums[s][c][m];
                    let m1 = (s1 / n).max(0.0);
                    let n4 = (s4 / n).max(0.0).sqrt().sqrt();
                    sum += WEIGHTS[base + m] * m1 + WEIGHTS[base + 3 + m] * n4;
                }
            }
        }
        sum
    }

    /// `100 - score`.
    pub(crate) fn deficit(&self) -> f32 {
        (100.0 - score_from_sum(self.weighted_sum())) as f32
    }

    /// The reconstruction changed inside `r` (full-resolution pixels): refresh.
    pub(crate) fn update(&mut self, recon: &Image3F, r: Rect) {
        if r.is_empty() || self.dims.is_empty() {
            return;
        }
        let (w0, h0) = (self.w0, self.h0);
        let r0 = Rect {
            x0: r.x0,
            y0: r.y0,
            x1: r.x1.min(w0),
            y1: r.y1.min(h0),
        };
        if r0.is_empty() {
            return;
        }
        // scale 0: linear + positive XYB of the changed pixels
        let m = crate::xyb::XybMatrix::SPEC;
        let [r, g, b] = &mut self.lin_rec[0];
        for y in r0.y0..r0.y1 {
            let span = y * w0 + r0.x0..y * w0 + r0.x1;
            crate::xyb::xyb_to_rgb_rows(
                &m,
                [
                    &recon.plane_row(0, y)[r0.x0..r0.x1],
                    &recon.plane_row(1, y)[r0.x0..r0.x1],
                    &recon.plane_row(2, y)[r0.x0..r0.x1],
                ],
                [&mut r[span.clone()], &mut g[span.clone()], &mut b[span]],
            );
        }
        let mut rs = r0;
        for s in 0..self.dims.len() {
            let (w, h) = self.dims[s];
            if s > 0 {
                // downsample the changed region from the previous scale
                let (pw, ph) = self.dims[s - 1];
                rs = rs.half(w, h);
                let (previous, current) = self.lin_rec.split_at_mut(s);
                let current = &mut current[0];
                for y in rs.y0..rs.y1 {
                    let (y0, y1) = ((2 * y).min(ph - 1), (2 * y + 1).min(ph - 1));
                    let (x0, x1) = (2 * rs.x0, (2 * rs.x1).min(pw));
                    for c in 0..3 {
                        let p = &previous[s - 1][c];
                        downsample_rows(
                            &p[y0 * pw + x0..y0 * pw + x1],
                            &p[y1 * pw + x0..y1 * pw + x1],
                            &mut current[c][y * w + rs.x0..y * w + rs.x1],
                        );
                    }
                }
            }
            let lin = &self.lin_rec[s];
            let [x, y, b] = &mut self.pos_rec[s];
            for row in rs.y0..rs.y1 {
                let span = row * w + rs.x0..row * w + rs.x1;
                positive_xyb_into(
                    [
                        &lin[0][span.clone()],
                        &lin[1][span.clone()],
                        &lin[2][span.clone()],
                    ],
                    [&mut x[span.clone()], &mut y[span.clone()], &mut b[span]],
                );
            }
            // moments/maps depend on rec within the blur radius
            let region = rs.expand(RADIUS, w, h);
            self.refresh_rec_moments_and_maps(s, region);
        }
    }

    /// Per scale, the rect `update(_, r)` rewrites in the reconstruction
    /// planes and the (blur-expanded) rect it rewrites in moments and maps.
    fn update_rects(&self, r: Rect) -> Vec<(Rect, Rect)> {
        let r0 = Rect {
            x0: r.x0,
            y0: r.y0,
            x1: r.x1.min(self.w0),
            y1: r.y1.min(self.h0),
        };
        let mut out = Vec::with_capacity(self.dims.len());
        if r.is_empty() || self.dims.is_empty() || r0.is_empty() {
            return out;
        }
        let mut rs = r0;
        for s in 0..self.dims.len() {
            let (w, h) = self.dims[s];
            if s > 0 {
                rs = rs.half(w, h);
            }
            out.push((rs, rs.expand(RADIUS, w, h)));
        }
        out
    }

    /// `update`, first saving everything it overwrites into `cp` so that
    /// `rollback(cp)` restores the previous state exactly with copies only.
    pub(crate) fn update_checkpointed(
        &mut self,
        recon: &Image3F,
        r: Rect,
        cp: &mut Ssimulacra2Checkpoint,
    ) {
        cp.data.clear();
        cp.rects = self.update_rects(r);
        cp.sums.clone_from(&self.sums);
        for (s, &(rs, region)) in cp.rects.iter().enumerate() {
            let w = self.dims[s].0;
            for c in 0..3 {
                save_rows(&self.lin_rec[s][c], w, rs, &mut cp.data);
                save_rows(&self.pos_rec[s][c], w, rs, &mut cp.data);
                for buf in [&self.mu2[s][c], &self.s22[s][c], &self.s12[s][c]] {
                    save_rows(buf, w, region, &mut cp.data);
                }
                for map in &self.maps[s][c] {
                    save_rows(map, w, region, &mut cp.data);
                }
            }
        }
        self.update(recon, r);
    }

    /// Undo the `update_checkpointed` that filled `cp`.
    pub(crate) fn rollback(&mut self, cp: &Ssimulacra2Checkpoint) {
        let mut at = 0usize;
        for (s, &(rs, region)) in cp.rects.iter().enumerate() {
            let w = self.dims[s].0;
            for c in 0..3 {
                at = load_rows(&mut self.lin_rec[s][c], w, rs, &cp.data, at);
                at = load_rows(&mut self.pos_rec[s][c], w, rs, &cp.data, at);
                at = load_rows(&mut self.mu2[s][c], w, region, &cp.data, at);
                at = load_rows(&mut self.s22[s][c], w, region, &cp.data, at);
                at = load_rows(&mut self.s12[s][c], w, region, &cp.data, at);
                for m in 0..3 {
                    at = load_rows(&mut self.maps[s][c][m], w, region, &cp.data, at);
                }
            }
        }
        self.sums.clone_from(&cp.sums);
    }

    /// Per-pixel linearized map in the loop's convention (mean = deficit).
    pub(crate) fn map(&self) -> Vec<f32> {
        let (w0, h0) = (self.w0, self.h0);
        let mut map = vec![0f32; w0 * h0];
        let sum = self.weighted_sum();
        let mut acc = Vec::new();
        for s in 0..self.dims.len() {
            let (w, h) = self.dims[s];
            let n = (w * h) as f64;
            acc.resize(w * h, 0.0);
            acc.fill(0.0);
            for c in 0..3 {
                let base = (c * 6 + s) * 6;
                for m in 0..3 {
                    if WEIGHTS[base + m] == 0.0 && WEIGHTS[base + 3 + m] == 0.0 {
                        continue;
                    }
                    let (_, s4) = self.sums[s][c][m];
                    let n4 = (s4 / n).max(0.0).sqrt().sqrt();
                    add_map_contribution(
                        &self.maps[s][c][m],
                        WEIGHTS[base + m],
                        WEIGHTS[base + 3 + m],
                        n4,
                        &mut acc,
                    );
                }
            }
            accumulate_scale_map(&acc, w0, h0, s, &mut map);
        }
        let scale = deficit_slope(sum) as f32 * (w0 * h0) as f32;
        for v in &mut map {
            *v *= scale;
        }
        map
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    #[test]
    #[ignore = "manual SSIMULACRA2 map and pooling timing probe"]
    fn benchmark_map_math_and_pooling() {
        use std::{hint::black_box, time::Instant};
        let (w, h) = (768, 513);
        let src = textured(7, w, h);
        let rec = textured(8, w, h);
        let mut state = Ssimulacra2State::new(&src, &rec);
        let start = Instant::now();
        for _ in 0..32 {
            for s in 0..state.dims.len() {
                let (w, h) = state.dims[s];
                state.refresh_maps(
                    s,
                    Rect {
                        x0: 0,
                        y0: 0,
                        x1: w,
                        y1: h,
                    },
                );
            }
            black_box(&state.sums);
        }
        eprintln!(
            "refresh_maps_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0 / 32.0
        );
        let mut acc = vec![0.0; w * h];
        let start = Instant::now();
        for _ in 0..32 {
            for s in 0..state.dims.len() {
                let (w, h) = state.dims[s];
                acc.resize(w * h, 0.0);
                acc.fill(0.0);
                for c in 0..3 {
                    let base = (c * NUM_SCALES + s) * 6;
                    for m in 0..3 {
                        black_box(pool(
                            black_box(&state.maps[s][c][m]),
                            WEIGHTS[base + m],
                            WEIGHTS[base + 3 + m],
                            &mut acc,
                        ));
                    }
                }
            }
            black_box(&acc);
        }
        eprintln!(
            "pool_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0 / 32.0
        );
        let start = Instant::now();
        for _ in 0..32 {
            black_box(state.map());
        }
        eprintln!(
            "state_map_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0 / 32.0
        );
    }

    #[test]
    #[ignore = "manual inverse XYB conversion timing probe"]
    fn benchmark_inverse_xyb_to_rgb() {
        use std::{hint::black_box, time::Instant};
        // Preserve the old full-image loop as the benchmark's baseline.
        let scalar = |img: &Image3F| {
            let (w, h) = (img.xsize(), img.ysize());
            let m = crate::xyb::XybMatrix::SPEC;
            let mut out = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
            for y in 0..h {
                let (x, yy, b) = (
                    img.plane_row(0, y),
                    img.plane_row(1, y),
                    img.plane_row(2, y),
                );
                for i in 0..w {
                    let rgb = crate::xyb::xyb_to_rgb_pixel_f32(&m, x[i], yy[i], b[i]);
                    for c in 0..3 {
                        out[c][y * w + i] = rgb[c];
                    }
                }
            }
            out
        };
        let img = textured(7, 768, 513);
        let expected = scalar(&img);
        let actual = linear_of_xyb(&img);
        for c in 0..3 {
            for (a, b) in actual[c].iter().zip(&expected[c]) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        let start = Instant::now();
        for _ in 0..64 {
            black_box(scalar(black_box(&img)));
        }
        let old = start.elapsed().as_secs_f64() * 1000.0 / 64.0;
        let start = Instant::now();
        for _ in 0..64 {
            black_box(linear_of_xyb(black_box(&img)));
        }
        let new = start.elapsed().as_secs_f64() * 1000.0 / 64.0;
        eprintln!(
            "inverse_scalar_ms={old:.3} inverse_simd_ms={new:.3} speedup={:.2}x",
            old / new
        );
    }

    #[test]
    #[ignore = "manual SSIMULACRA2 timing probe"]
    fn benchmark_full_score_state_and_local_updates() {
        use std::{hint::black_box, time::Instant};
        let (w, h) = (768, 513);
        let src = textured(7, w, h);
        let mut rec = textured(8, w, h);
        let mut state = Ssimulacra2State::new(&src, &rec);
        let plane_elements: usize = [
            &state.lin_rec,
            &state.pos_src,
            &state.pos_rec,
            &state.mu1,
            &state.mu2,
            &state.s11,
            &state.s22,
            &state.s12,
        ]
        .into_iter()
        .map(|scales| scales.iter().flatten().map(Vec::capacity).sum::<usize>())
        .sum::<usize>()
            + state
                .maps
                .iter()
                .flatten()
                .flatten()
                .map(Vec::capacity)
                .sum::<usize>();
        let dense_elements: usize = state.dims.iter().map(|&(w, h)| 33 * w * h).sum();
        eprintln!(
            "state_plane_bytes={} dense_plane_bytes={}",
            plane_elements * std::mem::size_of::<f32>(),
            dense_elements * std::mem::size_of::<f32>()
        );
        let start = Instant::now();
        for _ in 0..8 {
            black_box(ssimulacra2_deficit_and_map(&src, &rec));
        }
        eprintln!(
            "full_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0 / 8.0
        );
        let start = Instant::now();
        for _ in 0..8 {
            black_box(Ssimulacra2State::new(&src, &rec));
        }
        eprintln!(
            "state_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0 / 8.0
        );
        for size in [32, 256] {
            let r = Rect {
                x0: 17,
                y0: 19,
                x1: 17 + size,
                y1: 19 + size,
            };
            let start = Instant::now();
            for _ in 0..32 {
                rec.plane_row_mut(1, r.y0)[r.x0] += 0.0001;
                state.update(&rec, r);
                black_box(state.deficit());
            }
            eprintln!(
                "update_{size}_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0 / 32.0
            );
        }
        let (d, map) = ssimulacra2_deficit_and_map(&src, &rec);
        eprintln!(
            "deficit={d:.9} map_sum={:.9} state_deficit={:.9}",
            map.iter().map(|&v| v as f64).sum::<f64>(),
            state.deficit()
        );
    }

    fn textured(seed: u32, w: usize, h: usize) -> Image3F {
        let mut img = Image3F::new(w, h);
        let mut rng = seed;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (rng >> 24) as f32 / 255.0;
                    let base = if c == 1 { 0.3 } else { 0.0 };
                    img.plane_row_mut(c, y)[x] = base
                        + 0.3 * ((x / 8 + y / 8) % 2) as f32 * (if c == 1 { 1.0 } else { 0.1 })
                        + 0.05 * noise;
                }
            }
        }
        img
    }

    #[test]
    fn state_retains_only_weighted_fourth_norms_after_updates_and_reverts() {
        let (w, h) = (257, 259);
        let src = textured(7, w, h);
        let mut rec = textured(8, w, h);
        let mut state = Ssimulacra2State::new(&src, &rec);
        assert_eq!(state.dims.len(), NUM_SCALES);
        let check = |state: &Ssimulacra2State| {
            for s in 0..state.dims.len() {
                for c in 0..3 {
                    let base = (c * NUM_SCALES + s) * 6;
                    for m in 0..3 {
                        let (s1, s4) = state.sums[s][c][m];
                        let expected = map_sums(&state.maps[s][c][m]);
                        // The first sum also detects exact-zero maps when only
                        // their fourth norm is weighted.
                        assert!((s1 - expected.0).abs() <= 2e-12 * expected.0.abs().max(1.0));
                        if WEIGHTS[base + 3 + m] == 0.0 {
                            assert_eq!(s4, 0.0, "scale={s}, plane={c}, map={m}");
                        } else {
                            assert!((s4 - expected.1).abs() <= 2e-12 * expected.1.abs().max(1.0));
                        }
                    }
                }
            }
        };
        check(&state);
        let r = Rect {
            x0: w - 13,
            y0: h - 9,
            x1: w,
            y1: h,
        };
        for row in r.y0..r.y1 {
            for v in &mut rec.plane_row_mut(1, row)[r.x0..r.x1] {
                *v += 0.001;
            }
        }
        state.update(&rec, r);
        check(&state);
        state.update(&textured(8, w, h), r);
        check(&state);
    }

    #[test]
    fn fused_map_stays_close_to_unfused_contributions_and_expansion() {
        let mut max_relative_error = 0.0f32;
        for (w0, h0) in [(17, 19), (129, 67), (257, 259), (768, 513)] {
            let src = textured(7, w0, h0);
            for seed in [7, 8] {
                let rec = textured(seed, w0, h0);
                let state = Ssimulacra2State::new(&src, &rec);
                let mut expected = vec![0.0; w0 * h0];
                for (s, &(w, h)) in state.dims.iter().enumerate() {
                    let n = (w * h) as f64;
                    let mut acc = vec![0.0; w * h];
                    for c in 0..3 {
                        let base = (c * NUM_SCALES + s) * 6;
                        for m in 0..3 {
                            let (w1, w4) = (WEIGHTS[base + m], WEIGHTS[base + 3 + m]);
                            if w1 == 0.0 && w4 == 0.0 {
                                continue;
                            }
                            let n4 = (state.sums[s][c][m].1 / n).max(0.0).sqrt().sqrt();
                            let p = if w4 == 0.0 || n4 == 0.0 {
                                MapContribution {
                                    linear: (w1 / n) as f32,
                                    inv_norm: 0.0,
                                    norm_weight: 0.0,
                                }
                            } else {
                                let p = MapContribution {
                                    linear: (w1 / n) as f32,
                                    inv_norm: (1.0 / n4) as f32,
                                    norm_weight: (w4 * n4 / (4.0 * n)) as f32,
                                };
                                assert!(p.inv_norm.is_normal() && p.norm_weight.is_normal());
                                p
                            };
                            contribution_scalar(&state.maps[s][c][m], p, &mut acc);
                        }
                    }
                    // Preserve the previous per-output-pixel scale expansion
                    // as an independent oracle for indexing and arithmetic.
                    let step = 1 << s;
                    let share = 1.0 / (step * step) as f32;
                    for y in 0..h0 {
                        for x in 0..w0 {
                            expected[y * w0 + x] +=
                                acc[(y / step).min(h - 1) * w + (x / step).min(w - 1)] * share;
                        }
                    }
                }
                let factor = deficit_slope(state.weighted_sum()) as f32 * (w0 * h0) as f32;
                for v in &mut expected {
                    *v *= factor;
                }
                for (&actual, &reference) in state.map().iter().zip(&expected) {
                    let error = (actual - reference).abs() / reference.abs().max(f32::MIN_POSITIVE);
                    max_relative_error = max_relative_error.max(error);
                    assert!(error <= 1e-6, "{w0}x{h0}, seed={seed}: {error}");
                }
            }
        }
        eprintln!("FMA map max relative error vs unfused: {max_relative_error}");
    }

    #[test]
    fn incremental_update_matches_full_recompute() {
        let src = textured(7, 160, 96);
        let mut rec = textured(7, 160, 96);
        for y in 20..60 {
            for x in 40..90 {
                rec.plane_row_mut(1, y)[x] += 0.02;
            }
        }
        let mut st = Ssimulacra2State::new(&src, &rec);
        let (d_full, map_full) = ssimulacra2_deficit_and_map(&src, &rec);
        assert!(
            (st.deficit() - d_full).abs() < 1e-3,
            "{} vs {d_full}",
            st.deficit()
        );
        let map_st = st.map();
        let diff: f32 = map_st
            .iter()
            .zip(&map_full)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / map_full.len() as f32;
        // Both paths use the same SIMD blur and pooling convention.
        assert!(diff < 1e-2, "{diff}");
        // change a region and update incrementally
        for y in 48..80 {
            for x in 96..128 {
                rec.plane_row_mut(0, y)[x] += 0.01;
                rec.plane_row_mut(1, y)[x] -= 0.03;
            }
        }
        st.update(
            &rec,
            Rect {
                x0: 96,
                y0: 48,
                x1: 128,
                y1: 80,
            },
        );
        let (d2, _) = ssimulacra2_deficit_and_map(&src, &rec);
        assert!(
            (st.deficit() - d2).abs() < 1e-3,
            "incremental {} vs full {d2}",
            st.deficit()
        );
        // and back
        for y in 48..80 {
            for x in 96..128 {
                rec.plane_row_mut(0, y)[x] -= 0.01;
                rec.plane_row_mut(1, y)[x] += 0.03;
            }
        }
        st.update(
            &rec,
            Rect {
                x0: 96,
                y0: 48,
                x1: 128,
                y1: 80,
            },
        );
        assert!(
            (st.deficit() - d_full).abs() < 1e-3,
            "revert {} vs {d_full}",
            st.deficit()
        );
    }

    #[test]
    fn repeated_updates_match_rebuild_at_odd_edges_and_reuse_blur_scratch() {
        for (w, h) in [(9, 8), (31, 17), (257, 259)] {
            let src = textured(7, w, h);
            let mut rec = textured(8, w, h);
            let mut state = Ssimulacra2State::new(&src, &rec);
            // Warm the largest possible scratch before checking smaller updates.
            state.update(
                &rec,
                Rect {
                    x0: 0,
                    y0: 0,
                    x1: w,
                    y1: h,
                },
            );
            let scratch = state.blur.scratch_layout();
            let regions = [
                Rect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                Rect {
                    x0: w - 1,
                    y0: h - 1,
                    x1: w,
                    y1: h,
                },
                Rect {
                    x0: w / 3,
                    y0: h / 3,
                    x1: w.div_ceil(2),
                    y1: h.div_ceil(2),
                },
                Rect {
                    x0: 0,
                    y0: h - 3,
                    x1: w,
                    y1: h,
                },
                Rect {
                    x0: w - 3,
                    y0: 0,
                    x1: w,
                    y1: h,
                },
            ];
            for r in regions.into_iter().cycle().take(15) {
                let saved = rec.clone();
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        for v in &mut rec.plane_row_mut(c, y)[r.x0..r.x1] {
                            *v += if c == 1 { -0.003 } else { 0.001 };
                        }
                    }
                }
                for image in [&rec, &saved] {
                    state.update(image, r);
                    let fresh = Ssimulacra2State::new(&src, image);
                    assert!(
                        (state.deficit() - fresh.deficit()).abs() < 1e-3,
                        "{w}x{h}, {r:?}: {} vs {}",
                        state.deficit(),
                        fresh.deficit()
                    );
                    for (actual, expected) in state
                        .lin_rec
                        .iter()
                        .flatten()
                        .zip(fresh.lin_rec.iter().flatten())
                    {
                        assert_eq!(actual, expected, "linear pyramid: {w}x{h}, {r:?}");
                    }
                    for (actual, expected) in [&state.mu2, &state.s22, &state.s12]
                        .into_iter()
                        .zip([&fresh.mu2, &fresh.s22, &fresh.s12])
                    {
                        for (a, b) in actual
                            .iter()
                            .flatten()
                            .flatten()
                            .zip(expected.iter().flatten().flatten())
                        {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "moments: {w}x{h}, {r:?}: {a} vs {b}"
                            );
                        }
                    }
                    let map = state.map();
                    let fresh_map = fresh.map();
                    let diff = map
                        .iter()
                        .zip(&fresh_map)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>()
                        / (w * h) as f32;
                    assert!(diff < 1e-2, "map: {w}x{h}, {r:?}: {diff}");
                    assert_eq!(state.blur.scratch_layout(), scratch);
                }
                rec = saved;
            }
        }
    }

    #[test]
    fn weighted_state_matches_full_maps_across_all_six_scales() {
        let (w, h) = (257, 259);
        let src = textured(7, w, h);
        let mut rec = textured(8, w, h);
        let mut state = Ssimulacra2State::new(&src, &rec);
        assert_eq!(state.dims.len(), NUM_SCALES);
        let regions = [
            Rect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            Rect {
                x0: w - 7,
                y0: h - 5,
                x1: w,
                y1: h,
            },
            Rect {
                x0: 73,
                y0: 51,
                x1: 131,
                y1: 107,
            },
        ];
        for r in regions {
            let saved = rec.clone();
            for c in 0..3 {
                for y in r.y0..r.y1 {
                    for v in &mut rec.plane_row_mut(c, y)[r.x0..r.x1] {
                        *v += if c == 1 { -0.02 } else { 0.007 };
                    }
                }
            }
            for image in [&rec, &saved] {
                state.update(image, r);
                let (deficit, map) = ssimulacra2_deficit_and_map(&src, image);
                assert!(
                    (state.deficit() - deficit).abs() < 1e-4,
                    "{r:?}: {} vs {deficit}",
                    state.deficit()
                );
                let actual = state.map();
                let mean_error: f64 = actual
                    .iter()
                    .zip(&map)
                    .map(|(&a, &b)| (a as f64 - b as f64).abs())
                    .sum::<f64>()
                    / (w * h) as f64;
                assert!(mean_error < 1e-4, "{r:?}: map mean error {mean_error}");
            }
            rec = saved;
        }
    }

    #[test]
    fn updates_outside_the_image_or_below_minimum_scale_are_empty() {
        for (w, h) in [(0, 0), (7, 9), (9, 7), (9, 9)] {
            let src = textured(7, w, h);
            let mut state = Ssimulacra2State::new(&src, &src);
            state.update(
                &src,
                Rect {
                    x0: 0,
                    y0: 0,
                    x1: w,
                    y1: h,
                },
            );
            state.update(
                &src,
                Rect {
                    x0: w,
                    y0: h,
                    x1: w + 1,
                    y1: h + 1,
                },
            );
            assert_eq!(state.deficit(), 0.0);
            assert!(state.map().iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn repeated_local_reverts_preserve_near_identical_scores() {
        let (w, h) = (97, 81);
        let src = textured(7, w, h);
        let mut rec = src.clone();
        let mut state = Ssimulacra2State::new(&src, &rec);
        let initial = state.deficit();
        for iteration in 0..256 {
            let x = iteration * 13 % (w - 7);
            let y = iteration * 7 % (h - 5);
            let r = Rect {
                x0: x,
                y0: y,
                x1: x + 7,
                y1: y + 5,
            };
            for row in r.y0..r.y1 {
                for v in &mut rec.plane_row_mut(1, row)[r.x0..r.x1] {
                    *v += 0.001;
                }
            }
            state.update(&rec, r);
            if iteration % 32 == 0 {
                let (full, _) = ssimulacra2_deficit_and_map(&src, &rec);
                assert!(
                    (state.deficit() - full).abs() < 1e-4,
                    "{iteration}: {} vs {full}",
                    state.deficit()
                );
            }
            state.update(&src, r);
            for row in r.y0..r.y1 {
                rec.plane_row_mut(1, row)[r.x0..r.x1]
                    .copy_from_slice(&src.plane_row(1, row)[r.x0..r.x1]);
            }
            assert!(
                (state.deficit() - initial).abs() < 1e-4,
                "revert {iteration}: {} vs {initial}",
                state.deficit()
            );
        }
    }
}
