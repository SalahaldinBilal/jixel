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

//! Pyramid reduction and scoring without full-plane intermediate terms.
//! Pool vector results in their original pixel order to keep scores bit-exact.
#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub half: fn([&[f32]; 2], &mut [f32]),
    pub half_y: fn([&[f32]; 2], &mut [f32]),
    pub half_in_place: fn(&mut [f32], usize, usize),
    pub variance: fn(&[f32], &[f32], &mut [f32]),
    pub pool: fn([&[f32]; 5], [f32; 2]) -> [f32; 2],
    pub gradients: fn([&[f32]; 4], [f32; 3]) -> [f32; 3],
}
pub(super) fn selected() -> Kernels {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return avx::kernels();
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        return sse::kernels();
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return neon::kernels();
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return wasm::kernels();
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    scalar()
}
fn half<const BIAS: bool>(src: [&[f32]; 2], out: &mut [f32]) {
    for (x, v) in out.iter_mut().enumerate() {
        let tap = |row: usize, i| {
            if BIAS {
                src[row][i] + 0.01
            } else {
                src[row][i]
            }
        };
        *v = 0.25 * (tap(0, 2 * x) + tap(1, 2 * x) + tap(0, 2 * x + 1) + tap(1, 2 * x + 1));
    }
}
fn half_in_place(values: &mut [f32], w: usize, h: usize) {
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            let i = 2 * y * w + 2 * x;
            values[y * (w / 2) + x] =
                0.25 * (values[i] + values[i + w] + values[i + 1] + values[i + w + 1]);
        }
    }
}
fn variance(mean: &[f32], square: &[f32], out: &mut [f32]) {
    for ((out, &m), &s) in out.iter_mut().zip(mean).zip(square) {
        *out = (s - m * m).max(0.0);
    }
}
fn pool(src: [&[f32]; 5], mut sum: [f32; 2]) -> [f32; 2] {
    let [mean, square, cross, reference, variance] = src;
    for x in 0..mean.len() {
        let (a, b) = (reference[x], mean[x]);
        let var = (square[x] - b * b).max(0.0);
        let cov = if variance[x] == 0.0 || var == 0.0 {
            0.0
        } else {
            cross[x] - a * b
        };
        let s = (1.0
            - (1.0 - (a - b) * (a - b)) * (2.0 * cov + 0.0009) / (variance[x] + var + 0.0009))
            .max(0.0);
        sum[0] += s;
        sum[1] += s * s * s * s;
    }
    sum
}
fn gradients(src: [&[f32]; 4], mut sum: [f32; 3]) -> [f32; 3] {
    for x in 0..src[0].len() {
        let a = src[1][x] - src[0][x];
        let b = src[3][x] - src[2][x];
        sum[0] += (b - a).abs();
        sum[1] += (b.abs() - a.abs()).max(0.0);
        let loss = (a.abs() - b.abs()).max(0.0);
        sum[2] += loss * loss * loss * loss;
    }
    sum
}
#[allow(dead_code)]
pub(super) fn scalar() -> Kernels {
    Kernels {
        half: half::<false>,
        half_y: half::<true>,
        half_in_place,
        variance,
        pool,
        gradients,
    }
}
#[allow(unused_macros)]
macro_rules! backend {
    ($feature:literal,$lanes:literal,$load:ident,$store:ident,$splat:ident,$add:ident,$sub:ident,$mul:ident,$div:ident,$abs:ident,$max:ident,$nonzero:ident) => {
        #[target_feature(enable=$feature)]
        unsafe fn half<const BIAS: bool>(src: [&[f32]; 2], out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let mut taps = [$splat(0.0); 4];
                    for t in 0..4 {
                        let v: [f32; $lanes] =
                            std::array::from_fn(|i| src[t % 2][2 * (x + i) + t / 2]);
                        taps[t] = $load(v.as_ptr());
                        if BIAS {
                            taps[t] = $add(taps[t], $splat(0.01));
                        }
                    }
                    $store(
                        out.as_mut_ptr().add(x),
                        $mul(
                            $splat(0.25),
                            $add($add($add(taps[0], taps[1]), taps[2]), taps[3]),
                        ),
                    );
                }
            }
            super::half::<BIAS>(src.map(|s| &s[2 * end..]), &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        unsafe fn half_in_place(values: &mut [f32], w: usize, h: usize) {
            let ow = w / 2;
            let end = ow / $lanes * $lanes;
            for y in 0..h / 2 {
                unsafe {
                    for x in (0..end).step_by($lanes) {
                        let mut taps = [$splat(0.0); 4];
                        for t in 0..4 {
                            let v: [f32; $lanes] = std::array::from_fn(|i| {
                                values[(2 * y + t % 2) * w + 2 * (x + i) + t / 2]
                            });
                            taps[t] = $load(v.as_ptr());
                        }
                        $store(
                            values.as_mut_ptr().add(y * ow + x),
                            $mul(
                                $splat(0.25),
                                $add($add($add(taps[0], taps[1]), taps[2]), taps[3]),
                            ),
                        );
                    }
                }
                for x in end..ow {
                    let i = 2 * y * w + 2 * x;
                    values[y * ow + x] =
                        0.25 * (values[i] + values[i + w] + values[i + 1] + values[i + w + 1]);
                }
            }
        }
        #[target_feature(enable=$feature)]
        unsafe fn variance(mean: &[f32], square: &[f32], out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let m = $load(mean.as_ptr().add(x));
                    $store(
                        out.as_mut_ptr().add(x),
                        $max($sub($load(square.as_ptr().add(x)), $mul(m, m)), $splat(0.0)),
                    );
                }
            }
            super::variance(&mean[end..], &square[end..], &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        unsafe fn pool(src: [&[f32]; 5], mut sum: [f32; 2]) -> [f32; 2] {
            let end = src[0].len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let [b, square, cross, a, variance] = src.map(|s| $load(s.as_ptr().add(x)));
                    let zero = $splat(0.0);
                    let one = $splat(1.0);
                    let c = $splat(0.0009);
                    let var = $max($sub(square, $mul(b, b)), zero);
                    let cov = $nonzero($sub(cross, $mul(a, b)), variance, var);
                    let delta = $sub(a, b);
                    let s = $max(
                        $sub(
                            one,
                            $div(
                                $mul(
                                    $sub(one, $mul(delta, delta)),
                                    $add($mul($splat(2.0), cov), c),
                                ),
                                $add($add(variance, var), c),
                            ),
                        ),
                        zero,
                    );
                    let fourth = $mul($mul($mul(s, s), s), s);
                    let mut terms = [0.0; $lanes];
                    let mut powers = [0.0; $lanes];
                    $store(terms.as_mut_ptr(), s);
                    $store(powers.as_mut_ptr(), fourth);
                    for i in 0..$lanes {
                        sum[0] += terms[i];
                        sum[1] += powers[i];
                    }
                }
            }
            super::pool(src.map(|s| &s[end..]), sum)
        }
        #[target_feature(enable=$feature)]
        unsafe fn gradients(src: [&[f32]; 4], mut sum: [f32; 3]) -> [f32; 3] {
            let end = src[0].len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let [ai, aj, bi, bj] = src.map(|s| $load(s.as_ptr().add(x)));
                    let a = $sub(aj, ai);
                    let b = $sub(bj, bi);
                    let zero = $splat(0.0);
                    let error = $abs($sub(b, a));
                    let excess = $max($sub($abs(b), $abs(a)), zero);
                    let loss = $max($sub($abs(a), $abs(b)), zero);
                    let loss4 = $mul($mul($mul(loss, loss), loss), loss);
                    let mut e = [0.0; $lanes];
                    let mut ex = [0.0; $lanes];
                    let mut l = [0.0; $lanes];
                    $store(e.as_mut_ptr(), error);
                    $store(ex.as_mut_ptr(), excess);
                    $store(l.as_mut_ptr(), loss4);
                    for i in 0..$lanes {
                        sum[0] += e[i];
                        sum[1] += ex[i];
                        sum[2] += l[i];
                    }
                }
            }
            super::gradients(src.map(|s| &s[end..]), sum)
        }
        pub(super) fn kernels() -> super::Kernels {
            // The selector checks CPU features. The proxy supplies complete
            // source pairs for reduction and equal-length spans for scoring.
            super::Kernels {
                half: |s, o| unsafe { half::<false>(s, o) },
                half_y: |s, o| unsafe { half::<true>(s, o) },
                half_in_place: |v, w, h| unsafe { half_in_place(v, w, h) },
                variance: |m, s, o| unsafe { variance(m, s, o) },
                pool: |s, p| unsafe { pool(s, p) },
                gradients: |s, p| unsafe { gradients(s, p) },
            }
        }
    };
}
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "sse2")]
    fn abs(v: __m128) -> __m128 {
        _mm_andnot_ps(_mm_set1_ps(-0.0), v)
    }
    #[target_feature(enable = "sse2")]
    fn nonzero(v: __m128, a: __m128, b: __m128) -> __m128 {
        _mm_and_ps(
            v,
            _mm_and_ps(
                _mm_cmpneq_ps(a, _mm_setzero_ps()),
                _mm_cmpneq_ps(b, _mm_setzero_ps()),
            ),
        )
    }
    backend!(
        "sse2",
        4,
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_set1_ps,
        _mm_add_ps,
        _mm_sub_ps,
        _mm_mul_ps,
        _mm_div_ps,
        abs,
        _mm_max_ps,
        nonzero
    );
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx2")]
    fn abs(v: __m256) -> __m256 {
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), v)
    }
    #[target_feature(enable = "avx2")]
    fn nonzero(v: __m256, a: __m256, b: __m256) -> __m256 {
        _mm256_and_ps(
            v,
            _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_NEQ_UQ>(a, _mm256_setzero_ps()),
                _mm256_cmp_ps::<_CMP_NEQ_UQ>(b, _mm256_setzero_ps()),
            ),
        )
    }
    backend!(
        "avx2",
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_add_ps,
        _mm256_sub_ps,
        _mm256_mul_ps,
        _mm256_div_ps,
        abs,
        _mm256_max_ps,
        nonzero
    );
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    fn nonzero(v: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vbslq_f32(vorrq_u32(vceqzq_f32(a), vceqzq_f32(b)), vdupq_n_f32(0.0), v)
    }
    backend!(
        "neon",
        4,
        vld1q_f32,
        vst1q_f32,
        vdupq_n_f32,
        vaddq_f32,
        vsubq_f32,
        vmulq_f32,
        vdivq_f32,
        vabsq_f32,
        vmaxnmq_f32,
        nonzero
    );
}
#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn load(p: *const f32) -> v128 {
        unsafe { v128_load(p.cast()) }
    }
    #[target_feature(enable = "simd128")]
    unsafe fn store(p: *mut f32, v: v128) {
        unsafe { v128_store(p.cast(), v) }
    }
    #[target_feature(enable = "simd128")]
    fn max(a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, f32x4_gt(a, b))
    }
    #[target_feature(enable = "simd128")]
    fn nonzero(v: v128, a: v128, b: v128) -> v128 {
        let z = f32x4_splat(0.0);
        v128_bitselect(z, v, v128_or(f32x4_eq(a, z), f32x4_eq(b, z)))
    }
    backend!(
        "simd128",
        4,
        load,
        store,
        f32x4_splat,
        f32x4_add,
        f32x4_sub,
        f32x4_mul,
        f32x4_div,
        f32x4_abs,
        max,
        nonzero
    );
}
#[cfg(test)]
pub(super) fn available() -> Vec<(&'static str, Kernels)> {
    #[allow(unused_mut)]
    let mut k = vec![("scalar", scalar()), ("selected", selected())];
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        k.push(("avx2", avx::kernels()));
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        k.push(("sse2", sse::kernels()));
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    k.push(("neon", neon::kernels()));
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    k.push(("wasm", wasm::kernels()));
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }
    #[test]
    fn pyramid_and_pooling_match_scalar_bits_for_every_tail() {
        let scalar = scalar();
        for (name, k) in available() {
            for n in 1..=35 {
                let w = 2 * n + 1;
                let h = 7;
                let src: Vec<_> = (0..w * h)
                    .map(|i| ((i * 1877 + 941) % 3253) as f32 * 0.0001 - 0.03)
                    .collect();
                for (a, b) in [(scalar.half, k.half), (scalar.half_y, k.half_y)] {
                    let mut expected = vec![0.; n];
                    let mut actual = vec![0.; n];
                    a([&src[..w], &src[w..2 * w]], &mut expected);
                    b([&src[..w], &src[w..2 * w]], &mut actual);
                    assert_eq!(bits(&actual), bits(&expected), "{name}, half n={n}");
                }
                let mut a = src.clone();
                let mut b = src.clone();
                (scalar.half_in_place)(&mut a, w, h);
                (k.half_in_place)(&mut b, w, h);
                assert_eq!(bits(&a), bits(&b), "{name}, in place n={n}");
                let reference: Vec<_> = (0..n).map(|i| [0.21, 0.19, 0.0, 0.31][i % 4]).collect();
                let mean: Vec<_> = reference.iter().map(|v| v + 0.01).collect();
                let square: Vec<_> = mean
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * v + if i % 3 == 0 { 0.0 } else { 0.0001 })
                    .collect();
                let cross: Vec<_> = reference
                    .iter()
                    .zip(&mean)
                    .map(|(a, b)| a * b + 0.00005)
                    .collect();
                let variance: Vec<_> = (0..n)
                    .map(|i| if i % 5 == 0 { 0.0 } else { 0.0001 })
                    .collect();
                let mut a = vec![0.; n];
                let mut b = vec![0.; n];
                (scalar.variance)(&mean, &square, &mut a);
                (k.variance)(&mean, &square, &mut b);
                assert_eq!(bits(&a), bits(&b), "{name}, variance n={n}");
                let rows = [&mean[..], &square, &cross, &reference, &variance];
                assert_eq!(
                    (scalar.pool)(rows, [0.25, 0.75]).map(f32::to_bits),
                    (k.pool)(rows, [0.25, 0.75]).map(f32::to_bits),
                    "{name}, pool n={n}"
                );
                let rows = [
                    &src[..n],
                    &src[1..n + 1],
                    &src[w..w + n],
                    &src[w + 1..w + n + 1],
                ];
                assert_eq!(
                    (scalar.gradients)(rows, [0.25, 0.75, 0.125]).map(f32::to_bits),
                    (k.gradients)(rows, [0.25, 0.75, 0.125]).map(f32::to_bits),
                    "{name}, gradients n={n}"
                );
            }
        }
    }
}
