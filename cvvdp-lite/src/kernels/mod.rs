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

//! Private SIMD boundary. Safe entry points validate slice lengths; kernels
//! only touch complete vectors, and scalar tails handle every remaining lane.
//! Runtime CPU detection precedes every target-feature entry point.

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon;
#[cfg(all(
    feature = "simd",
    any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")
))]
mod vector;
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
mod x86;

/// Arithmetic backend. Scalar uses libm powers; SIMD uses a bounded polynomial
/// approximation to powers in the masking stage. Calibration/pooling are shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Neon,
    Avx2,
    Sse2,
}

impl Backend {
    /// Select the best compiled backend supported by the current CPU.
    pub fn detect() -> Self {
        static DETECTED: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
        *DETECTED.get_or_init(|| {
            for backend in [Self::Avx2, Self::Neon, Self::Sse2] {
                if backend.is_available() {
                    return backend;
                }
            }
            Self::Scalar
        })
    }

    pub fn is_available(self) -> bool {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if self == Self::Neon {
            return std::arch::is_aarch64_feature_detected!("neon");
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
        match self {
            Self::Avx2 => return std::arch::is_x86_feature_detected!("avx2"),
            Self::Sse2 => return std::arch::is_x86_feature_detected!("sse2"),
            _ => {}
        }
        self == Self::Scalar
    }
}

pub(crate) const EPSILON: f32 = 0.00001;
pub(crate) const MASK_P: f32 = 2.2643552;
pub(crate) const MASK_Q: [f32; 3] = [1.3026227, 2.8885908, 3.6807714];

#[derive(Clone, Copy)]
pub(crate) struct MaskParameters {
    pub mask_c: f32,
    pub dmax: f32,
    pub mask_zero: [f32; 3],
    pub diff_zero: f32,
    pub xcm: [[f32; 3]; 3],
}

pub(crate) struct Contrast<'a> {
    current: [&'a mut [f32]; 3],
    expanded: [&'a [f32]; 3],
    sensitivity: [&'a [f32]; 3],
    reference: [&'a [f32]; 3],
    mask: [&'a mut [f32]; 3],
    scale: f32,
}

pub(crate) struct Pool<'a> {
    current: [&'a [f32]; 3],
    reference: [&'a [f32]; 3],
    blurred: [&'a [f32]; 3],
    parameters: MaskParameters,
}

#[allow(unused_mut, unused_unsafe)] // Scalar-only builds have no dispatch block.
pub(crate) fn contrast(
    backend: Backend,
    current: [&mut [f32]; 3],
    expanded: [&[f32]; 3],
    sensitivity: [&[f32]; 3],
    reference: [&[f32]; 3],
    mask: [&mut [f32]; 3],
    scale: f32,
) {
    assert!(backend.is_available());
    let count = current[0].len();
    assert!(current.iter().all(|p| p.len() == count));
    assert!(
        expanded
            .iter()
            .chain(&sensitivity)
            .chain(&reference)
            .all(|p| p.len() == count)
    );
    assert!(mask.iter().all(|p| p.len() == count));
    let mut data = Contrast {
        current,
        expanded,
        sensitivity,
        reference,
        mask,
        scale,
    };
    let mut done = 0;
    // SAFETY: Equal initialized slice lengths were checked above. Workspace
    // construction validates backend availability; the dispatch checks it again
    // by architecture, and only a supported target-feature entry is reachable.
    unsafe {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            done = neon::contrast(&mut data);
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
        match backend {
            Backend::Avx2 => done = x86::avx2_contrast(&mut data),
            Backend::Sse2 => done = x86::sse2_contrast(&mut data),
            _ => {}
        }
    }
    let _ = backend;
    for i in done..count {
        let bg = data.expanded[0][i].max(0.01);
        for c in 0..3 {
            let weighted = ((data.current[c][i] - data.expanded[c][i]) / bg).min(1000.0)
                * data.scale
                * data.sensitivity[c][i];
            data.current[c][i] = weighted;
            data.mask[c][i] = weighted.abs().min(data.reference[c][i].abs());
        }
    }
}

#[allow(unused_mut, unused_unsafe)] // Scalar-only builds have no dispatch block.
pub(crate) fn pool(
    backend: Backend,
    current: [&[f32]; 3],
    reference: [&[f32]; 3],
    blurred: [&[f32]; 3],
    parameters: MaskParameters,
) -> [f64; 3] {
    assert!(backend.is_available());
    let count = current[0].len();
    assert!(
        current
            .iter()
            .chain(&reference)
            .chain(&blurred)
            .all(|p| p.len() == count)
    );
    let data = Pool {
        current,
        reference,
        blurred,
        parameters,
    };
    let parameters = data.parameters;
    let (mut done, mut squares) = (0, [0.0; 3]);
    // SAFETY: All input planes have the checked initialized count, and backend
    // is CPU-validated by Workspace. SIMD loops only consume complete vectors.
    unsafe {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            (done, squares) = neon::pool(&data);
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
        match backend {
            Backend::Avx2 => (done, squares) = x86::avx2_pool(&data),
            Backend::Sse2 => (done, squares) = x86::sse2_pool(&data),
            _ => {}
        }
    }
    let _ = backend;
    for i in done..count {
        let activity: [f32; 3] = std::array::from_fn(|c| {
            (data.blurred[c][i] * parameters.mask_c + EPSILON).powf(MASK_Q[c])
                - parameters.mask_zero[c]
        });
        for (c, sum) in squares.iter_mut().enumerate() {
            let mask = activity[0] * parameters.xcm[0][c]
                + activity[1] * parameters.xcm[1][c]
                + activity[2] * parameters.xcm[2][c];
            let diff = (data.current[c][i] - data.reference[c][i]).abs();
            let du = ((diff + EPSILON).powf(MASK_P) - parameters.diff_zero) / (1.0 + mask);
            let d = (parameters.dmax * du / (parameters.dmax + du)) as f64;
            *sum += d * d + 2.0 * (EPSILON as f64) * d;
        }
    }
    squares
}

#[allow(unused_mut, unused_unsafe)] // Scalar-only builds have no dispatch block.
pub(crate) fn blur(backend: Backend, rows: [&[f32]; 13], dst: &mut [f32], kernel: &[f32; 13]) {
    assert!(backend.is_available());
    assert!(rows.iter().all(|r| r.len() == dst.len()));
    let mut done = 0;
    // SAFETY: Equal initialized row lengths were checked. The destination is
    // exclusively borrowed, and SIMD accesses only whole in-bounds vectors.
    unsafe {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            done = neon::blur(rows, dst, kernel);
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
        match backend {
            Backend::Avx2 => done = x86::avx2_blur(rows, dst, kernel),
            Backend::Sse2 => done = x86::sse2_blur(rows, dst, kernel),
            _ => {}
        }
    }
    let _ = backend;
    for i in done..dst.len() {
        let mut sum = rows[6][i] * kernel[6];
        for k in 1..=6 {
            sum += (rows[6 - k][i] + rows[6 + k][i]) * kernel[6 - k];
        }
        dst[i] = sum;
    }
}

pub(crate) fn warmup(backend: Backend) {
    #[cfg(all(
        feature = "simd",
        any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")
    ))]
    if backend != Backend::Scalar {
        let _ = vector::power_tables();
    }
    let _ = backend;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(unused_mut, unused_unsafe)]
    fn powers(backend: Backend, input: &[f32], output: &mut [f32], exponent: f32) {
        assert!(backend.is_available());
        assert_eq!(input.len(), output.len());
        assert!(input.iter().all(|x| x.is_normal() && *x > 0.0));
        let mut done = 0;
        // SAFETY: Feature availability, normal positive inputs, equal lengths,
        // and normal result range are established by this test's input sweep.
        unsafe {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            if backend == Backend::Neon {
                done = neon::powers(input, output, exponent);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            match backend {
                Backend::Avx2 => done = x86::avx2_powers(input, output, exponent),
                Backend::Sse2 => done = x86::sse2_powers(input, output, exponent),
                _ => {}
            }
        }
        for i in done..input.len() {
            output[i] = input[i].powf(exponent);
        }
    }

    #[test]
    fn vector_powers_match_libm_across_masking_range_and_tails() {
        let input: Vec<f32> = (0..10003)
            .map(|i| 2.0_f32.powf(-16.6 + i as f32 * 40.0 / 10002.0))
            .collect();
        for backend in [Backend::Scalar, Backend::Neon, Backend::Avx2, Backend::Sse2] {
            if !backend.is_available() {
                continue;
            }
            for exponent in [MASK_P, MASK_Q[0], MASK_Q[1], MASK_Q[2]] {
                let mut output = vec![0.0; input.len()];
                powers(backend, &input, &mut output, exponent);
                for (&x, &got) in input.iter().zip(&output) {
                    let expected = x.powf(exponent);
                    let relative = (got / expected - 1.0).abs();
                    assert!(
                        relative < 0.000002,
                        "{backend:?}: {x}^{exponent} {got} != {expected}, error={relative}"
                    );
                }
            }
        }
    }
}
