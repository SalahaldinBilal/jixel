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

use crate::Error;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
enum Pixels<'a> {
    Srgb8(&'a [u8]),
    SrgbF32(&'a [f32]),
    Linear([&'a [f32]; 3]),
}

/// A borrowed SDR image, without alpha. Composite alpha and convert ICC color
/// profiles to sRGB before constructing this view. Strides are in samples.
#[derive(Clone, Copy)]
pub struct RgbImage<'a> {
    pixels: Pixels<'a>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    stride: usize,
}

impl<'a> RgbImage<'a> {
    /// Interleaved, gamma-encoded sRGB bytes. Row padding is allowed.
    pub fn srgb8(
        pixels: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 3, &[pixels.len()])?;
        Ok(Self {
            pixels: Pixels::Srgb8(pixels),
            width,
            height,
            stride,
        })
    }

    /// Interleaved, gamma-encoded sRGB floats. Finite samples outside `[0,1]`
    /// are clipped by the display model; non-finite samples fail evaluation.
    pub fn srgb_f32(
        pixels: &'a [f32],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 3, &[pixels.len()])?;
        Ok(Self {
            pixels: Pixels::SrgbF32(pixels),
            width,
            height,
            stride,
        })
    }

    /// Three linear sRGB planes, normalized to `[0,1]` (1 is display white).
    /// This is relative SDR light, not absolute HDR luminance. Finite samples
    /// outside `[0,1]` are clipped. The three planes share a stride.
    pub fn linear_rgb(
        planes: [&'a [f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 1, &planes.map(|p| p.len()))?;
        Ok(Self {
            pixels: Pixels::Linear(planes),
            width,
            height,
            stride,
        })
    }

    /// Image width, in pixels.
    pub fn width(self) -> usize {
        self.width
    }
    /// Image height, in pixels.
    pub fn height(self) -> usize {
        self.height
    }

    pub(crate) fn to_dkl(
        self,
        scale: f32,
        black: f32,
        out: &mut [Vec<f32>; 3],
    ) -> Result<(), Error> {
        for plane in out.iter_mut() {
            crate::pyramid::ensure_len(plane, self.width * self.height);
        }
        // The fused RGB -> XYZ -> LMS2006 -> DKLd65 matrix, computed using the
        // original's f32 matrices (not fcvvdp's rounded XYZ coefficients).
        const DKL: [[f32; 3]; 3] = [
            [0.23320122, 0.72883075, 0.088_995_87],
            [0.12762076, -0.08706811, -0.036777392],
            [-0.2148225, -0.62625366, 0.8514033],
        ];
        let table = srgb_table();
        for y in 0..self.height {
            for x in 0..self.width {
                let mut rgb = [0.0; 3];
                match self.pixels {
                    Pixels::Srgb8(p) => {
                        let offset = y * self.stride + x * 3;
                        for c in 0..3 {
                            rgb[c] = table[p[offset + c] as usize];
                        }
                    }
                    Pixels::SrgbF32(p) => {
                        let offset = y * self.stride + x * 3;
                        for c in 0..3 {
                            let v = p[offset + c];
                            if !v.is_finite() {
                                return Err(Error::NonFiniteSample);
                            }
                            rgb[c] = srgb_to_linear(v.clamp(0.0, 1.0));
                        }
                    }
                    Pixels::Linear(p) => {
                        for c in 0..3 {
                            let v = p[c][y * self.stride + x];
                            if !v.is_finite() {
                                return Err(Error::NonFiniteSample);
                            }
                            rgb[c] = v.clamp(0.0, 1.0);
                        }
                    }
                }
                for v in &mut rgb {
                    *v = *v * scale + black;
                }
                for c in 0..3 {
                    out[c][y * self.width + x] =
                        rgb[0] * DKL[c][0] + rgb[1] * DKL[c][1] + rgb[2] * DKL[c][2];
                }
            }
        }
        Ok(())
    }
}

fn validate(
    width: usize,
    height: usize,
    stride: usize,
    channels: usize,
    lengths: &[usize],
) -> Result<(), Error> {
    pixel_count(width, height)?;
    let row = width.checked_mul(channels).ok_or(Error::InvalidLayout)?;
    let needed = (height - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(row))
        .ok_or(Error::InvalidLayout)?;
    if stride < row || lengths.iter().any(|&n| n < needed) {
        return Err(Error::InvalidLayout);
    }
    Ok(())
}

fn srgb_to_linear(v: f32) -> f32 {
    if v > 0.04045 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

fn srgb_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

pub(crate) fn pixel_count(width: usize, height: usize) -> Result<usize, Error> {
    if width == 0
        || height == 0
        || width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(12))
            .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(Error::InvalidDimensions);
    }
    Ok(width * height)
}
