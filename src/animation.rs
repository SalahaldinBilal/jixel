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

//! Animated codestreams. Each animation frame is encoded as a still whose
//! image header declares the animation and whose displayed frame carries the
//! duration. The stills are then joined under the first one's header.
//!
//! Frame headers are written deep inside every coding path, so the frame being
//! encoded is announced through a thread-local scope instead of a new argument
//! on each of them. Every header write happens on the calling thread.

use crate::bit_writer::BitWriter;
use crate::encode_image::{EncodeConfig, encode_image_with_alpha};
use crate::util::EncodeError;
use crate::yellow_opsin::YellowSelection;
use std::cell::{Cell, RefCell};

/// One animation frame: interleaved 8-bit RGBA and how long it shows.
#[derive(Debug, Clone, Copy)]
pub struct AnimationFrame<'a> {
    pub rgba: &'a [u8],
    /// Display time in ticks of [`AnimationTiming`]. A frame of 0 ticks is not
    /// shown on its own: it becomes a layer of the frame after it. Any other
    /// duration must be at least 1/120 s, the Level 5 limit decoders enforce.
    pub duration: u32,
}

/// Tick rate and loop count of an animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationTiming {
    /// Ticks per second is `tps_numerator / tps_denominator`.
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    /// Times to play the animation. 0 loops forever.
    pub num_loops: u32,
}

impl Default for AnimationTiming {
    /// Millisecond ticks, looping forever.
    fn default() -> Self {
        Self {
            tps_numerator: 1000,
            tps_denominator: 1,
            num_loops: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct FrameScope {
    timing: AnimationTiming,
    duration: u32,
    last: bool,
}

thread_local! {
    static SCOPE: Cell<Option<FrameScope>> = const { Cell::new(None) };
    static YELLOW: Cell<Option<YellowSelection>> = const { Cell::new(None) };
    static HEADERS: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
}

/// Clears the scope when a frame's encode ends, also on error or panic.
struct ScopeGuard;

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE.set(None);
        YELLOW.set(None);
        HEADERS.with_borrow_mut(Vec::clear);
    }
}

pub(crate) fn is_animated() -> bool {
    SCOPE.get().is_some()
}

/// Writes a `U32` field given its four distributions, each a fixed value or
/// `(bits, offset)`.
fn write_u32(w: &mut BitWriter, value: u32, distributions: [Result<u32, (usize, u32)>; 4]) {
    for (selector, distribution) in distributions.into_iter().enumerate() {
        let fits = match distribution {
            Ok(fixed) => fixed == value,
            Err((bits, offset)) => value >= offset && u64::from(value - offset) < (1u64 << bits),
        };
        if fits {
            w.write(2, selector as u64);
            if let Err((bits, offset)) = distribution {
                w.write(bits, u64::from(value - offset));
            }
            return;
        }
    }
    unreachable!("every u32 fits the last distribution");
}

/// The `AnimationHeader` bundle, after `have_animation`.
pub(crate) fn write_animation_header(w: &mut BitWriter) {
    let timing = SCOPE.get().expect("animation scope").timing;
    write_u32(
        w,
        timing.tps_numerator,
        [Ok(100), Ok(1000), Err((10, 1)), Err((30, 1))],
    );
    write_u32(
        w,
        timing.tps_denominator,
        [Ok(1), Ok(1001), Err((8, 1)), Err((10, 1))],
    );
    write_u32(
        w,
        timing.num_loops,
        [Ok(0), Err((3, 0)), Err((16, 0)), Err((32, 0))],
    );
    w.write(1, 0); // have_timecodes = false
}

/// The fields of a regular frame header from `duration` to `save_before_ct`.
/// `last` is whether the frame ends its still, `replace_full` whether it is a
/// full kReplace frame, which signals `save_before_ct` when it can be
/// referenced. Outside an animation this writes what a still needs.
pub(crate) fn write_frame_end(w: &mut BitWriter, last: bool, replace_full: bool) {
    let scope = SCOPE.get();
    let duration = scope.filter(|_| last).map_or(0, |s| s.duration);
    if scope.is_some() {
        write_u32(w, duration, [Ok(0), Ok(1), Err((8, 0)), Err((32, 0))]);
    }
    let is_last = last && scope.is_none_or(|s| s.last);
    w.write(1, u64::from(is_last));
    if !is_last {
        w.write(2, 0); // save_as_reference = 0
        if duration == 0 && replace_full {
            w.write(1, 0); // save_before_color_transform = false
        }
    }
}

/// Records a finished codestream header, `w` being byte aligned.
pub(crate) fn record_header(w: &BitWriter) {
    if is_animated() {
        HEADERS.with_borrow_mut(|headers| headers.push(w.aligned_bytes().to_vec()));
    }
}

/// A frame's yellow-opsin choice with the first frame's XYB matrix, which the
/// image header signals for every frame.
pub(crate) fn pinned_yellow(select: impl FnOnce() -> YellowSelection) -> YellowSelection {
    let selection = select();
    if !is_animated() {
        return selection;
    }
    match YELLOW.get() {
        Some(first) => YellowSelection {
            matrix: first.matrix,
            ..selection
        },
        None => {
            YELLOW.set(Some(selection));
            selection
        }
    }
}

/// Encodes an animation of interleaved 8-bit RGBA frames, each
/// `width * height * 4` bytes, with the same settings as
/// [`encode_image_with_alpha`].
///
/// Every frame is coded on its own, as a full frame. EXIF, XMP and gain maps
/// are not supported.
pub fn encode_animation_with_alpha(
    frames: &[AnimationFrame<'_>],
    width: usize,
    height: usize,
    timing: AnimationTiming,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let mut encoder = AnimationEncoder::new(width, height, timing, config)?;
    for frame in frames {
        encoder.add_frame(frame.rgba, frame.duration)?;
    }
    encoder.finish()
}

/// Encodes an animation one frame at a time, like
/// [`encode_animation_with_alpha`], without holding every frame in memory.
///
/// A frame is encoded when the next one is added or on [`Self::finish`],
/// since only then is it known whether it is the last, so one frame's pixels
/// are kept until then.
pub struct AnimationEncoder {
    width: usize,
    height: usize,
    timing: AnimationTiming,
    config: EncodeConfig,
    pending: Option<(Vec<u8>, u32)>,
    yellow: Option<YellowSelection>,
    header_len: usize,
    out: Vec<u8>,
}

impl AnimationEncoder {
    pub fn new(
        width: usize,
        height: usize,
        timing: AnimationTiming,
        config: &EncodeConfig,
    ) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 {
            return Err(EncodeError::EmptyImage);
        }
        if config.exif.is_some() || config.xmp.is_some() || config.gain_map.is_some() {
            return Err(EncodeError::Unsupported(
                "animations cannot carry EXIF, XMP or a gain map",
            ));
        }
        if timing.tps_numerator == 0 || timing.tps_denominator == 0 {
            return Err(EncodeError::Unsupported(
                "animation tick rate must not be zero",
            ));
        }
        Ok(Self {
            width,
            height,
            timing,
            config: config.clone(),
            pending: None,
            yellow: None,
            header_len: 0,
            out: Vec::new(),
        })
    }

    /// Adds a frame of interleaved 8-bit RGBA shown for `duration` ticks. See
    /// [`AnimationFrame::duration`].
    pub fn add_frame(&mut self, rgba: &[u8], duration: u32) -> Result<(), EncodeError> {
        let expected = self.width * self.height * 4;
        if rgba.len() != expected {
            return Err(EncodeError::InputSizeMismatch {
                expected,
                actual: rgba.len(),
            });
        }
        let ticks_per_second = u128::from(self.timing.tps_numerator);
        if duration > 0
            && u128::from(duration) * u128::from(self.timing.tps_denominator) * 120
                < ticks_per_second
        {
            return Err(EncodeError::Unsupported(
                "a frame is shorter than 1/120 s, faster than Level 5 decoders accept",
            ));
        }
        let mut pixels = match self.pending.take() {
            Some((previous, previous_duration)) => {
                self.encode(&previous, previous_duration, false)?;
                previous
            }
            None => Vec::with_capacity(expected),
        };
        pixels.clear();
        pixels.extend_from_slice(rgba);
        self.pending = Some((pixels, duration));
        Ok(())
    }

    /// Encodes the last frame and returns the file.
    pub fn finish(mut self) -> Result<Vec<u8>, EncodeError> {
        let Some((pixels, duration)) = self.pending.take() else {
            return Err(EncodeError::EmptyImage);
        };
        self.encode(&pixels, duration, true)?;
        Ok(self.out)
    }

    fn encode(&mut self, rgba: &[u8], duration: u32, last: bool) -> Result<(), EncodeError> {
        let _guard = ScopeGuard;
        SCOPE.set(Some(FrameScope {
            timing: self.timing,
            duration,
            last,
        }));
        YELLOW.set(self.yellow);
        HEADERS.with_borrow_mut(Vec::clear);
        let encoded = encode_image_with_alpha(rgba, self.width, self.height, &self.config)?;
        self.yellow = YELLOW.get();
        let header = if self.out.is_empty() {
            HEADERS.with_borrow(|headers| {
                headers
                    .iter()
                    .filter(|h| encoded.starts_with(h))
                    .max_by_key(|h| h.len())
                    .map(Vec::len)
            })
        } else {
            let first = &self.out[..self.header_len];
            HEADERS
                .with_borrow(|headers| headers.iter().any(|h| h == first))
                .then_some(self.header_len)
                .filter(|_| encoded.starts_with(first))
        };
        let Some(header) = header else {
            return Err(EncodeError::Unsupported(
                "animation frames were coded with different image headers",
            ));
        };
        if self.out.is_empty() {
            self.header_len = header;
            self.out = encoded;
        } else {
            self.out.extend_from_slice(&encoded[header..]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Speed;
    use jxl::api::{Event, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat};

    struct Decoded {
        width: usize,
        height: usize,
        animated: bool,
        frames: Vec<(Vec<u8>, f64)>,
    }

    fn decode(bytes: &[u8]) -> Decoded {
        let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
        let mut input = bytes;
        let mut out = Decoded {
            width: 0,
            height: 0,
            animated: false,
            frames: Vec::new(),
        };
        loop {
            match decoder.process(&mut input, None, None).expect("decodes") {
                Event::BasicInfo => {
                    let info = decoder.basic_info().unwrap().clone();
                    (out.width, out.height) = info.size;
                    out.animated = info.animation.is_some();
                    decoder
                        .set_pixel_format(JxlPixelFormat::rgba8(info.extra_channels.len()))
                        .unwrap();
                }
                Event::FrameHeader => {
                    let duration = decoder
                        .frame_header()
                        .and_then(|h| h.duration)
                        .unwrap_or(0.0);
                    let mut pixels = vec![0u8; out.width * out.height * 4];
                    loop {
                        let mut buffers =
                            [JxlOutputBuffer::new(&mut pixels, out.height, out.width * 4)];
                        match decoder
                            .process(&mut input, Some(&mut buffers), None)
                            .unwrap()
                        {
                            Event::FrameComplete { .. } => break,
                            Event::NeedMoreInput { .. } => panic!("truncated frame"),
                            _ => {}
                        }
                    }
                    out.frames.push((pixels, duration));
                }
                Event::NeedMoreInput { .. } => assert!(!input.is_empty(), "truncated stream"),
                Event::FrameComplete { .. } => {}
                Event::Complete => return out,
            }
        }
    }

    /// Deterministic test content: a gradient with noise and a moving square,
    /// or flat UI-like blocks with small repeated marks when `flat`.
    fn frame(width: usize, height: usize, index: usize, flat: bool, alpha: bool) -> Vec<u8> {
        let mut seed = 0x9E37_79B9u32.wrapping_mul(index as u32 + 1);
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed & 15) as u8
        };
        let mut rgba = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                let in_square = (x + index * 9) % width < 40 && (y + index * 5) % height < 40;
                let px = if flat {
                    let mark = (x / 4 + y / 6) % 7 == 0 && (y % 6) < 4;
                    match (in_square, mark) {
                        (true, _) => [230, 200, 20],
                        (_, true) => [20, 20, 30],
                        _ => [245, 245, 248],
                    }
                } else if in_square {
                    [250, 220, 30]
                } else {
                    [
                        (x * 255 / width) as u8 ^ noise(),
                        (y * 255 / height) as u8 ^ noise(),
                        ((x + y + index * 20) % 256) as u8,
                    ]
                };
                let a = if alpha && index % 2 == 1 {
                    ((x * 3 + y) % 256) as u8
                } else {
                    255
                };
                rgba.extend_from_slice(&[px[0], px[1], px[2], a]);
            }
        }
        rgba
    }

    fn psnr(source: &[u8], output: &[u8]) -> f64 {
        let squared: f64 = source
            .iter()
            .zip(output)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        10.0 * (255.0 * 255.0 * source.len() as f64 / squared.max(1e-9)).log10()
    }

    fn round_trip(width: usize, height: usize, flat: bool, alpha: bool, config: &EncodeConfig) {
        let durations = [40u32, 100, 9, 250];
        let pixels: Vec<_> = (0..durations.len())
            .map(|i| frame(width, height, i, flat, alpha))
            .collect();
        let frames: Vec<_> = pixels
            .iter()
            .zip(durations)
            .map(|(rgba, duration)| AnimationFrame { rgba, duration })
            .collect();
        let bytes =
            encode_animation_with_alpha(&frames, width, height, AnimationTiming::default(), config)
                .expect("encodes");
        let decoded = decode(&bytes);
        assert!(decoded.animated);
        assert_eq!((decoded.width, decoded.height), (width, height));
        assert_eq!(decoded.frames.len(), durations.len());
        for (index, ((source, (output, duration)), expected)) in pixels
            .iter()
            .zip(&decoded.frames)
            .zip(durations)
            .enumerate()
        {
            assert_eq!(*duration, f64::from(expected));
            if config.lossless {
                assert!(
                    source == output,
                    "lossless frame {index} differs from its source"
                );
            } else {
                // Frames match their stills. Efforts that pick a yellow-opsin
                // matrix use the first frame's, so a frame that would pick
                // another one on its own only has to keep a sane quality.
                let still = encode_image_with_alpha(source, width, height, config).unwrap();
                let still = decode(&still).frames.remove(0).0;
                let (frame_psnr, still_psnr) = (psnr(source, output), psnr(source, &still));
                let pinned = config.speed.effort().yellow_opsin;
                assert!(
                    still == *output || (pinned && frame_psnr > 30.0),
                    "{width}x{height} flat {flat} alpha {alpha} {:?}: frame {index} at \
                     {frame_psnr:.2} dB, its still at {still_psnr:.2} dB",
                    config.speed
                );
            }
        }
    }

    #[test]
    fn lossless_round_trips_exactly() {
        let config = EncodeConfig::default().with_lossless(true);
        round_trip(96, 80, false, true, &config);
        round_trip(96, 80, true, false, &config);
    }

    #[test]
    fn lossless_flat_content_multi_group() {
        for speed in [Speed::Fast, Speed::Slow] {
            let config = EncodeConfig::default()
                .with_lossless(true)
                .with_speed(speed)
                .with_num_threads(4);
            round_trip(300, 280, true, false, &config);
        }
    }

    #[test]
    fn lossy_frames_match_their_stills() {
        for speed in [Speed::Fastest, Speed::Fast, Speed::Slow] {
            let config = EncodeConfig::default()
                .with_distance(1.0)
                .with_speed(speed)
                .with_num_threads(4);
            // Slow only at a single-group size, to keep debug test runs short.
            if speed != Speed::Slow {
                round_trip(300, 280, false, false, &config);
                round_trip(300, 280, true, false, &config);
            }
            round_trip(96, 80, false, true, &config);
            round_trip(96, 80, true, false, &config);
        }
    }

    #[test]
    fn single_frame_is_an_animation() {
        let rgba = frame(64, 48, 0, false, false);
        let frames = [AnimationFrame {
            rgba: &rgba,
            duration: 500,
        }];
        let bytes = encode_animation_with_alpha(
            &frames,
            64,
            48,
            AnimationTiming::default(),
            &EncodeConfig::default(),
        )
        .unwrap();
        let decoded = decode(&bytes);
        assert!(decoded.animated);
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(decoded.frames[0].1, 500.0);
    }

    #[test]
    fn fractional_tick_rates_keep_exact_durations() {
        let pixels: Vec<_> = (0..3).map(|i| frame(32, 24, i, true, false)).collect();
        let config = EncodeConfig::default().with_lossless(true);
        let cases: [(AnimationTiming, [u32; 3], [f64; 3]); 3] = [
            (
                AnimationTiming {
                    tps_numerator: 3000,
                    tps_denominator: 1,
                    num_loops: 0,
                },
                [40, 41, 3000],
                [40.0 / 3.0, 41.0 / 3.0, 1000.0],
            ),
            (
                AnimationTiming {
                    tps_numerator: 30000,
                    tps_denominator: 1001,
                    num_loops: 2,
                },
                [1, 2, 30],
                [1001.0 / 30.0, 2002.0 / 30.0, 1001.0],
            ),
            (
                AnimationTiming {
                    tps_numerator: 1_000_000,
                    tps_denominator: 1,
                    num_loops: 0,
                },
                [13_300, 16_667, 9_000],
                [13.3, 16.667, 9.0],
            ),
        ];
        for (timing, durations, expected) in cases {
            let frames: Vec<_> = pixels
                .iter()
                .zip(durations)
                .map(|(rgba, duration)| AnimationFrame { rgba, duration })
                .collect();
            let bytes = encode_animation_with_alpha(&frames, 32, 24, timing, &config).unwrap();
            let decoded = decode(&bytes);
            for ((_, duration), expected) in decoded.frames.iter().zip(expected) {
                assert!(
                    (duration - expected).abs() < 1e-9,
                    "{duration} != {expected}"
                );
            }
        }
    }

    #[test]
    fn zero_duration_frames_merge_into_the_next() {
        let pixels: Vec<_> = (0..3).map(|i| frame(64, 48, i, true, false)).collect();
        let frames: Vec<_> = pixels
            .iter()
            .zip([40, 0, 100])
            .map(|(rgba, duration)| AnimationFrame { rgba, duration })
            .collect();
        let config = EncodeConfig::default().with_lossless(true);
        let bytes =
            encode_animation_with_alpha(&frames, 64, 48, AnimationTiming::default(), &config)
                .unwrap();
        let decoded = decode(&bytes);
        assert_eq!(decoded.frames.len(), 2);
        assert_eq!(decoded.frames[1].0, pixels[2]);
        assert_eq!(decoded.frames[1].1, 100.0);
    }

    #[test]
    fn stills_are_unchanged_after_an_animation() {
        let rgba = frame(64, 48, 0, false, true);
        let config = EncodeConfig::default();
        let before = encode_image_with_alpha(&rgba, 64, 48, &config).unwrap();
        let frames = [AnimationFrame {
            rgba: &rgba,
            duration: 10,
        }];
        encode_animation_with_alpha(&frames, 64, 48, AnimationTiming::default(), &config).unwrap();
        let after = encode_image_with_alpha(&rgba, 64, 48, &config).unwrap();
        assert_eq!(before, after);
        assert!(!decode(&after).animated);
    }

    #[test]
    fn rejects_what_it_cannot_carry() {
        let rgba = frame(16, 16, 0, false, false);
        let frames = [AnimationFrame {
            rgba: &rgba,
            duration: 10,
        }];
        let timing = AnimationTiming::default();
        let exif = EncodeConfig::default().with_exif(vec![0; 8]);
        assert!(matches!(
            encode_animation_with_alpha(&frames, 16, 16, timing, &exif),
            Err(EncodeError::Unsupported(_))
        ));
        assert_eq!(
            encode_animation_with_alpha(&[], 16, 16, timing, &EncodeConfig::default()),
            Err(EncodeError::EmptyImage)
        );
        let mut encoder = AnimationEncoder::new(16, 16, timing, &EncodeConfig::default()).unwrap();
        assert!(matches!(
            encoder.add_frame(&rgba[4..], 10),
            Err(EncodeError::InputSizeMismatch { .. })
        ));
        let too_fast = [AnimationFrame {
            rgba: &rgba,
            duration: 8,
        }];
        assert!(matches!(
            encode_animation_with_alpha(&too_fast, 16, 16, timing, &EncodeConfig::default()),
            Err(EncodeError::Unsupported(_))
        ));
    }
}
