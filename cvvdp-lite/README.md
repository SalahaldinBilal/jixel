# cvvdp-lite

A dependency-free Rust implementation of the **still-image** ColorVideoVDP
0.5.7 base model, intended for repeated encoder candidate scoring. It is a
separate workspace crate so the metric can evolve independently and be used
by a future Jixel arbiter. The encoder does not call it yet.

"Lite" describes the scope and execution: three sustained color channels,
SDR sRGB, CPU, and scores plus per-band diagnostics. The calibrated spatial
model is preserved. There are no temporal filters, video buffers, neural
regressors, image decoders, GPU dependencies, HDR transfer functions, or heatmaps.

## API

```rust
use cvvdp_lite::{CvvdpLite, Display, RgbImage};

# fn example(reference_rgb: &[u8], test_rgb: &[u8], width: usize, height: usize)
#     -> Result<(), cvvdp_lite::Error> {
let metric = CvvdpLite::new(Display::standard_4k())?;
let reference = metric.prepare_reference(
    RgbImage::srgb8(reference_rgb, width, height, width * 3)?
)?;
let mut evaluator = reference.evaluator();
let score = evaluator.compare(
    RgbImage::srgb8(test_rgb, width, height, width * 3)?
)?;
println!("{} JOD; {} distortion", score.jod, score.distortion);
let bands = evaluator.band_scores();
# let _ = bands;
# Ok(())
# }
```

Prepare the reference once per image/display, then reuse the evaluator for
each reconstruction. Reference contrast and reference-derived CSF sensitivity
are cached; test pyramids stream one scale at a time. Candidate scratch
allocations are reserved before evaluation, including the first candidate.
Buffers keep their original orientation after odd pyramid depths, and scale
processing no longer repeatedly grows and zeroes vectors. Masking blurs write
back into their activity buffers, saving three full-resolution float planes. The reference is immutable and `Send + Sync`; use
independent evaluators on the encoder's existing worker pool when needed.

For a long-lived arbiter worker, keep `Workspace` independent of reference
lifetimes and reuse it for reference preparation too:

```rust
use cvvdp_lite::{CvvdpLite, Display, RgbImage, Workspace};

# fn worker(reference_rgb: &[u8], test_rgb: &[u8], width: usize, height: usize)
#     -> Result<(), cvvdp_lite::Error> {
let metric = CvvdpLite::new(Display::standard_4k())?;
let mut workspace = Workspace::new();
workspace.reserve(width, height)?;
let reference = metric.prepare_reference_with(
    RgbImage::srgb8(reference_rgb, width, height, width * 3)?,
    &mut workspace,
)?;
let score = workspace.compare(
    &reference,
    RgbImage::srgb8(test_rgb, width, height, width * 3)?,
)?;
# let _ = score;
# Ok(())
# }
```

The same workspace can compare another reference at equal or smaller reserved
width/height without allocating. Reference creation allocates its owned cache;
its temporary analysis buffers reuse the workspace. `Evaluator::into_workspace`
and `PreparedReference::evaluator_with` support moving existing evaluator state
between references. Both state and reference expose `allocated_bytes()` for
worker memory budgeting; storage grows as needed and does not shrink. A fresh
768×512 worker retains approximately 16.1 MiB of scratch plus a shared 12.0 MiB
reference cache. Use one workspace per concurrent worker and share references.

`RgbImage` accepts strided interleaved sRGB `u8`, interleaved sRGB `f32`, or
three strided **relative linear sRGB** `f32` planes. Linear 1 means SDR display
white, not one nit. Finite floats are clipped to [0,1]; NaN/infinity are rejected.
Composite alpha and convert ICC profiles to sRGB before calling. No rescaling
to the display resolution is performed. `Display::standard_4k()` matches the
existing Python benchmark default; `standard_fhd()` and custom display
photometry/angular resolution are also available.

`Score::jod` is higher for better quality, with 10 for identical images.
Negative scores are valid and are not clamped. `Score::distortion` is the
original's pooled value before JOD mapping, with smaller values better.
Band diagnostics expose regularized RMS for luminance, red/green, and
yellow/violet, before baseband weights. They are not an additive pixel loss
map. Full-image context is required: independently scoring cropped blocks
changes adaptation, masking, and the pyramid, so it is not equivalent to
scoring their contribution to a full frame.

## Reference fidelity and corrections

The original [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP) is the
model authority. [fcvvdp](https://github.com/halidecx/fcvvdp) was inspected for
comparison and performance ideas. The original revision is `0c1a0e093bd0f9b7c8f4e7e502f11a4d3fa7183c`;
the fcvvdp revision is `98bd21a7a876be29ba06a29325b03624047ee850`.

Project source uses the same BSD-3-Clause / Apache-2.0 licensing as Jixel.
The original MIT copyright and permission notice for the sustained CSF data
are retained inline in `src/tables.rs` and preserved by its generator.

The implementation preserves DKLd65, independent test/reference `weber_g1`
adaptation, sustained CASTLE CSF tables, the 1.45 red/green gain, mutual and
cross-channel masking, the sigma-3 13-tap phase uncertainty blur, the small
band blur cutoff, baseband CSF at 0.1 cycles/degree, baseband weights, image
integration, **epsilon-regularized pooling**, and the piecewise JOD mapping.
The pooling epsilon prevents collapsing the two fourth-norm reductions into
one; doing so measurably changes scores near perfect quality.

One original edge defect is deliberately corrected:
`lpyr_dec.gausspyr_reduce` checks height parity when padding the right edge.
This crate uses width parity and symmetric edge samples in each axis. The
validation harness records both unchanged and corrected upstream scores,
and applies only that one-line correction in memory. For a synthetic 63×65
image compared with black, the pinned upstream predicts 2.838480 JOD before
the fix and 2.288393 after it; this crate agrees with the corrected result.
Transpose symmetry has a regression test.

Images whose smaller dimension is below four use a single baseband rather
than failing the original pyramid level check. Thin and tiny inputs remain
finite, but this extension is not separately perceptually calibrated. A
baseband-only image has no spatial luminance contrast, and normalization by
its mean can cancel a uniform neutral luminance change; avoid using tiny
crops as an arbiter's sole evidence.

The inspected fcvvdp revision differs from the original in pyramid band
count, baseband frequency, phase uncertainty blur size/padding, rounded color
coefficients, and omission of the pooling epsilon. This implementation
follows the original model and tests those stages through end-to-end fixtures.
It does not claim improved perceptual calibration over ColorVideoVDP.

## SIMD and numerical behavior

The default `simd` feature selects NEON on AArch64, AVX2 on supported x86 CPUs,
and SSE2 on older x86 CPUs. CPU features are detected and cached. Other targets
use the scalar fallback. `Workspace::with_backend(Backend::Scalar)` explicitly
selects libm masking powers; `--no-default-features` compiles out all intrinsics.
Unsupported explicit backend requests return an error before entering CPU
instructions. The existing caller's worker pool owns concurrency.

SIMD accelerates contrast weighting, the separable Gaussian masking blur, and
masked-difference pooling. NEON/SSE2 process four pixels and AVX2 eight. Kernels
use exact vector division, shared model equations, scalar tails, and f64
reduction accumulators. Unsafe code is confined to the private kernel module,
whose safe dispatch validates lengths and CPU availability; other library
modules deny unsafe code.

Masking powers decompose a positive float into a centered mantissa and integer
exponent. A degree-ten binomial polynomial evaluates the mantissa power, and
shared immutable tables supply exponent scales. Tables occupy about 4 KiB
for all workers and are initialized before scoring. A sweep of 40,012 powers
for the four calibrated exponents, from roughly 1e-5 to 1e7, checks relative
error below 0.000002 against libm on available backends. Exact zeros are handled
explicitly. This is an approximation to the same calibrated model; it is not
bit-for-bit scalar arithmetic. Calibration constants and epsilon pooling are
unchanged.

## Validation and performance

```sh
cargo test -p cvvdp-lite --offline
cargo test -p cvvdp-lite --no-default-features --offline
cargo clippy -p cvvdp-lite --all-targets --offline -- -D warnings
cargo build -p cvvdp-lite --release --example score_raw --offline
```

Regression tests include 39 pinned upstream predictions, zero-allocation
scoring on the first/repeated candidate and across smaller references, backend
agreement for padded tails/display extremes, invalid input recovery, tiny
images, and transpose symmetry. Native NEON tests and AddressSanitizer pass;
SSE2 tests execute under Rosetta; this Rosetta runtime does not expose AVX2,
so AVX2 is build/lint checked and awaits native x86 runtime validation.
Strict Clippy also passes for the native,
x86, and scalar-only configurations. The WebAssembly fallback compiles.

`validation.json` records 51 comparisons against the corrected original,
including three distortions on each of four Kodak images. The maximum error
is **0.000001233 JOD**. Its single-thread 768×512 mildly noisy Kodak case
measured 10.96 ms per cached NEON candidate, 50.34 ms for fcvvdp with a reused
context, and 270.24 ms for original PyTorch CPU. Rust reference preparation
took 14.45 ms; reserving candidate state took 0.43 ms. These are local smoke
measurements, not a broad perceptual ranking or native x86 performance study.

`optimization-benchmark.json` compares the implementation before this
optimization pass, the current scalar backend, and NEON on the same inputs.
Three trials alternate execution order; each trial averages seven cached
comparisons. Image decoding and file I/O are outside the timed operations.

| Resolution | Before, ms | Current scalar, ms | NEON, ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 64×64 | 0.392 | 0.241 | 0.115 | 3.40× |
| 767×511 | 40.06 | 26.21 | 10.43 | 3.84× |
| 768×512 | 39.93 | 26.38 | 10.40 | 3.84× |
| 1920×1080 | 231.25 | 151.69 | 60.79 | 3.80× |
| 3840×2160 | 1056.86 | 639.08 | 329.00 | 3.21× |

The resized Kodak inputs use deterministic RGB noise; numerical agreement is
also checked separately on stronger distortions. The baseline binary's hash
and complete trial outputs are retained in the report.

Reproduce the benchmark with a pre-optimization `score_raw` binary if available:

```sh
python tools/benchmark.py --binary target/release/examples/score_raw \
  --baseline /path/to/previous/score_raw --image /path/to/kodak/01.png \
  --output /tmp/cvvdp-benchmark.json
```

The raw example additionally accepts `[auto|scalar|neon|avx2|sse2]` after the
display argument and reports reference/scratch bytes and reservation timing.
To regenerate upstream comparisons (requires Python validation dependencies):

```sh
python tools/validate.py --upstream /path/to/ColorVideoVDP \
  --binary target/release/examples/score_raw --output /tmp/cvvdp-validation.json \
  --image /path/to/kodak/01.png \
  --validation-images /path/to/kodak/05.png /path/to/kodak/13.png
```

Pass `--fcvvdp-library /path/to/libfcvvdp.dylib` (or `.so`) to benchmark fcvvdp
too. `tools/generate_tables.py` regenerates the sustained CSF data; run
`cargo fmt -p cvvdp-lite` afterwards. Python/Pillow are validation tools only;
the Rust crate still has zero runtime dependencies.

Future arbiter integration should use full-frame candidates and the raw
distortion score. A full-frame attribution map is still needed for local
quantization decisions; independent cropped-block scores are not equivalent.
Further optimization can target pyramid expansion and direct linear-plane
conversion, followed by held-out ranking checks on actual Jixel reconstructions.
