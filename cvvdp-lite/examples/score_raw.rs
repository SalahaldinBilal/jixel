//! Minimal byte-RGB harness for upstream differential tests and benchmarks.
use cvvdp_lite::{Backend, CvvdpLite, Display, RgbImage, Workspace};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 || args.len() > 8 {
        return Err("usage: score_raw WIDTH HEIGHT REF.rgb TEST.rgb [ITERATIONS] [4k|fhd] [auto|scalar|neon|avx2|sse2]".into());
    }
    let w: usize = args[1].parse()?;
    let h: usize = args[2].parse()?;
    let reference = std::fs::read(&args[3])?;
    let test = std::fs::read(&args[4])?;
    let iterations: u32 = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(1);
    if iterations == 0 {
        return Err("iterations must be nonzero".into());
    }
    let display = match args.get(6).map(String::as_str).unwrap_or("4k") {
        "4k" => Display::standard_4k(),
        "fhd" => Display::standard_fhd(),
        _ => return Err("display must be 4k or fhd".into()),
    };
    let started = Instant::now();
    let reference =
        CvvdpLite::new(display)?.prepare_reference(RgbImage::srgb8(&reference, w, h, w * 3)?)?;
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    let backend = match args.get(7).map(String::as_str).unwrap_or("auto") {
        "auto" => Backend::detect(),
        "scalar" => Backend::Scalar,
        "neon" => Backend::Neon,
        "avx2" => Backend::Avx2,
        "sse2" => Backend::Sse2,
        _ => return Err("backend must be auto, scalar, neon, avx2, or sse2".into()),
    };
    let reserve_started = Instant::now();
    let mut evaluator = reference.evaluator_with(Workspace::with_backend(backend)?);
    let reserve_ms = reserve_started.elapsed().as_secs_f64() * 1000.0;
    let test = RgbImage::srgb8(&test, w, h, w * 3)?;
    let first_started = Instant::now();
    let mut score = evaluator.compare(test)?;
    let first_ms = first_started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    for _ in 0..iterations {
        score = std::hint::black_box(evaluator.compare(std::hint::black_box(test))?);
    }
    let candidate_ms = started.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "{{\"jod\":{},\"distortion\":{},\"prepare_ms\":{},\"first_ms\":{},\"candidate_ms\":{},\"bands\":{},\"backend\":\"{:?}\",\"reserve_ms\":{},\"workspace_bytes\":{},\"reference_bytes\":{}}}",
        score.jod,
        score.distortion,
        prepare_ms,
        first_ms,
        candidate_ms,
        reference.band_count(),
        evaluator.backend(),
        reserve_ms,
        evaluator.allocated_bytes(),
        reference.allocated_bytes()
    );
    Ok(())
}
