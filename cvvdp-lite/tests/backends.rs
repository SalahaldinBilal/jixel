use cvvdp_lite::{Backend, CvvdpLite, Display, Error, RgbImage, Workspace};

#[test]
fn all_available_backends_agree_for_padded_tails_and_display_extremes() {
    let displays = [
        Display::default(),
        Display::standard_fhd(),
        Display {
            pixels_per_degree: 0.01,
            peak_luminance: 10000.0,
            black_luminance: 0.0,
            reflected_luminance: 0.0,
        },
        Display {
            pixels_per_degree: 10000.0,
            peak_luminance: 1.0,
            black_luminance: 0.0,
            reflected_luminance: 0.0,
        },
    ];
    for display in displays {
        let metric = CvvdpLite::new(display).unwrap();
        for (w, h) in [(1, 1), (7, 8), (17, 13), (31, 32), (33, 65)] {
            let stride = w * 3 + 7;
            let mut pixels = vec![0; stride * h];
            let mut changed = pixels.clone();
            let mut state = 123456789_u32;
            for y in 0..h {
                for x in 0..w * 3 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    pixels[y * stride + x] = (state >> 24) as u8;
                    changed[y * stride + x] = pixels[y * stride + x].saturating_add((x % 19) as u8);
                }
            }
            let reference = metric
                .prepare_reference(RgbImage::srgb8(&pixels, w, h, stride).unwrap())
                .unwrap();
            let image = RgbImage::srgb8(&changed, w, h, stride).unwrap();
            let expected = Workspace::with_backend(Backend::Scalar)
                .unwrap()
                .compare(&reference, image)
                .unwrap();
            for backend in [Backend::Neon, Backend::Avx2, Backend::Sse2] {
                if !backend.is_available() {
                    continue;
                }
                let mut workspace = Workspace::with_backend(backend).unwrap();
                let got = workspace.compare(&reference, image).unwrap();
                assert!(got.jod.is_finite() && got.distortion >= 0.0);
                assert!(
                    (got.jod - expected.jod).abs() < 0.0002,
                    "{backend:?} {w}x{h} {display:?}: {} != {}",
                    got.jod,
                    expected.jod
                );
                // Shared immutable reference, independent per-worker scratch.
                assert_eq!(workspace.backend(), backend);
            }
        }
    }
}

#[test]
fn unavailable_backend_is_rejected_without_entering_cpu_instructions() {
    for backend in [Backend::Neon, Backend::Avx2, Backend::Sse2] {
        if !backend.is_available() {
            assert!(matches!(
                Workspace::with_backend(backend),
                Err(Error::UnsupportedBackend)
            ));
        }
    }
}
