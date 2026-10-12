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

//! A thread-local allocation counter verifies the hot-path contract, including
//! the first candidate and switching to smaller references with odd depths.
#![allow(unsafe_code)]

use cvvdp_lite::{CvvdpLite, Display, RgbImage, Workspace};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountedAllocator;

fn record() {
    let _ = RECORDING.try_with(|active| {
        if active.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}

// SAFETY: Every allocation operation delegates with its unchanged pointer and
// layout to System; the thread-local counter never allocates or frees memory.
unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: GlobalAlloc's caller supplies a valid allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: GlobalAlloc's caller supplies a valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        // SAFETY: GlobalAlloc's caller supplies a live pointer and matching layout.
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: GlobalAlloc's caller supplies a live pointer and matching layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

fn measured<T>(function: impl FnOnce() -> T) -> (T, usize) {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            RECORDING.set(false);
        }
    }
    ALLOCATIONS.set(0);
    RECORDING.set(true);
    let guard = Stop;
    let value = function();
    drop(guard);
    (value, ALLOCATIONS.get())
}

#[test]
fn reserved_workspace_allocates_nothing_on_first_or_repeated_candidates() {
    let metric = CvvdpLite::new(Display::default()).unwrap();
    let mut workspace = Workspace::new();
    workspace.reserve(129, 513).unwrap();
    let bytes = workspace.allocated_bytes();
    for (w, h) in [
        (129, 513),
        (7, 513),
        (129, 7),
        (65, 65),
        (31, 33),
        (63, 64),
        (128, 127),
        (1, 17),
    ] {
        let pixels: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 31 + i / 3) % 256) as u8)
            .collect();
        let changed: Vec<u8> = pixels.iter().map(|v| v.saturating_add(3)).collect();
        let image = RgbImage::srgb8(&pixels, w, h, w * 3).unwrap();
        let candidate = RgbImage::srgb8(&changed, w, h, w * 3).unwrap();
        let reference = metric
            .prepare_reference_with(image, &mut workspace)
            .unwrap();
        for _ in 0..3 {
            let (score, allocations) =
                measured(|| workspace.compare(&reference, candidate).unwrap());
            assert_eq!(allocations, 0, "{w}x{h} candidate unexpectedly allocated");
            assert!(score.jod.is_finite());
        }
        assert_eq!(workspace.allocated_bytes(), bytes);
        let mut evaluator = reference.evaluator_with(workspace);
        let (score, allocations) = measured(|| evaluator.compare(image).unwrap());
        assert_eq!(allocations, 0);
        assert_eq!(score.jod, 10.0);
        workspace = evaluator.into_workspace();
    }
}
