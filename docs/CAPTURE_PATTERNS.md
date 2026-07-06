# High-Quality Capture Patterns for Gaussian-Splat Objects

This note explains **how to move the ZED 2i to get an ultra-high-quality splat
of a specific object in as little time as possible**, and how those rules are
encoded in `livesplat/capture_advisor.py` and the operator viewer.

The goal: an operator sweeps a scene to build a live map, spots an object worth
capturing well, boxes it in the viewer, and the system tells them exactly where
to point the camera next so the object reconstructs at benchmark quality.

---

## What the benchmarks actually say

Across the datasets that Gaussian-splatting and NeRF papers are scored on, the
same drivers of quality show up repeatedly:

| Dataset | Capture style | Lesson that transfers to us |
|---|---|---|
| **DTU (MVS)** | Object on a turntable, cameras on fixed arcs at a few elevations | Bounded objects need **multi-elevation angular coverage**, not more frames from one spot. |
| **Mip-NeRF 360** | Full 360° orbit around a central object at roughly constant radius | **Constant radius + full azimuth** is what kills "floaters" and background bleed. |
| **Tanks & Temples** | Handheld loops around objects/buildings | **Loop closure** (returning to earlier views) bounds drift and sharpens geometry. |
| **ScanNet++ / DSLR** | Slow, high-overlap coverage, dense laser reference | **High pairwise overlap** (slow motion, small baseline steps) beats fast sweeps. |
| **Replica / TUM (SplaTAM's own evals)** | Smooth trajectories, modest motion blur | **Smooth, blur-free motion** matters as much as coverage for splat sharpness. |

Distilled into rules of thumb for a single object:

1. **Angular coverage dominates.** Quality tracks *how much of the viewing
   sphere around the object you covered*, far more than raw frame count. A
   handful of well-spread views beats hundreds from one side.
2. **Roughly constant radius.** Orbit the object at a stable distance so it
   fills a similar fraction of the frame. Wildly varying distance starves some
   scales of detail and produces inconsistent Gaussian sizes.
3. **Multiple elevations.** One equatorial ring leaves the top and underside
   under-constrained. Two–three rings (low, mid, high) close most of the
   hemisphere. Avoid pure top-down: stereo depth degrades when the baseline is
   perpendicular to the surface.
4. **Generous overlap (~60–80%).** Move in small azimuth steps (~20–30°) so
   consecutive views share most of the object. Overlap is what lets tracking
   and mapping stay locked and geometry stay crisp.
5. **Close the loop.** Return to a couple of earlier viewpoints. With VIO-seeded
   tracking this re-anchors the pose and removes accumulated drift on the object.
6. **Move slowly and smoothly.** Motion blur and rolling-shutter smear cannot be
   recovered by the optimiser. Slow is fast here.
7. **Keep the object in the ZED's good depth band.** For NEURAL depth that's
   roughly **0.4–5 m**. Too close saturates disparity; too far gets noisy.

---

## The pattern we target

Given an object of radius `r` (half the diagonal of the operator's box), the
advisor plans a capture on a sphere of radius `R ≈ 2.5·r` (`DEFAULT_RADIUS_SCALE`):

```
        elevation ~70°   . . . . .      (high ring, looks down onto the top)
        elevation ~45°  . . . . . .     (mid ring, the workhorse)
        elevation ~15° . . . . . . .    (low ring, catches undersides/labels)
                        └ 24° azimuth steps → ~15 views per ring
```

* **3 elevation rings** at ~15°, 45°, 70° (`DEFAULT_ELEVATIONS_DEG`).
* **~24° azimuth spacing** (`DEFAULT_AZIMUTH_STEP_DEG`) → ~15 views/ring, ~45
  views total for full coverage — the empirically "enough" budget for a clean
  object splat, an order of magnitude less than a naive dense sweep.
* **Radius ≈ 2.5× object radius**, kept inside the NEURAL-depth sweet spot.

`livesplat.capture_advisor.ideal_capture_plan(roi)` returns exactly this target
set of viewpoints (camera position + look-at) for any boxed object.

---

## Closing the loop: coverage-driven guidance

You don't want to blindly execute all 45 views — you want to capture only what's
*missing*. That's what `analyse_coverage(roi, captured_w2c)` does:

1. Take the camera centres you've already captured (from the live poses).
2. Bin their viewing directions to the object over an **azimuth × elevation
   grid** (24 × 6 bins).
3. `coverage` = fraction of bins that saw the object (shown as a live bar in the
   viewer).
4. Rank the **empty** bins by how far they are from any covered bin (biggest
   holes first) and emit those as **next-best-view hints** — "go to azimuth
   262°, elevation 45°."

The operator viewer displays coverage % and the top hints live, so an operator
can fill the sphere efficiently and stop as soon as coverage is high, instead of
over-capturing one side and missing another. This is the "collect exactly the
requisite data, fast" system.

---

## How this plugs into the two-phase workflow

1. **Explore (live SLAM).** Walk the scene normally. VIO-seeded tracking + ZED
   neural depth build a live map. Frames are processed at the incoming rate
   (`process_every_n`, `map_every`), streamed to the operator laptop.
2. **Select.** Operator boxes an object in the browser viewer.
3. **Refine now.** The system immediately re-optimises that region from the
   keyframes that already see it (`run_roi_refinement`, iterations scale with
   the quality slider) — an instant quality bump from existing data.
4. **Guided recapture.** In parallel it reports coverage and next-best-view
   hints. The operator walks the missing arcs; those frames flow into the same
   map, and a second Refine locks in ultra-high quality.

Tunables live in `configs/zed2i/zed2i_splat_live.py` (`roi_refine`, `viewer`)
and in the advisor constants at the top of `livesplat/capture_advisor.py`.
