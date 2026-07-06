# Operator Viewer & Region-of-Interest Refinement

The live SplaTAM node serves a **browser-based operator viewer**. An operator on
a separate laptop opens a URL, watches the splat build in real time, boxes an
object, and requests a higher-quality splat of just that region.

No ROS2, CUDA, or Python is required on the laptop — only a browser.

```
 ZED 2i ──ROS2──▶  SplaTAM PC  ──HTTP──▶  Operator laptop (browser)
 (Orin)            zed2i_splat_live.py     three.js point-cloud viewer
                        │  ▲                      │
                        │  └──── ROI refine ◀─────┘  (draw box → "Refine")
                        ▼
                  livesplat/  (bridge, viewer_server, capture_advisor, roi)
```

## Running

Start the live pipeline as usual:

```bash
python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_splat_live.py
```

The console prints the viewer address. From the operator laptop open:

```
http://<splatam-pc-ip>:8080/
```

(Port and bind address are configurable under `viewer=dict(...)` in the config.
`host="0.0.0.0"` serves on all interfaces; use the PC's LAN IP from the laptop.)

## Using the viewer

* **Orbit / zoom / pan** the live point cloud with the mouse.
* Panel shows **frame count, Gaussian count, ROI coverage %, and next-best-view
  hints**.
* Position the green **ROI box** with the center X/Y/Z and size sliders.
* Set the **Quality** slider (1–5): higher spends more refinement iterations.
* Click **Refine this region**.

When you refine, the SLAM loop:

1. Pauses live ingest (BEST_EFFORT QoS simply drops frames during this).
2. Selects the keyframes that actually see the box.
3. Runs `base_iters × quality` extra optimisation iterations focused on those
   views (default `base_iters = 300`).
4. Recomputes **coverage** of the box from all captured poses and publishes
   **next-best-view hints** so you can walk the missing arcs (see
   `docs/CAPTURE_PATTERNS.md`), then refine again for ultra-high quality.

After the run finishes, the process keeps the viewer alive on the final splat so
you can keep inspecting and requesting refinements. `Ctrl-C` to exit.

## Endpoints (for scripting / debugging)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | operator HTML page |
| GET | `/status` | JSON: frames, gaussians, coverage, guidance |
| GET | `/snapshot_version` | JSON: integer, increments on new geometry |
| GET | `/snapshot.bin` | binary point cloud `[u32 n][n·3 f32 xyz][n·3 u8 rgb]` |
| POST | `/roi` | JSON `RefineRequest` → queues a refinement |

## Config knobs (`configs/zed2i/zed2i_splat_live.py`)

```python
viewer=dict(
    enable=True,
    host="0.0.0.0",
    port=8080,
    publish_every=3,           # push a splat snapshot every N frames
    max_preview_points=150000, # cap points streamed to the laptop
),
roi_refine=dict(base_iters=300),  # * quality slider = refine iterations
```

## Offline operator laptop

The page loads three.js from a CDN. If the laptop has no internet, vendor
`three.module.js` + `OrbitControls.js` next to `livesplat/viewer_server.py` and
point `THREE_JS_URL` / `ORBIT_URL` at those local files.
