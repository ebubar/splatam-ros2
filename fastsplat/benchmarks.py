"""Run fastsplat on a standard benchmark dataset with one command.

Downloads a small, directly-fetchable image set and runs the full pipeline so
you can produce a splat quickly without a ZED capture.

    # download + splat the instant-ngp "fox" set (50 real photos)
    python -m fastsplat.benchmarks --dataset fox

    # just fetch the images (e.g. to inspect / run stages manually)
    python -m fastsplat.benchmarks --dataset fox --download-only

Datasets are fetched from public GitHub repos via the contents API (no auth, no
registration). Add your own to :data:`DATASETS`.
"""

import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request

_IMG_EXT = (".jpg", ".jpeg", ".png")

# Each dataset is fetched from raw.githubusercontent.com using its own
# transforms.json for the frame list (no GitHub API, no auth, no registration).
DATASETS = {
    # instant-ngp quickstart: real photos of a fox plush, object-centric.
    "fox": dict(
        raw_base="https://raw.githubusercontent.com/NVlabs/instant-ngp/master/data/nerf/fox",
        transforms="transforms.json",
        desc="instant-ngp fox: ~60 real photos, object-centric, ~a few MB.",
    ),
}


def _get(url, accept=None):
    headers = {"User-Agent": "fastsplat-benchmark"}
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers)


def _urlopen_retry(req, timeout=120, attempts=4):
    """Open a URL with exponential backoff (raw.githubusercontent can 404/429
    transiently for datacenter IPs)."""
    last = None
    for i in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last = e
            if i < attempts - 1:
                time.sleep(2 ** i)
    raise last


def _frame_names(transforms):
    """Extract image basenames from a NeRF-style transforms.json."""
    names = []
    for fr in transforms.get("frames", []):
        rel = (fr.get("file_path") or "").lstrip("./")
        if not rel:
            continue
        base = os.path.basename(rel)
        if not base.lower().endswith(_IMG_EXT):
            base += ".jpg"          # some transforms omit the extension
        names.append((rel if rel.lower().endswith(_IMG_EXT) else rel + ".jpg", base))
    return names


def download_dataset(name, root="data/benchmarks"):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; options: {sorted(DATASETS)}")
    ds = DATASETS[name]
    images_dir = os.path.join(root, name, "images")
    os.makedirs(images_dir, exist_ok=True)

    tf_url = f"{ds['raw_base']}/{ds['transforms']}"
    with _urlopen_retry(_get(tf_url, "application/json"), timeout=60) as r:
        transforms = json.load(r)
    files = _frame_names(transforms)
    if not files:
        raise RuntimeError(f"no frames listed in {tf_url}")

    print(f"[benchmark] {name}: downloading {len(files)} images -> {images_dir}")
    for i, (rel, base) in enumerate(files):
        dest = os.path.join(images_dir, base)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        with _urlopen_retry(_get(f"{ds['raw_base']}/{rel}")) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        time.sleep(0.15)          # be polite to raw.githubusercontent rate limits
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"[benchmark]   {i + 1}/{len(files)}")
    return images_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fox", choices=sorted(DATASETS))
    ap.add_argument("--config", default="configs/fastsplat/benchmark.py")
    ap.add_argument("--root", default="data/benchmarks")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    images_dir = download_dataset(args.dataset, args.root)
    if args.download_only:
        print(f"[benchmark] images ready: {images_dir}")
        return

    # Imported here so --download-only needs no torch/gsplat/mapanything.
    from .run_pipeline import run
    from .utils.config import load_config

    cfg = load_config(args.config)
    cfg["ingest"]["source"] = "folder"
    cfg["ingest"]["image_dir"] = images_dir
    cfg["run_name"] = f"benchmark_{args.dataset}"
    run(cfg)


if __name__ == "__main__":
    main()
