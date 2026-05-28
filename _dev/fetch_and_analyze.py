"""Run after a full per-block sweep finishes. Fetches every cell image + the
labeled grid PNG, then ranks blocks by impact (MSE between value=0 and
value=1.0 for the same block — high MSE = critical block, low MSE = expendable).

Usage:
  # FLUX double-block sweep (default — backward compatible)
  python fetch_and_analyze.py <prompt_id>

  # Qwen-Image full sweep
  python fetch_and_analyze.py <prompt_id> --model qwen

  # explicit overrides
  python fetch_and_analyze.py <prompt_id> --prefix B --count 60 \
      --out-dir sweep_B_results --grid-node 9 --cell-node 10

SERVER env var overrides default ComfyUI URL.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SERVER = os.environ.get("SERVER", "http://127.0.0.1:8188")

PRESETS = {
    "flux": dict(prefix="D", count=19, out_dir="sweep_D_results",
                 grid_node="8", cell_node="9"),
    "qwen": dict(prefix="B", count=60, out_dir="sweep_B_results",
                 grid_node="9", cell_node="10"),
}
VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def view_url(filename, subfolder="", type_="output"):
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
    return f"{SERVER}/view?{qs}"


def fetch(url, dest):
    data = urllib.request.urlopen(url, timeout=30).read()
    dest.write_bytes(data)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_id")
    ap.add_argument("--model", choices=PRESETS.keys(), default="flux")
    ap.add_argument("--prefix", help="block prefix letter, e.g. D, S, B (overrides --model)")
    ap.add_argument("--count", type=int, help="number of blocks (overrides --model)")
    ap.add_argument("--out-dir", help="output subdir under _dev/ (overrides --model)")
    ap.add_argument("--grid-node", help="grid-saver node id (overrides --model)")
    ap.add_argument("--cell-node", help="per-cell SaveImage node id (overrides --model)")
    args = ap.parse_args()

    p = dict(PRESETS[args.model])
    for k in ("prefix", "count", "out_dir", "grid_node", "cell_node"):
        v = getattr(args, k.replace("-", "_"))
        if v is not None:
            p[k] = v

    blocks = [f"{p['prefix']}{i:02d}" for i in range(p["count"])]
    out_dir = Path(__file__).parent / p["out_dir"]
    out_dir.mkdir(exist_ok=True)

    hist = json.loads(urllib.request.urlopen(f"{SERVER}/history/{args.prompt_id}").read())
    outputs = hist[args.prompt_id]["outputs"]

    grid_imgs = outputs.get(p["grid_node"], {}).get("images", [])
    cell_imgs = outputs.get(p["cell_node"], {}).get("images", [])

    print(f"grid PNG count: {len(grid_imgs)}, cell PNG count: {len(cell_imgs)}")

    if grid_imgs:
        g = grid_imgs[0]
        size = fetch(view_url(g["filename"], g.get("subfolder", ""), g.get("type", "output")), out_dir / g["filename"])
        print(f"  saved grid: {g['filename']} ({size/1024/1024:.1f} MB)")

    cells = []
    for i, c in enumerate(cell_imgs):
        local = out_dir / f"cell_{i:03d}_{c['filename']}"
        fetch(view_url(c["filename"], c.get("subfolder", ""), c.get("type", "output")), local)
        cells.append(local)
    print(f"  saved {len(cells)} cells")

    expected = len(blocks) * len(VALUES)
    if len(cells) != expected:
        print(f"!! expected {expected} cells, got {len(cells)}")
        return

    arrays = {}
    for idx, path in enumerate(cells):
        block_i = idx // len(VALUES)
        val_i = idx % len(VALUES)
        block = blocks[block_i]
        val = VALUES[val_i]
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        arrays[(block, val)] = arr

    ref_imgs = [arrays[(b, 1.0)] for b in blocks]
    ref_mse = [float(np.mean((ref_imgs[0] - ri) ** 2)) for ri in ref_imgs]
    print(f"\nSanity: max MSE across the v=1.0 column = {max(ref_mse):.6f} (should be ~0)")

    impacts = []
    for b in blocks:
        diff = arrays[(b, 0.0)] - arrays[(b, 1.0)]
        mse = float(np.mean(diff ** 2))
        impacts.append((b, mse))

    impacts_sorted = sorted(impacts, key=lambda x: -x[1])
    print(f"\n=== Block impact ranking (knockout v=0 vs full v=1.0) ===")
    print(f"{'block':<6} {'MSE':>10}  bar")
    max_mse = impacts_sorted[0][1] if impacts_sorted[0][1] > 0 else 1.0
    for b, m in impacts_sorted:
        bar = "#" * int(40 * m / max_mse)
        print(f"{b:<6} {m:>10.6f}  {bar}")

    ranking_path = out_dir / "impact_ranking.txt"
    with open(ranking_path, "w") as f:
        f.write("block\tmse_v0_vs_v1\n")
        for b, m in impacts:
            f.write(f"{b}\t{m:.6f}\n")
    print(f"\nWrote {ranking_path}")


if __name__ == "__main__":
    main()
