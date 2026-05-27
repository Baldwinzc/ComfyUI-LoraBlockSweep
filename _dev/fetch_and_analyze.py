"""Run after the full D sweep finishes. Fetches every cell image + the labeled
grid PNG, then ranks blocks by impact (MSE between value=0 and value=1.0 for
the same block — high MSE = critical block, low MSE = expendable).

Usage:  python fetch_and_analyze.py <prompt_id>
        SERVER env var overrides default ComfyUI URL.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SERVER = os.environ.get("SERVER", "http://127.0.0.1:8188")
PROMPT_ID = sys.argv[1] if len(sys.argv) > 1 else ""
if not PROMPT_ID:
    sys.exit("usage: python fetch_and_analyze.py <prompt_id>  (SERVER env optional)")
OUT_DIR = Path(__file__).parent / "sweep_D_results"
OUT_DIR.mkdir(exist_ok=True)

BLOCKS = [f"D{i:02d}" for i in range(19)]
VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def view_url(filename, subfolder="", type_="output"):
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
    return f"{SERVER}/view?{qs}"


def fetch(url, dest):
    data = urllib.request.urlopen(url, timeout=30).read()
    dest.write_bytes(data)
    return len(data)


def main():
    hist = json.loads(urllib.request.urlopen(f"{SERVER}/history/{PROMPT_ID}").read())
    outputs = hist[PROMPT_ID]["outputs"]

    grid_imgs = outputs.get("8", {}).get("images", [])
    cell_imgs = outputs.get("9", {}).get("images", [])

    print(f"grid PNG count: {len(grid_imgs)}, cell PNG count: {len(cell_imgs)}")

    if grid_imgs:
        g = grid_imgs[0]
        size = fetch(view_url(g["filename"], g.get("subfolder", ""), g.get("type", "output")), OUT_DIR / g["filename"])
        print(f"  saved grid: {g['filename']} ({size/1024/1024:.1f} MB)")

    cells = []
    for i, c in enumerate(cell_imgs):
        local = OUT_DIR / f"cell_{i:03d}_{c['filename']}"
        fetch(view_url(c["filename"], c.get("subfolder", ""), c.get("type", "output")), local)
        cells.append(local)
    print(f"  saved {len(cells)} cells")

    if len(cells) != len(BLOCKS) * len(VALUES):
        print(f"!! expected {len(BLOCKS)*len(VALUES)} cells, got {len(cells)}")
        return

    arrays = {}
    for idx, path in enumerate(cells):
        block_i = idx // len(VALUES)
        val_i = idx % len(VALUES)
        block = BLOCKS[block_i]
        val = VALUES[val_i]
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        arrays[(block, val)] = arr

    ref_imgs = [arrays[(b, 1.0)] for b in BLOCKS]
    ref_mse = [float(np.mean((ref_imgs[0] - ri) ** 2)) for ri in ref_imgs]
    print(f"\nSanity: max MSE across the v=1.0 column = {max(ref_mse):.6f} (should be ~0)")

    impacts = []
    for b in BLOCKS:
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

    ranking_path = OUT_DIR / "impact_ranking.txt"
    with open(ranking_path, "w") as f:
        f.write("block\tmse_v0_vs_v1\n")
        for b, m in impacts:
            f.write(f"{b}\t{m:.6f}\n")
    print(f"\nWrote {ranking_path}")


if __name__ == "__main__":
    main()
