"""Horizontal bar chart of per-block MSE (knockout vs full).
Renders without matplotlib to avoid the dependency in _dev tooling.

Usage:
  python make_chart.py                              # FLUX.1 (default, backward compat)
  python make_chart.py --model qwen                 # Qwen-Image
  python make_chart.py --model sd35                 # SD3.5 Large
  python make_chart.py --results-dir sweep_X --title "..." --out custom.png
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PRESETS = {
    "flux": dict(
        results_dir="sweep_D_results",
        title="Per-block impact ranking — Flux double-blocks, frosting_lane LoRA",
        out_name="impact_chart.png",
    ),
    "qwen": dict(
        results_dir="sweep_B_results",
        title="Per-block impact ranking — Qwen-Image transformer blocks, modern-anime LoRA",
        out_name="impact_chart_b.png",
    ),
    "sd35": dict(
        results_dir="sweep_J_results",
        title="Per-block impact ranking — SD3.5 Large joint-blocks, anime LoRA",
        out_name="impact_chart_j.png",
    ),
}


def load_font(size):
    for p in ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=PRESETS.keys(), default="flux")
    ap.add_argument("--results-dir", help="overrides preset")
    ap.add_argument("--title", help="overrides preset")
    ap.add_argument("--out", help="output filename (relative to results dir), overrides preset")
    args = ap.parse_args()

    p = dict(PRESETS[args.model])
    if args.results_dir:
        p["results_dir"] = args.results_dir
    if args.title:
        p["title"] = args.title
    if args.out:
        p["out_name"] = args.out

    cells_dir = Path(__file__).parent / p["results_dir"]
    ranking = cells_dir / "impact_ranking.txt"

    rows = []
    with open(ranking) as f:
        next(f)
        for line in f:
            block, mse = line.strip().split("\t")
            rows.append((block, float(mse)))
    rows.sort(key=lambda r: -r[1])

    W = 1400
    ROW_H = 46 if len(rows) <= 30 else 28
    PAD = 30
    LABEL_W = 90
    VALUE_W = 130
    TITLE_H = 90
    BAR_AREA = W - LABEL_W - VALUE_W - 2 * PAD

    H = TITLE_H + len(rows) * ROW_H + 2 * PAD

    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)

    title_font = load_font(30)
    sub_font = load_font(18)
    block_font = load_font(22 if len(rows) <= 30 else 16)
    value_font = load_font(20 if len(rows) <= 30 else 14)

    d.text((PAD, PAD), p["title"], font=title_font, fill=(20, 20, 30))
    d.text((PAD, PAD + 42), "MSE between weight=0 (knockout) and weight=1.0 (full) for the same block. Higher bar = block carries more of the LoRA's style.",
           font=sub_font, fill=(110, 110, 130))

    max_mse = max(r[1] for r in rows)
    y0 = PAD + TITLE_H

    p33 = sorted(r[1] for r in rows)[int(len(rows) * 0.66)]
    p10 = sorted(r[1] for r in rows)[int(len(rows) * 0.33)]

    for i, (block, mse) in enumerate(rows):
        y = y0 + i * ROW_H
        d.text((PAD, y + (ROW_H - 22) // 2), block, font=block_font, fill=(20, 20, 30))
        bx = PAD + LABEL_W
        bw = int(BAR_AREA * mse / max_mse)
        if mse >= p33:
            color = (220, 90, 80)
        elif mse >= p10:
            color = (140, 160, 200)
        else:
            color = (190, 200, 215)
        d.rectangle([bx, y + 8, bx + bw, y + ROW_H - 8], fill=color)
        d.text((bx + BAR_AREA + 12, y + (ROW_H - 20) // 2), f"{mse:.5f}", font=value_font, fill=(60, 60, 80))

    out = cells_dir / p["out_name"]
    img.save(out, "PNG", compress_level=4)
    print(f"wrote {out}  {img.size[0]}x{img.size[1]}  ({out.stat().st_size/1024:.0f} KB)")

    half = img.resize((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    out2 = cells_dir / p["out_name"].replace(".png", "_half.png")
    half.save(out2, "PNG", compress_level=6)
    print(f"wrote {out2}  {half.size[0]}x{half.size[1]}  ({out2.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
