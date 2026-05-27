"""Horizontal bar chart of per-block MSE (knockout vs full).
Renders without matplotlib to avoid the dependency in _dev tooling.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELLS_DIR = Path(__file__).parent / "sweep_D_results"
RANKING = CELLS_DIR / "impact_ranking.txt"


def load_font(size):
    for p in ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    rows = []
    with open(RANKING) as f:
        next(f)
        for line in f:
            block, mse = line.strip().split("\t")
            rows.append((block, float(mse)))
    # Sort descending by MSE — most impactful first
    rows.sort(key=lambda r: -r[1])

    W = 1400
    ROW_H = 46
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
    block_font = load_font(22)
    value_font = load_font(20)

    d.text((PAD, PAD), "Per-block impact ranking — Flux double-blocks, frosting_lane LoRA", font=title_font, fill=(20, 20, 30))
    d.text((PAD, PAD + 42), "MSE between weight=0 (knockout) and weight=1.0 (full) for the same block. Higher bar = block carries more of the LoRA's style.",
           font=sub_font, fill=(110, 110, 130))

    max_mse = max(r[1] for r in rows)
    y0 = PAD + TITLE_H

    # Bucket coloring
    p33 = sorted(r[1] for r in rows)[int(len(rows) * 0.66)]
    p10 = sorted(r[1] for r in rows)[int(len(rows) * 0.33)]

    for i, (block, mse) in enumerate(rows):
        y = y0 + i * ROW_H
        # Label
        d.text((PAD, y + 10), block, font=block_font, fill=(20, 20, 30))
        # Bar
        bx = PAD + LABEL_W
        bw = int(BAR_AREA * mse / max_mse)
        if mse >= p33:
            color = (220, 90, 80)  # critical
        elif mse >= p10:
            color = (140, 160, 200)  # mid
        else:
            color = (190, 200, 215)  # negligible
        d.rectangle([bx, y + 8, bx + bw, y + ROW_H - 8], fill=color)
        # Value text
        d.text((bx + BAR_AREA + 12, y + 10), f"{mse:.5f}", font=value_font, fill=(60, 60, 80))

    out = CELLS_DIR / "impact_chart.png"
    img.save(out, "PNG", compress_level=4)
    print(f"wrote {out}  {img.size[0]}x{img.size[1]}  ({out.stat().st_size/1024:.0f} KB)")

    # Half for README
    half = img.resize((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    out2 = CELLS_DIR / "impact_chart_half.png"
    half.save(out2, "PNG", compress_level=6)
    print(f"wrote {out2}  {half.size[0]}x{half.size[1]}  ({out2.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
