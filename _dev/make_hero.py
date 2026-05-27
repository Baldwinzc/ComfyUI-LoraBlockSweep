"""4-up group knockout hero — Full | Top-7 off | Bot-7 off | No LoRA.

Fetches the four images produced by group_hero_workflow.json, lays them out
side-by-side with labels, and writes hero_group.png.

Usage:  python make_hero.py <prompt_id>
        SERVER env var overrides default ComfyUI URL.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SERVER = os.environ.get("SERVER", "http://127.0.0.1:8188")
PROMPT_ID = sys.argv[1] if len(sys.argv) > 1 else ""
if not PROMPT_ID:
    sys.exit("usage: python make_hero_v4.py <prompt_id>  (SERVER env optional)")
OUT_DIR = Path(__file__).parent / "sweep_D_results"
OUT_DIR.mkdir(exist_ok=True)

# The 4 SaveImage nodes in group_hero_workflow.json, in display order.
# (Branch base offset i = 10 + idx*10, SaveImage = base+5.)
BRANCHES = [
    ("15", "full", "Full LoRA",   "all 19 D + 38 S blocks @ 1.0"),
    ("25", "top7", "Top-7 off",   "D00 D02 D03 D07 D08 D09 D15 → 0"),
    ("35", "bot7", "Bot-7 off",   "D01 D10 D12 D13 D14 D17 D18 → 0"),
    ("45", "none", "No LoRA",     "all 57 blocks @ 0"),
]


def load_font(size, bold=False):
    cand = [
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in cand:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def view_url(filename, subfolder="", type_="output"):
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
    return f"{SERVER}/view?{qs}"


def main():
    hist = json.loads(urllib.request.urlopen(f"{SERVER}/history/{PROMPT_ID}").read())
    outputs = hist[PROMPT_ID]["outputs"]

    fetched = {}
    for node_id, slug, _, _ in BRANCHES:
        info = outputs.get(node_id, {}).get("images", [])
        if not info:
            print(f"!! node {node_id} ({slug}) produced no image")
            return
        img_info = info[0]
        url = view_url(img_info["filename"], img_info.get("subfolder", ""), img_info.get("type", "output"))
        local = OUT_DIR / f"group_{slug}_{img_info['filename']}"
        local.write_bytes(urllib.request.urlopen(url, timeout=30).read())
        fetched[slug] = local
        print(f"  saved {slug}: {local.name}")

    CELL = 560
    PAD = 18
    GAP = 18
    HEADER_H = 96
    FOOT_H = 70

    cols = len(BRANCHES)
    W = cols * CELL + (cols - 1) * GAP + 2 * PAD
    H = HEADER_H + CELL + FOOT_H + 2 * PAD

    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)

    title_font = load_font(30, bold=True)
    sub_font = load_font(18)
    cap_font = load_font(16)

    for ci, (_, slug, title, note) in enumerate(BRANCHES):
        x = PAD + ci * (CELL + GAP)
        bb = d.textbbox((0, 0), title, font=title_font)
        d.text((x + (CELL - (bb[2] - bb[0])) // 2, PAD + 6), title, font=title_font, fill=(20, 20, 30))
        bb2 = d.textbbox((0, 0), note, font=sub_font)
        d.text((x + (CELL - (bb2[2] - bb2[0])) // 2, PAD + 50), note, font=sub_font, fill=(120, 120, 140))

        c = Image.open(fetched[slug]).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        img.paste(c, (x, PAD + HEADER_H))

    cap = ("Same seed, same prompt. Knocking out the 7 highest-MSE blocks "
           "(Top-7 off) strips most of the LoRA's style — close to the No-LoRA "
           "baseline. Knocking out the 7 lowest-MSE blocks (Bot-7 off) is "
           "visually indistinguishable from Full.")
    words = cap.split()
    lines, cur = [], ""
    max_w = W - 60
    for w in words:
        test = (cur + " " + w).strip()
        bb = d.textbbox((0, 0), test, font=cap_font)
        if bb[2] - bb[0] > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    cy = PAD + HEADER_H + CELL + 16
    for line in lines:
        bb = d.textbbox((0, 0), line, font=cap_font)
        d.text(((W - (bb[2] - bb[0])) // 2, cy), line, font=cap_font, fill=(70, 70, 90))
        cy += 22

    out = OUT_DIR / "hero_group.png"
    img.save(out, "PNG", compress_level=4)
    print(f"\nwrote {out}  {img.size[0]}x{img.size[1]}  ({out.stat().st_size/1024:.0f} KB)")

    half = img.resize((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    out2 = OUT_DIR / "hero_group_half.png"
    half.save(out2, "PNG", compress_level=6)
    print(f"wrote {out2}  {half.size[0]}x{half.size[1]}  ({out2.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
