"""Render docs/demo.png — the pipeline figure used in the README.

    python docs/make_demo.py [--digit 7]

Shows what actually happens to a drawing: the raw canvas, the 28x28 MNIST-format
array the models receive, and both models' probability distributions. Every
number in it comes from a real forward pass, not a mockup.
"""

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference  # noqa: E402
import preprocess  # noqa: E402
from digits import draw_digit  # noqa: E402

BG = (15, 17, 22)
PANEL = (23, 26, 33)
LINE = (38, 43, 54)
TEXT = (232, 234, 240)
MUTED = (139, 147, 167)
ACCENT = {"CNN": (110, 168, 254), "MLP": (240, 160, 75)}

W, H = 1200, 392
PAD = 32
ROW = 9           # vertical pitch of one probability bar
BAR_H = 5
BLOCK = 106       # vertical pitch of one model's block


def font(size, bold=False):
    candidates = [
        f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def rounded(d, box, radius, fill, outline=None):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)


def arrow(d, x, y, length=44):
    d.line([(x, y), (x + length, y)], fill=MUTED, width=2)
    d.polygon([(x + length, y), (x + length - 9, y - 5), (x + length - 9, y + 5)],
              fill=MUTED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--digit", type=int, default=7, choices=range(10))
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "demo.png"))
    args = p.parse_args()

    inference.load_sessions()

    drawing = draw_digit(args.digit)
    arr = preprocess.to_mnist_array(drawing, ink_is_dark=True)
    x = preprocess.to_input_array(drawing, ink_is_dark=True)
    results = {name: inference.predict(name, x) for name in inference.model_names()}

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_title, f_label, f_small = font(21, True), font(14), font(12)
    f_digit = font(46, True)
    f_tiny = font(9)   # class indices, one per 9px bar row

    d.text((PAD, 24), "What the model actually sees", font=f_title, fill=TEXT)
    d.text((PAD, 54),
           "A drawing is size-normalised into a 20x20 box and centred by centre of "
           "mass on a 28x28 canvas — the same pipeline MNIST itself used.",
           font=f_small, fill=MUTED)

    top = 92
    box = 224

    # 1. the raw drawing
    rounded(d, [PAD, top, PAD + box, top + box], 10, (255, 255, 255))
    img.paste(drawing.resize((box - 16, box - 16), Image.LANCZOS), (PAD + 8, top + 8))
    d.text((PAD, top + box + 12), "1 · canvas  280x280", font=f_label, fill=MUTED)

    # 2. the 28x28 model input
    x2 = PAD + box + 70
    rounded(d, [x2, top, x2 + box, top + box], 10, (0, 0, 0), LINE)
    img.paste(preprocess.array_to_image(arr).resize((box - 16, box - 16), Image.NEAREST),
              (x2 + 8, top + 8))
    d.text((x2, top + box + 12), "2 · model input  28x28", font=f_label, fill=MUTED)
    arrow(d, PAD + box + 13, top + box // 2)

    # 3. the two distributions
    x3 = x2 + box + 70
    arrow(d, x2 + box + 13, top + box // 2)
    rounded(d, [x3, top - 10, W - PAD, top + box - 4], 10, PANEL, LINE)

    # Each model gets a block: an identity column on the left, its ten
    # probability bars on the right.
    bx = x3 + 168
    bw = W - PAD - 24 - bx
    y = top - 2
    for name in ("MLP", "CNN"):
        cls, probs, _ = results[name]
        colour = ACCENT[name]

        d.text((x3 + 20, y + 6), name.upper(), font=f_small, fill=MUTED)
        d.text((x3 + 20, y + 20), str(cls), font=f_digit, fill=colour)
        d.text((x3 + 20, y + 70), f"{probs[cls] * 100:.1f}% confidence",
               font=f_small, fill=MUTED)

        for i in range(10):
            by = y + i * ROW
            d.rectangle([bx, by, bx + bw, by + BAR_H], fill=(35, 41, 54))
            fill_w = int(bw * probs[i])
            if fill_w > 0:
                d.rectangle([bx, by, bx + fill_w, by + BAR_H], fill=colour)
            d.text((bx - 13, by - 2), str(i), font=f_tiny, fill=MUTED)
        y += BLOCK

    d.text((x3 + 20, top + box + 12), "3 · predictions  10 classes",
           font=f_label, fill=MUTED)

    img.save(args.out)
    print(f"wrote {args.out}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
