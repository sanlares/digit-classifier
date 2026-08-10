"""Synthetic freehand digits, used for tests and for the UI placeholder.

Strokes are defined on a 280x280 canvas (the same one the UI uses), drawn in
black on white so they go through exactly the same preprocessing as a real
drawing.
"""

from PIL import Image, ImageDraw

SIZE = 280
STROKE = 22

STROKES = {
    0: [[(140, 45), (95, 70), (78, 140), (95, 212), (140, 236), (185, 212), (202, 140), (185, 70), (140, 45)]],
    1: [[(112, 82), (142, 55), (146, 235)]],
    2: [[(80, 90), (110, 55), (165, 58), (188, 100), (150, 150), (85, 215), (200, 218)]],
    3: [[(82, 68), (150, 52), (190, 88), (150, 135), (110, 140)],
        [(150, 135), (196, 175), (165, 226), (95, 228), (74, 205)]],
    4: [[(165, 45), (80, 168), (208, 168)], [(160, 105), (158, 238)]],
    5: [[(190, 55), (95, 58), (88, 130), (140, 122), (190, 150), (182, 205), (120, 232), (78, 212)]],
    6: [[(178, 55), (110, 95), (85, 165), (95, 215), (150, 236), (192, 200), (180, 152), (125, 140), (88, 168)]],
    7: [[(75, 62), (200, 58), (125, 238)], [(95, 150), (172, 148)]],
    8: [[(140, 48), (95, 78), (105, 122), (145, 140), (185, 165), (180, 215), (135, 238), (92, 212), (98, 165), (145, 140), (180, 112), (172, 68), (140, 48)]],
    9: [[(175, 105), (150, 62), (100, 68), (82, 112), (110, 148), (168, 138), (178, 95)],
        [(172, 120), (160, 200), (128, 238)]],
}


def draw_digit(digit, size=SIZE, stroke=STROKE):
    """Return a PIL.Image (RGB) with the digit drawn in black on white."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    scale = size / SIZE
    for path in STROKES[digit]:
        pts = [(x * scale, y * scale) for x, y in path]
        d.line(pts, fill="black", width=int(stroke * scale), joint="curve")
        # Rounded caps: PIL does not add them on its own
        r = stroke * scale / 2
        for x, y in (pts[0], pts[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill="black")
    return img
