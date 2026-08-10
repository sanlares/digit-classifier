"""Turn an arbitrary drawing into a 28x28 image in MNIST format.

MNIST digits are not raw crops: the dataset size-normalises each digit into a
20x20 box preserving aspect ratio, then centres it on a 28x28 canvas using the
**centre of mass** — not the centre of the bounding box. The models were trained
on those images as-is (pixels scaled to [0,1], no mean/std normalisation), so
this module replicates that pipeline to make a hand-drawn digit look like what
they saw during training. Skipping it costs a lot of accuracy.

Numpy and PIL only: this runs in the serving container, which has no torch.
"""

import numpy as np
from PIL import Image

CANVAS_SIZE = 28
BOX_SIZE = 20  # inner box the digit is scaled into


def to_mnist_array(image, ink_is_dark=True, threshold=0.12):
    """Return a float32 28x28 array in [0, 1]: white digit on black background.

    image        : PIL.Image of any size/mode.
    ink_is_dark  : True if the source has dark strokes on a light background
                   (paper-like). False if it is already MNIST-style.
    threshold    : values below this count as background when locating the
                   digit's bounding box; keeps antialiasing from inflating it.
    """
    img = image.convert("L")
    a = np.asarray(img, dtype=np.float32) / 255.0

    if ink_is_dark:
        a = 1.0 - a  # strokes are now high, background ~0

    # Bounding box of the digit, ignoring antialiasing noise
    mask = a > threshold
    if not mask.any():
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Scale into a 20x20 box, preserving aspect ratio
    h, w = a.shape
    scale = BOX_SIZE / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    digit = Image.fromarray((a * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    a = np.asarray(digit, dtype=np.float32) / 255.0

    # Centre by centre of mass on the 28x28 canvas
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    total = a.sum()
    if total > 0:
        ys, xs = np.indices(a.shape)
        com_y = float((ys * a).sum() / total)
        com_x = float((xs * a).sum() / total)
    else:
        com_y, com_x = new_h / 2.0, new_w / 2.0

    top = int(round(CANVAS_SIZE / 2.0 - com_y))
    left = int(round(CANVAS_SIZE / 2.0 - com_x))
    top = max(0, min(CANVAS_SIZE - new_h, top))
    left = max(0, min(CANVAS_SIZE - new_w, left))

    canvas[top:top + new_h, left:left + new_w] = a
    return np.clip(canvas, 0.0, 1.0)


def to_input_array(image, ink_is_dark=True):
    """Same as to_mnist_array, shaped and typed for onnxruntime.

    Returns a C-contiguous float32 (1, 1, 28, 28) array. onnxruntime rejects
    non-contiguous buffers and float64, so both are made explicit rather than
    relied upon.
    """
    a = to_mnist_array(image, ink_is_dark=ink_is_dark)
    return np.ascontiguousarray(a, dtype=np.float32).reshape(
        1, 1, CANVAS_SIZE, CANVAS_SIZE
    )


def array_to_image(a, scale=1):
    """28x28 array in [0,1] -> grayscale PIL.Image (MNIST format)."""
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode="L")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img
