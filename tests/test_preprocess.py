"""Properties of the MNIST normalisation pipeline.

These assert the pipeline's contract rather than exact pixel values: the digit
lands in a 20x20 box, gets centred by centre of mass, and comes out in the range
and dtype the models expect. Those are the properties that, if broken, quietly
cost accuracy without raising anything.
"""

import numpy as np
import pytest
from PIL import Image

import preprocess
from digits import draw_digit


def test_output_shape_and_range():
    arr = preprocess.to_mnist_array(draw_digit(3))
    assert arr.shape == (28, 28)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_digit_is_white_on_black():
    """MNIST convention: high values are strokes, the background is ~0."""
    arr = preprocess.to_mnist_array(draw_digit(8))
    assert arr.max() > 0.9, "stroke should reach near 1.0"
    assert arr[0, 0] == 0.0, "corner should be background"
    assert arr.mean() < 0.3, "most of the canvas is background"


def test_fits_in_20x20_box():
    """The dataset scales every digit into a 20x20 box; so do we."""
    for digit in range(10):
        arr = preprocess.to_mnist_array(draw_digit(digit))
        rows = np.where(arr.any(axis=1))[0]
        cols = np.where(arr.any(axis=0))[0]
        height = rows[-1] - rows[0] + 1
        width = cols[-1] - cols[0] + 1
        assert max(height, width) <= 20, f"digit {digit} is {height}x{width}"


def test_centred_by_centre_of_mass():
    """Centre of mass should land on the middle of the canvas, not the bbox centre."""
    for digit in range(10):
        arr = preprocess.to_mnist_array(draw_digit(digit))
        ys, xs = np.indices(arr.shape)
        total = arr.sum()
        com_y = (ys * arr).sum() / total
        com_x = (xs * arr).sum() / total
        # One pixel of slack: the paste offset is rounded to whole pixels.
        assert abs(com_y - 14) <= 1.5, f"digit {digit} centre of mass y={com_y:.2f}"
        assert abs(com_x - 14) <= 1.5, f"digit {digit} centre of mass x={com_x:.2f}"


def test_translation_invariance():
    """The same digit drawn in a different corner must normalise identically."""
    canvas = Image.new("RGB", (280, 280), "white")
    canvas.paste(draw_digit(5).resize((120, 120)), (10, 10))
    shifted = Image.new("RGB", (280, 280), "white")
    shifted.paste(draw_digit(5).resize((120, 120)), (150, 150))

    np.testing.assert_allclose(
        preprocess.to_mnist_array(canvas),
        preprocess.to_mnist_array(shifted),
        atol=1e-6,
    )


def test_blank_canvas_is_all_zero():
    blank = Image.new("RGB", (280, 280), "white")
    arr = preprocess.to_mnist_array(blank)
    assert arr.max() == 0.0


def test_inverted_input():
    """ink_is_dark=False accepts an image already in MNIST orientation."""
    mnist_style = preprocess.array_to_image(preprocess.to_mnist_array(draw_digit(2)))
    arr = preprocess.to_mnist_array(mnist_style, ink_is_dark=False)
    assert arr.max() > 0.9


def test_to_input_array_is_model_ready():
    x = preprocess.to_input_array(draw_digit(7))
    assert x.shape == (1, 1, 28, 28)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"], "onnxruntime rejects non-contiguous buffers"


@pytest.mark.parametrize("digit", range(10))
def test_every_synthetic_digit_produces_ink(digit):
    assert preprocess.to_mnist_array(draw_digit(digit)).sum() > 0
