"""ONNX Runtime scoring: sessions, softmax numerics, and model quality.

The accuracy assertions use the ten synthetic digits rather than the MNIST test
set so the suite stays fast and needs no dataset download. Full test-set
accuracy is verified separately by `evaluate.py --onnx`, which is a training-time
tool and needs torch.
"""

import numpy as np
import pytest

import inference
import preprocess
from digits import draw_digit


@pytest.fixture(scope="module", autouse=True)
def sessions():
    inference.load_sessions()


def test_both_models_load():
    assert set(inference.model_names()) == {"MLP", "CNN"}
    assert inference.provider() == "CPUExecutionProvider"
    assert inference.runtime().startswith("onnxruntime")


def test_softmax_sums_to_one():
    for logits in (np.array([1.0, 2.0, 3.0]), np.zeros(10), np.arange(10.0)):
        probs = inference.softmax(logits)
        assert probs.sum() == pytest.approx(1.0)
        assert (probs >= 0).all()


def test_softmax_does_not_overflow():
    """The max-shift is what keeps large logits finite; a plain exp would inf."""
    probs = inference.softmax(np.array([1000.0, 999.0, 0.0]))
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert probs.argmax() == 0


def test_softmax_is_shift_invariant():
    logits = np.array([2.0, 1.0, 0.5])
    np.testing.assert_allclose(
        inference.softmax(logits), inference.softmax(logits + 50.0), atol=1e-9
    )


def test_predict_contract():
    x = preprocess.to_input_array(draw_digit(4))
    cls, probs, no_signal = inference.predict("CNN", x)
    assert isinstance(cls, int) and 0 <= cls <= 9
    assert len(probs) == 10
    assert probs.sum() == pytest.approx(1.0)
    assert no_signal is False


@pytest.mark.parametrize("digit", range(10))
def test_cnn_classifies_every_synthetic_digit(digit):
    x = preprocess.to_input_array(draw_digit(digit))
    cls, probs, _ = inference.predict("CNN", x)
    assert cls == digit, f"CNN read a {digit} as {cls}"
    assert probs[cls] > 0.5


@pytest.mark.parametrize("digit", range(10))
def test_mlp_classifies_every_synthetic_digit(digit):
    x = preprocess.to_input_array(draw_digit(digit))
    cls, _, no_signal = inference.predict("MLP", x)
    assert cls == digit, f"MLP read a {digit} as {cls}"
    # The MLP's output layer has no activation, so ten identical logits — which
    # would mean the model expressed no preference — must never happen.
    assert no_signal is False


def test_blank_input_is_not_confidently_classified():
    """A blank canvas should not produce a high-confidence digit."""
    x = np.zeros((1, 1, 28, 28), dtype=np.float32)
    _, probs, _ = inference.predict("CNN", x)
    assert probs.max() < 0.999
