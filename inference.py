"""Scoring layer: onnxruntime sessions over the exported model artifacts.

This is the only module the server imports for prediction. No torch — the
serving image ships numpy, PIL and onnxruntime, which is what takes the
container from 1.4 GB down to under 300 MB and the cold start from ~15s to ~1s.
"""

import numpy as np
import onnxruntime as ort

import weights
from artifacts import MODEL_FILES

# name -> (session, input tensor name)
_SESSIONS = {}


def _session_options():
    so = ort.SessionOptions()
    # These models are a handful of small matmuls. onnxruntime defaults to one
    # thread per core, which on a many-core host is pure scheduling overhead
    # and extra memory for no speedup.
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # errors only
    return so


def load_sessions(force=False):
    """Build every session once, at startup.

    Called from the server's startup hook rather than lazily on first request:
    if the artifact store is unreachable we want the process to fail at boot,
    not to serve a 500 to whoever happens to arrive first.
    """
    if _SESSIONS and not force:
        return _SESSIONS

    options = _session_options()
    for name, filename in MODEL_FILES.items():
        path = weights.fetch(filename, force=force)
        session = ort.InferenceSession(path, options,
                                       providers=["CPUExecutionProvider"])
        # Read the input name off the graph rather than hardcoding it, and
        # cache it so it is not re-queried on every request.
        _SESSIONS[name] = (session, session.get_inputs()[0].name)
    return _SESSIONS


def model_names():
    return list(MODEL_FILES)


def provider():
    """Active execution provider, e.g. 'CPUExecutionProvider'."""
    if not _SESSIONS:
        return "unloaded"
    session, _ = next(iter(_SESSIONS.values()))
    return session.get_providers()[0]


def runtime():
    return f"onnxruntime {ort.__version__}"


def softmax(logits):
    """Numerically stable softmax over the last axis.

    Subtracting the row max makes the largest exponent exactly exp(0) == 1, so
    the sum always lands in [1, n] and cannot overflow. torch.softmax does the
    same shift internally, which is why the two agree to float32 rounding.
    """
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict(name, x):
    """Score a (1,1,28,28) float32 array. Returns (class, probs[10], no_signal).

    `no_signal` flags a model that produced ten identical logits and therefore
    expressed no preference — argmax would return class 0 purely by tie-break.
    The current models never do this; it is a guard against shipping a
    degenerate artifact, and is expected to stay false.
    """
    session, input_name = _SESSIONS[name]
    logits = session.run(None, {input_name: x})[0][0]
    probs = softmax(logits)
    no_signal = bool(np.all(logits == logits.max()))
    return int(np.argmax(logits)), probs, no_signal
