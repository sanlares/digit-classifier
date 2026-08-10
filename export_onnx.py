"""Export the trained checkpoints to ONNX.

    python export_onnx.py

Produces `mlp.onnx` and `cnn.onnx` with a dynamic batch axis, then checks that
onnxruntime reproduces torch's logits before declaring success. These artifacts
are what the service actually runs — torch is a build-time dependency only.
"""

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.export import Dim

import preprocess
from artifacts import CNN_ONNX, MLP_ONNX
from digits import draw_digit
from models import NetCNN, NetMLP, load_torch_model

OPSET = 18


def synthetic_batch():
    """The ten synthetic digits as one (10,1,28,28) float32 batch."""
    return np.concatenate([
        preprocess.to_input_array(draw_digit(d)) for d in range(10)
    ])


def export(model, path, example):
    # dynamic_shapes is keyed by the *forward parameter name* ("x"), not by
    # input_names. And the example must have batch > 1: torch.export
    # specialises size-1 dimensions to a constant, which would silently bake
    # batch=1 into the graph and make the dynamic axis disappear.
    torch.onnx.export(
        model,
        (example,),
        path,
        dynamo=True,
        external_data=False,   # keep it a single self-contained file
        optimize=True,
        opset_version=OPSET,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: Dim("batch", min=1, max=1024)}},
    )


def graph_shapes(path):
    m = onnx.load(path)

    def dims(value):
        return [d.dim_param or d.dim_value
                for d in value.type.tensor_type.shape.dim]

    return dims(m.graph.input[0]), dims(m.graph.output[0])


def check(name, torch_model, path, batch):
    in_shape, out_shape = graph_shapes(path)
    print(f"  {name}: input {in_shape} -> output {out_shape}")
    if not isinstance(out_shape[0], str):
        raise SystemExit(
            f"{path}: batch axis is fixed at {out_shape[0]}, expected a symbolic dim"
        )

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    with torch.no_grad():
        expected = torch_model(torch.from_numpy(batch)).numpy()
    got = sess.run(None, {input_name: batch})[0]

    if not np.allclose(expected, got, atol=1e-5):
        raise SystemExit(f"{path}: logits differ from torch by "
                         f"{np.abs(expected - got).max():.2e}")
    if not (expected.argmax(1) == got.argmax(1)).all():
        raise SystemExit(f"{path}: predictions differ from torch")
    print(f"     parity ok (max diff {np.abs(expected - got).max():.2e})")

    # The batch axis must really be dynamic, and batching must not change a
    # single row's result.
    for n in (1, 2, 16):
        padded = np.resize(batch, (n, 1, 28, 28)).astype(np.float32)
        out = sess.run(None, {input_name: padded})[0]
        if out.shape != (n, 10):
            raise SystemExit(f"{path}: batch {n} produced shape {out.shape}")
    single = sess.run(None, {input_name: batch[:1]})[0]
    full = sess.run(None, {input_name: batch})[0]
    if not np.allclose(single[0], full[0], atol=1e-5):
        raise SystemExit(f"{path}: row 0 changes with batch size")
    print("     dynamic batch ok (1, 2, 16)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-checkpoint", default="mlp.pth")
    p.add_argument("--cnn-checkpoint", default="cnn.pth")
    args = p.parse_args()

    example = torch.randn(2, 1, 28, 28)
    batch = synthetic_batch()

    targets = [
        ("MLP", NetMLP, args.mlp_checkpoint, MLP_ONNX),
        ("CNN", NetCNN, args.cnn_checkpoint, CNN_ONNX),
    ]

    for name, cls, checkpoint, path in targets:
        model = load_torch_model(cls, checkpoint)
        export(model, path, example)
        check(name, model, path, batch)

    print("\nexported: " + ", ".join(path for *_, path in targets))


if __name__ == "__main__":
    main()
