"""Evaluate the torch checkpoints on the full MNIST test set.

    python evaluate.py                 # every checkpoint that exists
    python evaluate.py --synthetic     # also score the ten synthetic digits

Reports accuracy, a per-class report, and two diagnostics that catch a
degenerate classifier: how many inputs produce ten identical logits (the model
expressing no preference at all), and how many distinct classes it ever
predicts.
"""

import argparse
import os

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import preprocess
from digits import draw_digit
from models import NetCNN, NetMLP, NetMLPLegacy, get_device, load_torch_model

# name -> (class, checkpoint file)
CANDIDATES = [
    ("MLP (superseded)", NetMLPLegacy, "mlp20.pth"),
    ("MLP", NetMLP, "mlp.pth"),
    ("CNN", NetCNN, "cnn.pth"),
]


def load_test_set():
    test = datasets.MNIST(
        root="./data", train=False, download=not os.path.exists("./data/MNIST/raw"),
        transform=torchvision.transforms.ToTensor(),
    )
    x, y = next(iter(DataLoader(test, batch_size=len(test))))
    return x, y


@torch.no_grad()
def score(model, x, y, device):
    logits = torch.cat([model(x[i:i + 2000].to(device)).cpu()
                        for i in range(0, len(x), 2000)])
    pred = logits.argmax(1)
    # Count first, divide once. A float32 mean is not bit-comparable with the
    # float64 one numpy produces, which makes cross-runtime equality checks
    # fail for no real reason.
    accuracy = int((pred == y).sum()) / len(y)
    # A row where every logit is identical carries no class preference; argmax
    # returns index 0 purely by tie-break.
    degenerate = int((logits == logits.max(dim=1, keepdim=True).values).all(dim=1).sum())
    return accuracy, pred, degenerate


def score_onnx(x, y, torch_accuracy, torch_preds):
    """Score the exported artifacts and assert they match their checkpoints.

    This is the check that matters before shipping: a transposed input, a wrong
    flatten or a dtype mismatch survives a ten-image spot check but shows up
    immediately across ten thousand.
    """
    import onnxruntime as ort

    from artifacts import MODEL_FILES

    print("\nExported ONNX artifacts:")
    batch = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"  {name:16s} skipped — {path} not found")
            continue
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        logits = np.concatenate([sess.run(None, {input_name: batch[i:i + 2000]})[0]
                                 for i in range(0, len(batch), 2000)])
        pred = logits.argmax(1)
        accuracy = int((pred == y.numpy()).sum()) / len(y)
        expected = torch_accuracy.get(name)
        if expected is not None and accuracy != expected:
            raise SystemExit(
                f"  {name}: ONNX accuracy {accuracy:.6f} != torch {expected:.6f}"
            )
        disagree = int((pred != torch_preds[name].numpy()).sum()) if name in torch_preds else 0
        print(f"  {name:16s} accuracy {accuracy:.4f}   matches its checkpoint "
              f"({disagree} differing predictions)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="also score the ten synthetic digits from digits.py")
    parser.add_argument("--report", action="store_true",
                        help="print the per-class precision/recall/F1 table")
    parser.add_argument("--onnx", action="store_true",
                        help="also score the exported .onnx artifacts and "
                             "assert they match their torch checkpoints")
    args = parser.parse_args()

    device = get_device()
    x, y = load_test_set()
    print(f"MNIST test set: {len(x)} images · device: {device}\n")

    models = {}
    torch_accuracy = {}
    torch_preds = {}
    for name, cls, checkpoint in CANDIDATES:
        if not os.path.exists(checkpoint):
            print(f"{name:18s} skipped — {checkpoint} not found")
            continue
        model = load_torch_model(cls, checkpoint, device)
        models[name] = model
        accuracy, pred, degenerate = score(model, x, y, device)
        torch_accuracy[name] = accuracy
        torch_preds[name] = pred
        classes = sorted(set(pred.tolist()))
        print(f"{name:18s} accuracy {accuracy:.4f}   "
              f"ten-identical-logits: {degenerate:5d}/{len(x)}   "
              f"predicts {len(classes)}/10 classes {classes if len(classes) < 10 else ''}")

        if args.report:
            from sklearn.metrics import classification_report
            print(classification_report(y.numpy(), pred.numpy(), digits=4))

    if args.onnx:
        score_onnx(x, y, torch_accuracy, torch_preds)

    if not args.synthetic:
        return

    print("\nSynthetic freehand digits:")
    correct = dict.fromkeys(models, 0)
    for d in range(10):
        arr = preprocess.to_mnist_array(draw_digit(d), ink_is_dark=True)
        t = torch.from_numpy(arr).view(1, 1, 28, 28)
        row = []
        for name, model in models.items():
            with torch.no_grad():
                pred = int(model(t.to(device)).cpu().argmax(1)[0])
            correct[name] += int(pred == d)
            row.append(f"{name}: {pred}")
        print(f"  drew {d} -> " + "   ".join(row))
    print()
    for name in models:
        print(f"  {name:18s} {correct[name]}/10")


if __name__ == "__main__":
    main()
