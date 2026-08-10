"""End-to-end check without a browser.

Generates a freehand digit (black stroke on white, same as the UI canvas), runs
it through the same preprocessing and the same ONNX artifacts the service uses.

    python test_local.py              # draws a 3
    python test_local.py --digit 7
    python test_local.py --all        # all ten, with a summary
    python test_local.py --image my_drawing.png
"""

import argparse

from PIL import Image

import inference
import preprocess
import weights
from digits import draw_digit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=3, choices=range(10))
    parser.add_argument("--image", help="use your own PNG instead of a generated digit")
    parser.add_argument("--all", action="store_true", help="test all ten digits")
    parser.add_argument("--save", action="store_true",
                        help="write the drawing and the 28x28 input to PNGs")
    args = parser.parse_args()

    inference.load_sessions()
    print(f"{inference.runtime()} on {inference.provider()} "
          f"(artifacts: {weights.describe()})\n")

    if args.image:
        run_one(Image.open(args.image), label=args.image)
        return

    correct = dict.fromkeys(inference.model_names(), 0)
    digits = range(10) if args.all else [args.digit]
    for d in digits:
        img = draw_digit(d)
        arr, preds = run_one(img, label=f"drawn digit: {d}")
        if args.save:
            img.save(f"sample_drawing_{d}.png")
            preprocess.array_to_image(arr, scale=8).save(f"sample_input28_{d}.png")
        for name, cls in preds.items():
            correct[name] += int(cls == d)

    if args.all:
        print("Summary over the ten synthetic digits:")
        for name in inference.model_names():
            print(f"  {name}: {correct[name]}/10 correct")


def run_one(image, label=""):
    arr = preprocess.to_mnist_array(image, ink_is_dark=True)
    x = preprocess.to_input_array(image, ink_is_dark=True)

    print(f"--- {label} ---")
    preds = {}
    for name in inference.model_names():
        cls, probs, no_signal = inference.predict(name, x)
        preds[name] = cls
        if no_signal:
            print(f"  {name:3s} -> —   (no signal: all ten logits identical)")
            continue
        top = sorted(range(10), key=lambda i: -probs[i])[:3]
        detail = ", ".join(f"{i}: {probs[i] * 100:.1f}%" for i in top)
        print(f"  {name:3s} -> {cls}   (top-3: {detail})")
    print()
    return arr, preds


if __name__ == "__main__":
    main()
