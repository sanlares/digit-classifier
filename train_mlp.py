"""Train the multilayer perceptron on MNIST.

    python train_mlp.py                    # defaults below
    python train_mlp.py --epochs 20 --lr 1e-3

Defaults are deliberately modest (15 hidden units, 5 epochs, lr 1e-4) — the
point of this model is to be the small baseline the CNN is measured against,
not to squeeze out accuracy.
"""

import argparse
import time

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import NetMLP, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mlp.pth")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    train = datasets.MNIST(root="./data", train=True, download=True,
                           transform=torchvision.transforms.ToTensor())
    test = datasets.MNIST(root="./data", train=False, download=True,
                          transform=torchvision.transforms.ToTensor())
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    net = NetMLP(28 * 28, args.hidden, 10).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"training on {device}: {args.epochs} epochs, lr={args.lr}, "
          f"batch={args.batch_size}, hidden={args.hidden}, seed={args.seed}")

    started = time.time()
    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            # NetMLP.forward flattens with x.size(0), so a ragged final batch
            # is handled without a hardcoded batch size.
            loss = criterion(net(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"  epoch {epoch + 1}/{args.epochs}  mean loss {running / len(loader):.4f}")

    net.eval()
    x_test, y_test = next(iter(DataLoader(test, batch_size=len(test))))
    with torch.no_grad():
        pred = net(x_test.to(device)).cpu().argmax(1)
    accuracy = (pred == y_test).float().mean().item()

    torch.save(net.state_dict(), args.out)
    print(f"\ntest accuracy {accuracy:.4f} · {time.time() - started:.0f}s · saved {args.out}")


if __name__ == "__main__":
    main()
