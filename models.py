"""Model architectures — training and export only.

Nothing in the serving path imports this module: at runtime the models are ONNX
graphs executed by onnxruntime (see `inference.py`), so torch is not installed
in the container. This file exists to train the models and export them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 28
NUM_CLASSES = 10
HIDDEN_UNITS = 15


class NetMLP(torch.nn.Module):
    """784 -> 15 (ReLU) -> 10 logits."""

    def __init__(self, input_features=IMAGE_SIZE * IMAGE_SIZE,
                 size_hidden=HIDDEN_UNITS, n_output=NUM_CLASSES):
        super(NetMLP, self).__init__()
        self.hidden1 = torch.nn.Linear(input_features, size_hidden)
        self.hidden2 = torch.nn.Linear(size_hidden, n_output)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden1(x))
        # No activation here: these are logits. CrossEntropyLoss applies the
        # log-softmax itself, and squashing them would throw away the negative
        # scores the network uses to rule classes out.
        return self.hidden2(x)


class NetMLPLegacy(torch.nn.Module):
    """The superseded MLP, kept only to reproduce its baseline. Do not train.

    Two defects, both visible below: `out` is never called by `forward` (so it
    never received a gradient and holds its random initialisation), and the
    ReLU on `hidden2` clamps all ten logits to zero whenever the network wants
    to score every class negatively — which happens for ~32% of the test set,
    leaving argmax to pick class 0 by tie-break.
    """

    def __init__(self, input_features=IMAGE_SIZE * IMAGE_SIZE,
                 size_hidden=HIDDEN_UNITS, n_output=NUM_CLASSES):
        super(NetMLPLegacy, self).__init__()
        self.hidden1 = torch.nn.Linear(input_features, size_hidden)
        self.hidden2 = torch.nn.Linear(size_hidden, n_output)
        self.out = torch.nn.Linear(n_output, n_output)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return x


class NetCNN(nn.Module):
    """Reduced LeNet: conv(6) -> pool -> conv(16) -> pool -> 400-120-84-10."""

    def __init__(self):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 5 * 5 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_torch_model(cls, checkpoint, device=None):
    """Instantiate `cls`, load `checkpoint` into it, put it in eval mode."""
    model = cls()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    if device is not None:
        model.to(device)
    return model.eval()
