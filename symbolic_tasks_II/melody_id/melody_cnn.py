"""
PyTorch version of the melody identification CNN of

    Simonetta, Cancino-Chacón, Ntalampiras & Widmer (2019). "A Convolutional
    Approach to Melody Line Identification in Symbolic Scores". ISMIR 2019.

The original implementation (https://github.com/LIMUNIMI/Symbolic-Melody-Identification)
was written in Python 2 with Theano and Lasagne. The network is a convolutional
"auto-encoder with tied weights", whose input and output are piano roll windows
of the same size (128 pitches x 64 time steps):

    x ──conv1──> z1 ──conv2──> z2 ──sigmoid──> a2
                                                 │
    out = x * sigmoid( conv1ᵀ( conv2ᵀ( a2 * σ'(z2) ) ) )     (masking with the input)

* ``conv1``: 1 -> K channels, kernel (32, 16), no bias, no nonlinearity
* ``conv2``: K -> K channels, kernel (32, 16), no bias, sigmoid
* The "decoder" has no parameters of its own. Lasagne's ``InverseLayer`` is the
  *gradient* of a layer with respect to its input, so it is a transposed
  convolution that **shares the weights** of the encoder layer and (for conv2)
  also contains the derivative of the sigmoid.
* The output is multiplied by the input (``MASKED = True`` in the original
  settings), so that only pixels where there is a note can be melody pixels.

Pre-trained Theano weights can be reused: see ``convert_theano_weights.py``.
The only subtlety is that Theano/Lasagne computes a *true convolution*
(``flip_filters=True``), while PyTorch computes a cross-correlation, so the
kernels have to be flipped along the pitch and time axes.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

WIN_HEIGHT = 128  # pitches
WIN_WIDTH = 64  # time steps (8 per beat)

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class MelodyCNN(nn.Module):
    """
    Fully convolutional network that estimates, for each pixel of a piano roll,
    the probability of belonging to the melody.

    Parameters
    ----------
    num_kernels : int
        Number of channels of the convolutional layers (``num_kernel0`` in the original code).
    kernel_size : tuple(int, int)
        (pitch, time) size of the kernels (``kernel_h0``, ``kernel_w0``).
    masking : bool
        If True, the output is multiplied by the input piano roll.
    """

    def __init__(self, num_kernels: int = 21, kernel_size=(32, 16), masking: bool = True):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = tuple(kernel_size)
        self.masking = masking
        self.conv1 = nn.Conv2d(1, num_kernels, self.kernel_size, bias=False)
        self.conv2 = nn.Conv2d(num_kernels, num_kernels, self.kernel_size, bias=False)
        # In the original code, the network can be trained to predict (input - output)
        # instead of the output ("target switching"). Both released models do not use it.
        self.switch = False

    def reset_parameters(self):
        """Glorot (Xavier) uniform initialization of the kernels, as in the original code."""
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, 1, pitch, time)
            Binary piano roll windows (height >= 2 * kernel_h - 1 and width >= 2 * kernel_w - 1).

        Returns
        -------
        Tensor (batch, 1, pitch, time) with the melody probability of each pixel.
        """
        a2 = torch.sigmoid(self.conv2(self.conv1(x)))

        # "Inverse" layers (tied weights). For conv2 the gradient also contains the
        # derivative of its sigmoid, sigma'(z2) = a2 * (1 - a2).
        h = F.conv_transpose2d(a2 * a2 * (1.0 - a2), self.conv2.weight)
        out = torch.sigmoid(F.conv_transpose2d(h, self.conv1.weight))

        if self.switch:
            out = x - out  # only used if the model was trained with target switching
        if self.masking:
            out = x * out
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference (the network has no dropout/batch-norm, but this is clearer)."""
        was_training = self.training
        self.eval()
        out = self(x)
        self.train(was_training)
        return out


def get_device() -> torch.device:
    """CUDA if available, then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def theano_kernels_to_state_dict(w1, w2):
    """
    Convert the kernels of the Lasagne model (arrays with shape (out, in, h, w))
    to a PyTorch state dict. The kernels are flipped, since Theano uses true
    convolutions, whereas PyTorch uses cross-correlations.
    """
    w1 = torch.as_tensor(w1, dtype=torch.float32).flip(-1, -2).contiguous()
    w2 = torch.as_tensor(w2, dtype=torch.float32).flip(-1, -2).contiguous()
    return {"conv1.weight": w1, "conv2.weight": w2}


def load_pretrained(name: str = "mozart", weights_dir: Optional[str] = None, device=None) -> MelodyCNN:
    """
    Load one of the models trained by the authors of the paper
    (``name`` is "mozart" or "pop", i.e., the dataset used for training).
    """
    path = os.path.join(weights_dir or WEIGHTS_DIR, f"melody_cnn_{name}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    w1 = checkpoint["state_dict"]["conv1.weight"]
    model = MelodyCNN(num_kernels=w1.shape[0], kernel_size=tuple(w1.shape[2:]))
    model.load_state_dict(checkpoint["state_dict"])
    model.switch = bool(checkpoint.get("switch", False))
    model.eval()
    return model.to(device) if device is not None else model
