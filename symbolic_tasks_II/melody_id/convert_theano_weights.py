"""
Convert the kernels of the pre-trained Theano/Lasagne models of the melody
identification repository (``nn_kernels_mozart.pkl``, ``nn_kernels_pop.pkl``)
to PyTorch checkpoints (``weights/melody_cnn_<name>.pt``).

The legacy files are Python 2 pickles (protocol 0) containing the list
``[W1, W2, switch]``, where ``W1`` and ``W2`` are plain numpy arrays. They can
be read with Python 3 (``encoding="latin1"``), without Theano or Lasagne.
Only load pickles that you trust!

Usage
-----
    python convert_theano_weights.py path/to/Symbolic-Melody-Identification
"""
import argparse
import os
import pickle

import torch

from melody_cnn import WEIGHTS_DIR, theano_kernels_to_state_dict


def convert(pkl_path: str, out_path: str):
    with open(pkl_path, "rb") as f:
        w1, w2, switch = pickle.load(f, encoding="latin1")
    state_dict = theano_kernels_to_state_dict(w1, w2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": state_dict, "switch": bool(switch),
                "source": os.path.basename(pkl_path)}, out_path)
    print(f"{pkl_path} -> {out_path}  W1 {tuple(w1.shape)}, W2 {tuple(w2.shape)}, switch={bool(switch)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="folder of the original melody identification repository")
    parser.add_argument("-o", "--out", default=WEIGHTS_DIR, help="output folder")
    args = parser.parse_args()
    for name in ("mozart", "pop"):
        convert(os.path.join(args.repo, f"nn_kernels_{name}.pkl"),
                os.path.join(args.out, f"melody_cnn_{name}.pt"))
