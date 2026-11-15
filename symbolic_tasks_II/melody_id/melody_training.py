"""
Training of the melody identification CNN (Section 4.3 of the paper), following the original code:

* Data: windows of 128 x 64 pixels (50% overlap) of the piano rolls of the pieces of the training set.
* Augmentation: 50% more windows, in which the melody is transposed down two octaves, down one octave or up
  one octave.
* Loss: root of the squared error between the (masked) output and the target, normalized by the number of active
  pixels of the input.
* Optimizer: AdaDelta with learning rate 1. Glorot uniform initialization.
* Early stopping: the training stops after ``patience`` epochs without improvement of the validation loss, and
  the best weights are restored.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from melody_utils import load_piece, make_pianorolls, split_windows


def split_pieces(files, test_seed: int = 1992, val_seed: int = 34):
    """
    Split of the pieces (never of the windows, since windows of the same piece overlap) in training, validation
    and test sets, as in the original code (``trainer.py``): a random group split with 20% of the pieces for the
    test set, and then another one with 20% of the remaining pieces for the validation set.

    Returns three lists of files (train, validation, test). For the 38 pieces of the Mozart dataset: 24, 6, 8.
    """
    files = sorted(files)
    pieces = np.arange(len(files))
    remaining, test = next(GroupShuffleSplit(test_size=0.2, random_state=test_seed).split(pieces, groups=pieces))
    train, validation = next(GroupShuffleSplit(test_size=0.2, random_state=val_seed).split(remaining, groups=remaining))
    return ([files[i] for i in remaining[train]], [files[i] for i in remaining[validation]], [files[i] for i in test])


def pianoroll_windows(files):
    """Windows (n, 1, 128, 64) of the piano rolls of the score (inputs) and of the melody (targets), as uint8."""
    inputs, targets = [], []
    for f in files:
        score, melody, _ = make_pianorolls(load_piece(f))
        inputs.append(split_windows(score).astype(np.uint8))
        targets.append(split_windows(melody).astype(np.uint8))
    return np.concatenate(inputs), np.concatenate(targets)


def augment_melody(inputs, targets, fraction: float = 0.5, shifts=(-24, -12, 12), seed: int = 0):
    """
    Data augmentation: copies of ``fraction`` of the windows (chosen at random), in which the melody is transposed
    by a random number of semitones (the same for the whole window). The accompaniment does not change.
    Returns the original and the new windows.
    """
    rng = np.random.default_rng(seed)
    new_inputs, new_targets = [], []
    for i in rng.integers(0, len(inputs), size=int(len(inputs) * fraction)):
        melody = targets[i, 0]
        shift = int(rng.choice(shifts))
        shifted = np.zeros_like(melody)
        if shift > 0:
            shifted[shift:] = melody[:-shift]
        else:
            shifted[:shift] = melody[-shift:]
        if shifted.sum() != melody.sum():  # part of the melody would be out of the range of the piano roll
            continue
        accompaniment = np.where(melody > 0, 0, inputs[i, 0])
        new_inputs.append(np.maximum(accompaniment, shifted)[None, None])
        new_targets.append(shifted[None, None])
    return np.concatenate([inputs] + new_inputs), np.concatenate([targets] + new_targets)


def melody_loss(output: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """sqrt(sum of squared errors / number of active pixels of the input) (``trainer.py`` of the original code)."""
    return torch.sqrt(((output - target) ** 2).sum() / inputs.sum().clamp(min=1.0))


def train_model(model, train_data, validation_data, device, max_epochs: int = 200, patience: int = 20,
                batch_size: int = 100, learning_rate: float = 1.0, seed: int = 0, verbose: bool = True,
                checkpoint_path=None):
    """
    Train the model (in place) with early stopping, and restore the weights of the best epoch.

    Parameters
    ----------
    train_data, validation_data : tuples (inputs, targets) of windows, as returned by ``pianoroll_windows``
    checkpoint_path : if given, the best weights so far are saved there after every improvement (so that a long
        training can be interrupted)

    Returns
    -------
    history : dict with the training and validation losses of each epoch, and the best epoch
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.reset_parameters()
    model.to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95, eps=1e-6)  # as in Lasagne

    def batches(data, order):
        for start in range(0, len(order), batch_size):
            b = np.sort(order[start:start + batch_size])
            yield (torch.from_numpy(data[0][b]).float().to(device), torch.from_numpy(data[1][b]).float().to(device))

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_loss, best_state = np.inf, None
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        train_loss = []
        for x, y in batches(train_data, rng.permutation(len(train_data[0]))):
            optimizer.zero_grad()
            loss = melody_loss(model(x), y, x)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = [melody_loss(model(x), y, x).item() for x, y in batches(validation_data, np.arange(len(validation_data[0])))]
        history["train_loss"].append(float(np.mean(train_loss)))
        history["val_loss"].append(float(np.mean(val_loss)))
        if history["val_loss"][-1] < best_loss:
            best_loss, best_state = history["val_loss"][-1], copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            if checkpoint_path is not None:
                torch.save({"state_dict": best_state, "switch": False, "history": history}, checkpoint_path)
        if verbose:
            print(f"Epoch {epoch + 1:3d}: train loss {history['train_loss'][-1]:.4f}, validation loss "
                  f"{history['val_loss'][-1]:.4f} ({time.time() - start_time:.0f} s)"
                  + (" *" if history["best_epoch"] == epoch else ""))
        if epoch - history["best_epoch"] >= patience:
            if verbose:
                print(f"No improvement in {patience} epochs: stop.")
            break
    model.load_state_dict(best_state)
    model.eval()
    return history
