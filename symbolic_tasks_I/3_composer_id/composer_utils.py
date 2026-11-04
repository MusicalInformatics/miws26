"""
Utilities for the composer identification notebook
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """
    The best device to run the neural network: a CUDA GPU (e.g., in Google Colab), the GPU of
    Apple Silicon Macs (the MPS backend of PyTorch) or, if there is none, the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


COMPOSER_CLASSES = {
    "Claude Debussy": 0,
    "Franz Liszt": 1,
    "Franz Schubert": 2,
    "Johann Sebastian Bach": 3,
    "Ludwig van Beethoven": 4,
    "Maurice Ravel": 5,
    "Robert Schumann": 6,
    "Sergei Rachmaninoff": 7,
    "Wolfgang Amadeus Mozart": 8,
}

CLASSES_COMPOSER = {
    0: "Claude Debussy",
    1: "Franz Liszt",
    2: "Franz Schubert",
    3: "Johann Sebastian Bach",
    4: "Ludwig van Beethoven",
    5: "Maurice Ravel",
    6: "Robert Schumann",
    7: "Sergei Rachmaninoff",
    8: "Wolfgang Amadeus Mozart",
}


def encode_composer(composers: Union[np.ndarray, str, pd.DataFrame]) -> np.ndarray:

    if isinstance(composers, str):

        return np.array(
            [COMPOSER_CLASSES[composers]],
            dtype=int,
        )

    else:

        composer_indices = np.array(
            [COMPOSER_CLASSES[cmp] for cmp in composers],
            dtype=int,
        )

        return composer_indices


def decode_composer(composer_index: Union[np.ndarray, int]) -> np.ndarray:

    if isinstance(composer_index, np.ndarray):
        return np.array(
            [CLASSES_COMPOSER[idx] for idx in composer_index],
            dtype="U256",
        )

    else:
        return np.array(
            [CLASSES_COMPOSER[composer_index]],
            dtype="U256",
        )


def segment_array(array: np.ndarray, window_length: int) -> np.ndarray:
    """
    Segments a 2D numpy array into non-overlapping windows of a specified length.
    Pads the end of the array with zeros along the first axis if necessary.

    Parameters
    -----------
    array : np.ndaarray
        2D array of shape (length, 88).

    window_length: int
        Length of each segment window.

    Returns
    -------
    segmented_array: np.ndarray
        Segmented 3D array of shape (n_windows, window_length, 88).
    """
    length = array.shape[0]
    n_windows = np.ceil(length / window_length).astype(int)
    padded_length = n_windows * window_length

    # Pad the array with zeros along the first axis
    padded_array = np.zeros((padded_length, array.shape[1]))
    padded_array[:length, :] = array

    # Reshape into non-overlapping windows
    segmented_array = padded_array.reshape(n_windows, window_length, array.shape[1])

    return segmented_array


def predict_piece(
    pianoroll: np.ndarray,
    segment_length: int,
    cnn: nn.Module,
    device: Optional[torch.device] = None,
) -> int:

    if device is None:
        device = get_device()
        print(f"Using device: {device}")

    cnn.to(device)

    cnn.eval()

    segmented_pr = segment_array(
        array=pianoroll,
        window_length=segment_length,
    )

    segmented_pr_tensor = (
        torch.tensor(
            segmented_pr,
            dtype=torch.float32,
        )
        .unsqueeze(1)
        .to(device)
    )

    output = cnn(segmented_pr_tensor)

    _, preds = torch.max(output.data, 1)

    preds = preds.cpu().numpy()

    # Agregate using the most common class
    unique_labels, counts = np.unique(
        preds,
        return_counts=True,
    )
    mode_index = counts.argmax()
    mode = unique_labels[mode_index]
    return mode
