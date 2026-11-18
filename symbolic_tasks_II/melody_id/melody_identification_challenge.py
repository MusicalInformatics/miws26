"""
Evaluates a submission on the challenge server, computing the note-level F1-score of the melody notes.

The submission is a compressed numpy file (``np.savez_compressed``) with an array for each file of the test set, named
as the file without the extension. The array has one value for each note of the file, in the order of the rows of the
file: 1 if the note is predicted to be part of the melody, and 0 otherwise:

    np.savez_compressed("results.npz", test_000=np.array([0, 1, 1, 0, ...]), test_001=np.array([...]), ...)

For each piece, the F1-score is computed for the melody notes (the harmonic mean of the precision and the recall of
the melody notes). The score is the mean of the F1-scores of the pieces (between 0 and 1, the higher the better), so
that all the pieces have the same weight, whatever their length. The notes that are not in the submission (an array
that is too short) are predicted as not melody, and a piece that is not in the submission has an F1-score of 0.
"""
import argparse

import numpy as np


def load_submission(fn: str) -> dict:
    """Load a submission (or the ground truth): dictionary name of the file -> array with the label of each note."""
    with np.load(fn, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def f1_melody(target: np.ndarray, prediction: np.ndarray = None) -> float:
    """F1-score of the melody notes of a piece. ``target`` and ``prediction``: arrays with the label of each note."""
    target = np.asarray(target).astype(int).ravel() == 1
    predicted = np.zeros(len(target), dtype=bool)
    if prediction is not None:
        prediction = np.asarray(prediction).astype(float).ravel()[: len(target)] == 1  # missing notes: not melody
        predicted[: len(prediction)] = prediction
    true_positive = float(np.sum(target & predicted))
    precision = true_positive / predicted.sum() if predicted.any() else 0.0
    recall = true_positive / target.sum() if target.any() else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(submission: dict, target: dict) -> float:
    """Mean F1-score of the melody notes of the pieces of the target."""
    return float(np.mean([f1_melody(labels, submission.get(piece)) for piece, labels in target.items()]))


def print_results(value: float):
    """Print the score of the challenge."""
    print(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    print_results(score(load_submission(args.submission), load_submission(args.target)))
