#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates a submission on the challenge server, computing the accuracy of the piece identification.

Each file of the test set is a sequence of notes in which the pianist switches from one piece to another. The sequence
(the note array of the MIDI file, sorted by onset and then by pitch) is split into consecutive windows of
``QUERY_LENGTH`` = 50 notes, which are the queries. The query ``k`` of a file consists of the notes ``50 k`` to
``50 k + 49``, and the correct piece of a query is the piece that is being played at its last note. A window with fewer
than 50 notes at the end of the file is not a query.

The submission is a compressed numpy file (``np.savez_compressed``) with an array for each file of the test set, named
as the file without the extension, with the predicted piece (a string, the name of the score without the extension) of
each of its queries, in order (an empty string if there is no prediction):

    np.savez_compressed("results.npz", mi2026_fingerprinting_test_000=np.array(["Mozart_K331_1st-mov", ...]), ...)

The score is the accuracy (between 0 and 1, the higher the better): the fraction of the queries of the test set for which
the piece is correct. The queries without a prediction (a missing file or array that is too short) are wrong.
"""

import argparse

import numpy as np

QUERY_LENGTH = 50


def load_submission(fn: str) -> dict:
    """Load a submission (or the ground truth): dictionary name of the file -> array with the piece of each query."""
    with np.load(fn, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def accuracy(submission: dict, target: dict) -> float:
    """Fraction of the queries of the target with the correct piece in the submission."""
    correct = total = 0
    for name, pieces in target.items():
        total += len(pieces)
        if name in submission:
            predicted = np.asarray(submission[name]).astype(str).ravel()
            n = min(len(predicted), len(pieces))
            correct += int((predicted[:n] == pieces[:n]).sum())
    return correct / total


def print_results(score: float, n_queries: int):
    """Print the score of the challenge."""
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    target = load_submission(args.target)
    print_results(accuracy(load_submission(args.submission), target), len(target))
