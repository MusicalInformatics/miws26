#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline submission for the real-time fingerprinting challenge for Musical Informatics WS26: the online fingerprinting
with histogram bins (``OnlineFingerprinting``) with the hash table of the scores, and windows of 50 notes.

    python Baseline_Fingerprinting.py -i path_to_the_data_directory -o baseline_results_fingerprinting.csv
"""

import argparse
import csv
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="partitura.*")

from fingerprinting import OnlineFingerprinting, build_hash_table_v2
from fingerprinting_challenge import QUERY_LENGTH, accuracy, load_submission
from utils import dataset_scores, list_files, load_note_array


def identify(midi_file: str, hash_table, query_length: int = QUERY_LENGTH) -> list:
    """Piece of each query (window of ``query_length`` notes, without overlap) of the sequence of a MIDI file."""
    notes = load_note_array(midi_file)
    model = OnlineFingerprinting(hash_table, window_size=query_length, n1=5, n2=5)
    model.run_offline(notes)
    return [
        ranking[0] if ranking else ""
        for ranking in model.rankings[: len(notes) // query_length]
    ]


def predictions_of(files: list, hash_table) -> dict:
    """Dictionary name of the file (without extension) -> array with the piece of each query."""
    predictions = {}
    for i, midi_file in enumerate(files):
        name = os.path.splitext(os.path.basename(midi_file))[0]
        predictions[name] = np.array(identify(midi_file, hash_table), dtype=str)
        print(f"{i + 1}/{len(files)} {name}")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time fingerprinting")
    parser.add_argument(
        "--datadir", "-i", help="path to the input files", default=None, type=str
    )
    parser.add_argument(
        "--outfn",
        "-o",
        help="Output file with results",
        type=str,
        default="fingerprinting_results.npz",
    )
    args = parser.parse_args()
    if args.datadir is None:
        raise ValueError("No data directory given")

    score_files, unfold = dataset_scores(args.datadir)
    print(f"Building the hash table of {len(score_files)} scores...")
    hash_table = build_hash_table_v2(score_files, n1=5, n2=5, unfold=unfold)

    # The method does not need training: this evaluation on the training set shows how well it works. If you train a
    # method for your submission, replace this block.
    train_files = list_files(os.path.join(args.datadir, "train"), (".mid",))
    train_gt = os.path.join(args.datadir, "train", "fingerprinting_train_gt.npz")
    if train_files and os.path.exists(train_gt):
        score = accuracy(
            predictions_of(train_files, hash_table), load_submission(train_gt)
        )
        print("#" * 35)
        print("Piece identification (queries of 50 notes), training set\n")
        print(f"   Accuracy: {100 * score:.2f}")
        print("#" * 35)

    predictions = predictions_of(
        list_files(os.path.join(args.datadir, "test"), (".mid",)), hash_table
    )

    # This part will only save results for the test set!
    np.savez_compressed(args.outfn, **predictions)
