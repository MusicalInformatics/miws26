#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline submission for the melody identification challenge for Musical Informatics WS26: the CNN of Simonetta et al.
(2019), trained from scratch on the training pieces of the dataset (``weights/melody_cnn_mozart_reproduced.pt``), with
the steps of the paper to get a melody line: the probability of each note, the threshold of the clustering, and the
shortest path in the graph of the notes.

    python Baseline_MelodyIdentification.py -i path_to_the_data_directory -o baseline_results_melody_identification.npz
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", module="partitura")

from melody_cnn import get_device, load_pretrained
from melody_identification_challenge import f1_melody
from melody_utils import identify_melody, load_challenge_piece


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Melody identification")
    parser.add_argument("--datadir", "-i", help="path to the input files", default=None, type=str)
    parser.add_argument("--outfn", "-o", help="Output file with results", type=str,
                        default="melody_identification_results.npz")
    args = parser.parse_args()
    if args.datadir is None:
        raise ValueError("No data directory given")

    model = load_pretrained("mozart_reproduced", device=get_device())

    # The CNN was trained on the training set, so this evaluation on the training set is optimistic. If you train a
    # method for your submission, replace this block.
    train_files = sorted(f for f in os.listdir(os.path.join(args.datadir, "train")) if f.endswith(".csv")
                         and not f.startswith("."))  # (hidden files: ._name.csv on exFAT drives)
    scores = []
    for i, name in enumerate(train_files):
        ids, notes = load_challenge_piece(os.path.join(args.datadir, "train", name))
        predicted = identify_melody(notes, model)
        scores.append(f1_melody(notes["soprano"], predicted))
        print(f"{i + 1}/{len(train_files)} {name}: F1 = {scores[-1]:.3f}")
    print("#" * 35)
    print("Melody identification, training set\n")
    print(f"   Mean F1-score of the melody notes: {100 * np.mean(scores):.2f}")
    print("#" * 35)

    # This part will only save results for the test set!
    test_files = sorted(f for f in os.listdir(os.path.join(args.datadir, "test")) if f.endswith(".csv")
                        and not f.startswith("."))
    predictions = {}
    for i, name in enumerate(test_files):
        _, notes = load_challenge_piece(os.path.join(args.datadir, "test", name))
        predictions[os.path.splitext(name)[0]] = identify_melody(notes, model).astype(np.int8)  # a label for each note
        print(f"{i + 1}/{len(test_files)} {name}")
    np.savez_compressed(args.outfn, **predictions)
