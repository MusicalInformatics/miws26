"""
Train the melody identification CNN of the ISMIR 2019 paper from scratch (random initialization, the weights of the
authors are NOT used for training) on the Mozart dataset, and evaluate it on held-out pieces.

The paper evaluates with a 10-fold cross-validation over the whole dataset. Here we use instead a single, fixed
split of the 13 works (sonatas) in training, validation and test sets:

* The split is made by *work*, not by movement or by window: movements of the same sonata share style, key and
  material, and windows of the same piece overlap, so splitting at a lower level would leak information.
* The three sets cover the early (K. 279-284), middle (K. 330-333) and late (K. 457, 475, 533) sonatas.

The training follows the paper and the original code: windows of 128 x 64 pixels, 50% additional examples with the
melody transposed (-2, -1, +1 octaves), loss on the masked output, AdaDelta (learning rate 1), Glorot uniform
initialization, and early stopping after 20 epochs without improvement of the validation loss.

Usage (from this folder):

    python train_melody_id.py                       # trains and evaluates
    python train_melody_id.py --evaluate-only       # evaluates an existing checkpoint
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import warnings

import numpy as np
import torch

from melody_cnn import MelodyCNN, get_device, load_pretrained
from melody_training import augment_melody, pianoroll_windows, train_model
from melody_utils import (
    cluster_threshold,
    get_mozart_dataset,
    load_piece,
    make_pianorolls,
    monophonic_melody,
    note_level_scores,
    note_probabilities,
    predict_pianoroll,
    to_partitura_note_array,
)
from skyline import skyline

warnings.filterwarnings("ignore", module="partitura")

HERE = os.path.dirname(os.path.abspath(__file__))

# Köchel numbers of the works in each set (13 works, 38 movements: 23 / 6 / 9)
TEST_WORKS = (280, 332, 457)  # early, middle, late
VALIDATION_WORKS = (283, 331)  # early, middle
# training: 279, 281, 282, 284 (early), 330, 333 (middle), 475, 533 (late)


def work_of(path: str) -> int:
    """Köchel number of a file such as ``kv280_2.pyc.bz``."""
    return int(re.match(r"kv(\d+)_", os.path.basename(path)).group(1))


def split_by_work(files):
    """Training, validation and test lists of files, with a fixed assignment of the works."""
    files = sorted(files)
    test = [f for f in files if work_of(f) in TEST_WORKS]
    validation = [f for f in files if work_of(f) in VALIDATION_WORKS]
    train = [f for f in files if f not in test and f not in validation]
    return train, validation, test


def piece_name(path: str) -> str:
    return os.path.basename(path).replace(".pyc.bz", "")


def evaluate_piece(note_array, model):
    """Note-level (precision, recall, F1) of the skyline and of the CNN with the steps of the paper."""
    piece_notes = to_partitura_note_array(note_array)
    score, _, idx = make_pianorolls(note_array)
    true = piece_notes["soprano"] == 1
    probs = note_probabilities(predict_pianoroll(model, score), idx)
    threshold = cluster_threshold(probs)
    return {
        "CNN, threshold 0.5": note_level_scores(true, probs >= 0.5),
        "CNN, clustering threshold": note_level_scores(true, probs >= threshold),
        "CNN, clustering threshold + shortest path": note_level_scores(true, monophonic_melody(probs, idx, threshold)),
    }, note_level_scores(true, skyline(piece_notes))


def evaluate_files(files, models, results_csv=None, label=""):
    """Evaluate the skyline and each model ``{name: model}`` on the files, print the means and (optionally) save."""
    rows = []
    for f in files:
        note_array = load_piece(f)
        for name, model in models.items():
            scores, skyline_scores = evaluate_piece(note_array, model)
            if name == next(iter(models)):
                rows.append((piece_name(f), "Skyline", *skyline_scores))
            for method, s in scores.items():
                rows.append((piece_name(f), f"{name}: {method}", *s))
    methods = list(dict.fromkeys(r[1] for r in rows))
    print(f"\n{label} ({len(files)} pieces)")
    print(f"{'':70s} precision  recall     F1")
    for m in methods:
        p, r, f1 = np.mean([r_[2:] for r_ in rows if r_[1] == m], axis=0)
        print(f"{m:70s} {p:9.3f} {r:7.3f} {f1:6.3f}")
    if results_csv is not None:
        with open(results_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["piece", "method", "precision", "recall", "f1"])
            writer.writerows(rows)
        print(f"Per-piece results saved to {results_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=os.path.join(HERE, "datasets"))
    parser.add_argument("--output", default=os.path.join(HERE, "weights", "melody_cnn_mozart_reproduced.pt"))
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--max-epochs", type=int, default=1000, help="upper bound; early stopping ends before")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-only", action="store_true", help="skip the training, load --output")
    args = parser.parse_args()

    device = get_device()
    files = get_mozart_dataset(args.dataset_dir)
    train_files, validation_files, test_files = split_by_work(files)
    print(f"Device: {device}")
    for name, subset in [("train", train_files), ("validation", validation_files), ("test", test_files)]:
        works = sorted({work_of(f) for f in subset})
        print(f"{name:10s}: {len(subset):2d} pieces from K. {', '.join(map(str, works))}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if not args.evaluate_only:
        print("\nBuilding the piano roll windows...")
        train_data = pianoroll_windows(train_files)
        validation_data = pianoroll_windows(validation_files)
        n_train = len(train_data[0])
        train_data = augment_melody(*train_data, seed=args.seed)
        print(f"Training windows: {n_train} (+{len(train_data[0]) - n_train} augmented), "
              f"validation windows: {len(validation_data[0])}")

        model = MelodyCNN()  # random (Glorot uniform) initialization: training from scratch
        history = train_model(model, train_data, validation_data, device, max_epochs=args.max_epochs,
                              patience=args.patience, batch_size=args.batch_size, seed=args.seed,
                              checkpoint_path=args.output)
        torch.save({"state_dict": model.state_dict(), "switch": False, "history": history}, args.output)
        print(f"\nBest epoch: {history['best_epoch'] + 1}, validation loss {min(history['val_loss']):.4f}. "
              f"Weights saved to {args.output}")

    name = os.path.basename(args.output).replace("melody_cnn_", "").replace(".pt", "")
    reproduced = load_pretrained(name, weights_dir=os.path.dirname(args.output), device=device)
    models = {"Reproduced": reproduced}

    evaluate_files(validation_files, models, os.path.join(args.results_dir, "validation_results.csv"), "Validation")
    evaluate_files(test_files, models, os.path.join(args.results_dir, "test_results.csv"), "Test")

    # Reference: the weights released by the authors. They were obtained with their own cross-validation, so
    # the test pieces were probably part of their training data: these numbers are an upper reference only.
    authors = load_pretrained("mozart", device=device)
    evaluate_files(test_files, {"Authors' weights": authors},
                   os.path.join(args.results_dir, "test_results_authors.csv"),
                   "Test, weights of the authors (probably trained on these pieces)")


if __name__ == "__main__":
    main()
