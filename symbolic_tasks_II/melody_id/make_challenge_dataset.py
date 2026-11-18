"""
Create the dataset of the melody identification challenge from the Mozart dataset of the paper.

Each piece is written as a csv file with one row per note (the information of a Partitura note array):

    id, onset_beat, duration_beat, pitch, melody        (melody = 1 if the note belongs to the melody, else 0)

* ``train/``: the pieces of the training set, with the ``melody`` annotations. The files keep the name of the piece
  (e.g. ``kv279_1.csv``: Köchel number of the sonata and movement), so that the movements of the same sonata can be
  kept together when the training set is split for validation.
* ``test/``: the pieces of the test set, WITHOUT the ``melody`` column, with anonymous names (``test_000.csv``).

The annotations of the test set and the names of the test pieces are written in a separate folder (the "solution"),
which is not part of the dataset for the students.

The split is by sonata, as in ``train_melody_id.py``: the test set are the sonatas K. 280, 332 and 457 (9 pieces),
and the training set are the other 29 pieces (the validation set of ``train_melody_id.py`` is part of the training set).

Usage (from this folder):

    python make_challenge_dataset.py --dataset-dir datasets --output-dir datasets/melody_id_challenge
"""
from __future__ import annotations

import argparse
import csv
import os
import warnings
import zipfile

import numpy as np

from melody_utils import get_mozart_dataset, load_piece, to_partitura_note_array
from train_melody_id import TEST_WORKS, piece_name, work_of

warnings.filterwarnings("ignore", module="partitura")

FIELDS = ["id", "onset_beat", "duration_beat", "pitch"]


def notes_table(path: str) -> np.ndarray:
    """Note array of the piece (sorted by onset and pitch) with a unique ``id`` for each note."""
    notes = to_partitura_note_array(load_piece(path))
    notes = notes[np.lexsort((notes["pitch"], notes["onset_beat"]))]
    ids = np.array([f"n{i:04d}" for i in range(len(notes))])
    table = np.zeros(len(notes), dtype=[("id", "U8")] + [(n, notes.dtype[n]) for n in notes.dtype.names])
    table["id"] = ids
    for name in notes.dtype.names:
        table[name] = notes[name]
    return table


def write_csv(path: str, table: np.ndarray, with_melody: bool, melody_only: bool = False):
    """Write the notes; with ``melody_only`` only the id and the annotation (for the solution file)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if melody_only:
            writer.writerow(["id", "melody"])
            writer.writerows([[n["id"], int(n["soprano"])] for n in table])
            return
        writer.writerow(FIELDS + (["melody"] if with_melody else []))
        for n in table:
            row = [n["id"], np.format_float_positional(n["onset_beat"], trim="-"),
                   np.format_float_positional(n["duration_beat"], trim="-"), int(n["pitch"])]
            writer.writerow(row + ([int(n["soprano"])] if with_melody else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="datasets", help="folder with the original Mozart dataset")
    parser.add_argument("--output-dir", default="datasets/melody_id_challenge", help="folder of the students' dataset")
    parser.add_argument("--solution-dir", default="datasets/melody_id_challenge_solution")
    parser.add_argument("--zip", default=None, help="zip file of the dataset of the students (train and test)")
    args = parser.parse_args()

    files = sorted(get_mozart_dataset(args.dataset_dir))
    test_files = [f for f in files if work_of(f) in TEST_WORKS]
    train_files = [f for f in files if f not in test_files]

    for d in (os.path.join(args.output_dir, "train"), os.path.join(args.output_dir, "test"),
              os.path.join(args.solution_dir, "test")):
        os.makedirs(d, exist_ok=True)

    gt_train, gt_test = {}, {}  # label of each note of each piece (in the order of the rows), the format of the submissions
    for f in train_files:
        table = notes_table(f)
        write_csv(os.path.join(args.output_dir, "train", piece_name(f) + ".csv"), table, with_melody=True)
        gt_train[piece_name(f)] = table["soprano"].astype(np.int8)

    with open(os.path.join(args.solution_dir, "test_pieces.csv"), "w", newline="") as mapping:
        writer = csv.writer(mapping)
        writer.writerow(["file", "original_piece"])
        for i, f in enumerate(test_files):
            name = f"test_{i:03d}"
            table = notes_table(f)
            write_csv(os.path.join(args.output_dir, "test", name + ".csv"), table, with_melody=False)
            write_csv(os.path.join(args.solution_dir, "test", name + ".csv"), table, with_melody=True, melody_only=True)
            writer.writerow([name + ".csv", piece_name(f)])
            gt_test[name] = table["soprano"].astype(np.int8)

    np.savez_compressed(os.path.join(args.output_dir, "train", "melody_identification_train_gt.npz"), **gt_train)
    np.savez_compressed(os.path.join(args.solution_dir, "melody_identification_test_gt.npz"), **gt_test)

    if args.zip:  # what the students download: the training set (with the annotations) and the test set (without them)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for folder in ("train", "test"):
                for name in sorted(n for n in os.listdir(os.path.join(args.output_dir, folder)) if not n.startswith(".")):
                    z.write(os.path.join(args.output_dir, folder, name), os.path.join(folder, name))
        print(f"Zip file: {args.zip} ({os.path.getsize(args.zip) / 1e6:.1f} MB)")

    print(f"{len(train_files)} training pieces and {len(test_files)} test pieces written to {args.output_dir}")
    print(f"Solution of the test set (do not share): {args.solution_dir}")


if __name__ == "__main__":
    main()
