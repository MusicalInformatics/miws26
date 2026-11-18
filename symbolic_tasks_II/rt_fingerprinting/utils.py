"""Helper functions: loading of scores and performances as Partitura note arrays, file names, and hash tables."""

import csv
import os
import pickle
import re
import zipfile

import numpy as np
import numpy.lib.recfunctions as rfn
import partitura as pt

MIDI_EXTENSIONS = (".mid", ".midi")


_score_cache = {}
CACHE_DIR = None  # if set: folder where the note arrays of the scores are stored (loading the repeats is slow)


def load_note_array(
    path: str, tempo: float = 120.0, unfold: bool = False
) -> np.ndarray:
    """
    Load a score (e.g. MusicXML) or a performance (MIDI) as a Partitura structured note array, sorted by onset.

    The note array always has the fields ``pitch``, ``onset_sec`` and ``duration_sec``. For the performances, the
    times are those of the MIDI file. For the scores, the times are computed from the score time in quarter notes,
    with a fixed ``tempo`` (in beats per minute).

    With ``unfold`` the repeats of a score are written out with ``unfold_part_maximal`` of partitura. This gives the
    notes the ids that the alignments of the match files use for scores with repeats (e.g. ``n15-2``).
    """
    if os.path.splitext(path)[1].lower() in MIDI_EXTENSIONS:
        performance = pt.load_performance(path)
        for ppart in performance.performedparts:
            ppart.sustain_pedal_threshold = 127  # the pedal does not lengthen the notes
        notes = performance.note_array()
    else:
        key = (path, tempo, unfold)
        cached = (
            None
            if CACHE_DIR is None
            else os.path.join(
                CACHE_DIR,
                f"{piece_name(path)}_{'unfolded' if unfold else 'folded'}_{tempo:g}.npy",
            )
        )
        if key not in _score_cache and cached is not None and os.path.exists(cached):
            _score_cache[key] = np.load(cached)
        if key not in _score_cache:
            score = pt.load_score(path)
            notes = score.note_array()
            if unfold:
                notes = pt.utils.music.note_array_from_part_list(
                    [pt.score.unfold_part_maximal(p) for p in score.parts]
                )
            seconds_per_quarter = 60.0 / tempo
            notes = rfn.append_fields(
                notes,
                ["onset_sec", "duration_sec"],
                [
                    notes["onset_quarter"] * seconds_per_quarter,
                    notes["duration_quarter"] * seconds_per_quarter,
                ],
                usemask=False,
            )
            _score_cache[key] = notes[np.lexsort((notes["pitch"], notes["onset_sec"]))]
            if cached is not None:
                os.makedirs(CACHE_DIR, exist_ok=True)
                np.save(cached, _score_cache[key])
        return _score_cache[key]
    return notes[np.lexsort((notes["pitch"], notes["onset_sec"]))]


def list_files(folder: str, extensions: tuple) -> list:
    """
    Sorted paths of the files of a folder with one of the extensions. The hidden files (their name starts with a dot) are
    ignored: macOS writes AppleDouble files (``._name.mid``) next to the files on FAT and exFAT drives.
    """
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(tuple(extensions)) and not f.startswith(".")
    )


def piece_name(path: str) -> str:
    """Name of the file without folder and extension, e.g. ``Mozart_K331_1st-mov``."""
    return os.path.splitext(os.path.basename(path))[0]


def piece_of_performance(path: str) -> str:
    """Name of the piece of a performance of the Vienna 4x22 corpus: ``Mozart_K331_1st-mov_p01.mid`` -> ``Mozart_K331_1st-mov``."""
    return re.sub(r"_p\d+$", "", piece_name(path))


def performances_of_scores(performance_files: list, score_files: list) -> list:
    """The performance files that are performances of one of the scores (by the name of the file)."""
    names = {piece_name(f) for f in score_files}
    return [f for f in performance_files if piece_of_performance(f) in names]


def load_segments(path: str) -> np.ndarray:
    """Segments of a sequence (csv with ``start``, ``end``, ``piecename``, ``transposition``) as a structured array."""
    return np.atleast_1d(
        np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding=None, comments=None
        )
    )


def load_note_map(path: str) -> np.ndarray:
    """Note map of a sequence (``seq_*_notes.csv``): for each note, its piece, performance, original note id, ..."""
    return np.atleast_1d(
        np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding=None, comments=None
        )
    )


def load_sequences(folder: str) -> list:
    """
    ``(note array, segments, note map)`` of each sequence (MIDI file) of a folder, with the segments in
    ``<name>_segments.csv`` and, if it exists, the note map in ``<name>_notes.csv`` (otherwise the note map is ``None``).
    """
    files = list_files(folder, MIDI_EXTENSIONS)
    sequences = []
    for f in files:
        base = os.path.splitext(f)[0]
        note_map = (
            load_note_map(base + "_notes.csv")
            if os.path.exists(base + "_notes.csv")
            else None
        )
        sequences.append(
            (load_note_array(f), load_segments(base + "_segments.csv"), note_map)
        )
    return sequences


def should_unfold(path: str, unfold) -> bool:
    """``unfold`` is a bool (for all the scores) or a collection of the paths of the scores that are unfolded."""
    return unfold if isinstance(unfold, bool) else path in unfold


def load_scores(score_files: list, unfold=False) -> dict:
    """Note arrays of the scores (``load_note_array``): piece -> note array. See ``should_unfold`` for ``unfold``."""
    return {
        piece_name(path): load_note_array(path, unfold=should_unfold(path, unfold))
        for path in score_files
    }


def dataset_scores(data_dir: str):
    """
    Score files of the dataset of the challenge (``<data_dir>/scores/scores.csv``) and the set of the score files that are
    unfolded (the repeats are written out). The result can be given to ``load_scores`` and the hash tables.
    """
    with open(os.path.join(data_dir, "scores", "scores.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    files = [os.path.join(data_dir, "scores", row["file"]) for row in rows]
    return files, {
        os.path.join(data_dir, "scores", row["file"])
        for row in rows
        if int(row["unfold"])
    }


def missing_files(data_dir: str) -> list:
    """Files of ``<data_dir>/metadata.csv`` (columns ``file`` and ``size``) that are missing or have another size."""
    metadata = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata):
        return ["metadata.csv"]
    with open(metadata, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        r["file"]
        for r in rows
        if not os.path.exists(os.path.join(data_dir, r["file"]))
        or os.path.getsize(os.path.join(data_dir, r["file"])) != int(r["size"])
    ]


def unpack_data(zip_path: str, data_dir: str = "data") -> str:
    """
    Unpack the zip file of the dataset (``scores``, ``train``, ``test`` and ``metadata.csv``) into ``data_dir``.

    If ``data_dir`` already exists and has all the files of ``metadata.csv``, nothing is unpacked. If files are missing
    (e.g., an interrupted unzipping), the zip file is unpacked again. Returns ``data_dir``.
    """
    if os.path.isdir(data_dir) and not missing_files(data_dir):
        print(f"{data_dir} has all the files of the dataset")
        return data_dir
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"{zip_path} does not exist. Download the dataset from Moodle and give its path."
        )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    missing = missing_files(data_dir)
    if missing:
        raise RuntimeError(
            f"{len(missing)} files are missing in {data_dir} after unpacking, e.g., {missing[:3]}"
        )
    print(f"Dataset unpacked in {data_dir}")
    return data_dir


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # utils.py: <repo>/symbolic_tasks_II/rt_fingerprinting


def relocate(path: str) -> str:
    """
    The ``performances.csv`` of the datasets of ``challenge_dev`` have absolute paths of the computer where they were
    made. If a file is not at that path, look for it at the same place of this copy of the repository (the part of the
    path after ``<repository name>/``).
    """
    marker = os.sep + os.path.basename(REPO_ROOT) + os.sep
    if not os.path.exists(path) and marker in path:
        candidate = os.path.join(REPO_ROOT, path.split(marker, 1)[1])
        if os.path.exists(candidate):
            return candidate
    return path


def load_dataset_sources(folder: str):
    """
    Scores and performances of a dataset of sequences, from its ``performances.csv``.

    Returns the list of score files, the set of score files that are unfolded, and the dictionary
    performance name -> match file.
    """
    table = np.atleast_1d(
        np.genfromtxt(
            os.path.join(folder, "performances.csv"),
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
            comments=None,
        )
    )
    score_files = sorted({relocate(row["score_file"]) for row in table})
    unfold = {relocate(row["score_file"]) for row in table if row["unfold"]}
    return score_files, unfold, {row["performance"]: row["match_file"] for row in table}


def save_hash_table(hash_table: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(hash_table, f)


def load_hash_table(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
