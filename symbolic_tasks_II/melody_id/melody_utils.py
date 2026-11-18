"""
Data handling for melody identification (Python 3 port of the helpers of the
original repository: ``data_handling.parse_data``, ``utils.pianoroll_utils``,
``melody_extractor.misc_tools`` and ``melody_extractor.graph_tools``).

Pipeline
--------
note array (pitch, onset, duration, soprano)
    -> piano roll of the score and of the melody (128 pitches x 8 pixels per beat)
    -> overlapping windows of 128 x 64 pixels (50% overlap)
    -> CNN (``melody_cnn.MelodyCNN``): melody probability of each pixel
    -> overlap-add of the windows: probability piano roll of the whole piece
    -> probability of each note (median of its pixels) and threshold -> melody notes

``melody_probabilities`` and ``identify_melody`` run the whole pipeline on a piece (without annotations).
"""
from __future__ import annotations

import bz2
import csv
import glob
import math
import os
import pickle
import tarfile
import urllib.request
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from partitura.utils import compute_pianoroll

from melody_cnn import WIN_HEIGHT, WIN_WIDTH

BEAT_DIV = 8  # pixels per beat
NOTE_DTYPE = [("pitch", "<i4"), ("onset", "<f4"), ("duration", "<f4"), ("soprano", "<i4")]
NOTE_DTYPE_NAMES = tuple(name for name, _ in NOTE_DTYPE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class _Ignored:
    """Placeholder for the music21 objects that are stored in some of the files."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _NoteArrayUnpickler(pickle.Unpickler):
    """Unpickler that only builds numpy arrays; music21 objects are replaced by placeholders."""

    def find_class(self, module, name):
        if module.startswith("music21"):
            return _Ignored
        if module.startswith("numpy") or module in ("builtins", "__builtin__", "copy_reg", "copyreg", "collections"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unexpected object in the file: {module}.{name}")


def load_piece(path: str) -> np.ndarray:
    """
    Load a piece of the dataset (``.pyc.bz`` files of the original repository).
    These are bz2-compressed Python 2 pickles of a numpy structured array with
    the fields ``pitch``, ``onset`` (in beats), ``duration`` (in beats) and
    ``soprano`` (1 if the note is part of the melody). Some files have
    additional fields with music21 objects (time and key signature), which are
    ignored. Only numpy objects are allowed when unpickling.
    """
    with bz2.BZ2File(path, "r") as f:
        note_array = _NoteArrayUnpickler(f, encoding="latin1").load()
    if isinstance(note_array, dict):  # some files store a dictionary of arrays
        columns = [np.asarray(note_array[name]) for name in NOTE_DTYPE_NAMES]
        out = np.zeros(len(columns[0]), dtype=NOTE_DTYPE)
        for name, column in zip(NOTE_DTYPE_NAMES, columns):
            out[name] = column
        return out
    return note_array[list(NOTE_DTYPE_NAMES)]


DATASET_URL = ("https://github.com/LIMUNIMI/Symbolic-Melody-Identification/raw/master/data/"
               "solo-accompaniment-dataset-pickled-nopop.tar.xz")
DATASET_ARCHIVE = "solo-accompaniment-dataset-pickled-nopop.tar.xz"


def get_mozart_dataset(dataset_dir: str, url: str = DATASET_URL) -> list:
    """
    Make sure that the Mozart dataset (38 movements of 13 piano sonatas, with the
    melody annotated by a pianist) is in ``dataset_dir/mozart``. If it is not,
    the archive of the companion repository of the paper is downloaded (4 MB;
    unless it is already in ``dataset_dir``) and the Mozart files are extracted.

    Returns the sorted list of paths of the pieces.
    """
    mozart_dir = os.path.join(dataset_dir, "mozart")
    if not glob.glob(os.path.join(mozart_dir, "*.pyc.bz")):
        os.makedirs(mozart_dir, exist_ok=True)
        archive = os.path.join(dataset_dir, DATASET_ARCHIVE)
        if not os.path.exists(archive):
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts = member.name.split("/")
                if member.isfile() and len(parts) == 3 and parts[1] == "mozart" and parts[2].endswith(".pyc.bz"):
                    with tar.extractfile(member) as src, open(os.path.join(mozart_dir, parts[2]), "wb") as dst:
                        dst.write(src.read())
    return sorted(glob.glob(os.path.join(mozart_dir, "*.pyc.bz")))


def load_challenge_piece(path: str):
    """
    Notes of a csv file of the challenge dataset (columns ``id``, ``onset_beat``, ``duration_beat``, ``pitch`` and, in
    the training set, ``melody``). Returns the list of the note ids and the note array of the dataset (fields
    ``pitch``, ``onset``, ``duration`` and ``soprano``, which is 0 if the file has no annotations), in the order of
    the rows of the file.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    notes = np.zeros(len(rows), dtype=NOTE_DTYPE)
    notes["pitch"] = [int(r["pitch"]) for r in rows]
    notes["onset"] = [float(r["onset_beat"]) for r in rows]
    notes["duration"] = [float(r["duration_beat"]) for r in rows]
    notes["soprano"] = [int(r.get("melody") or 0) for r in rows]
    return [r["id"] for r in rows], notes


def load_notes(path: str) -> np.ndarray:
    """Note array of a piece, from a ``.csv`` file of the challenge dataset or a ``.pyc.bz`` file of the Mozart dataset."""
    return load_challenge_piece(path)[1] if path.endswith(".csv") else load_piece(path)


def get_challenge_files(dataset_dir: str, subset: str = "train") -> list:
    """
    Sorted paths of the csv files of the ``train`` (or ``test``) folder of the challenge dataset, which has to be
    unzipped in ``dataset_dir/melody_id_challenge`` (the zip file is on Moodle).
    """
    folder = os.path.join(dataset_dir, "melody_id_challenge", subset)
    files = sorted(f for f in glob.glob(os.path.join(folder, "*.csv")) if not os.path.basename(f).startswith("."))
    if not files:
        raise FileNotFoundError(
            f"No csv files in {folder}. Download the zip file of the dataset from Moodle and unzip it in "
            f"{os.path.join(dataset_dir, 'melody_id_challenge')} (it has the folders 'train' and 'test')."
        )
    return files


# ---------------------------------------------------------------------------
# Piano rolls
# ---------------------------------------------------------------------------
def to_partitura_note_array(note_array: np.ndarray) -> np.ndarray:
    """
    Convert the note array of the dataset to a partitura-style note array
    (fields ``pitch``, ``onset_beat`` and ``duration_beat``), without grace notes
    (notes with duration 0), which are not shown in the piano roll.
    """
    notes = note_array[note_array["duration"] > 0]
    out = np.zeros(len(notes), dtype=[("pitch", "i4"), ("onset_beat", "f4"),
                                      ("duration_beat", "f4"), ("soprano", "i4")])
    out["pitch"] = notes["pitch"]
    out["onset_beat"] = notes["onset"]
    out["duration_beat"] = notes["duration"]
    out["soprano"] = notes["soprano"]
    return out


def make_score_pianoroll(note_array: np.ndarray, beat_div: int = BEAT_DIV):
    """
    Piano roll of the score, computed with partitura (``partitura.utils.compute_pianoroll``). It does not need the
    annotations of the melody, so it can be used with the pieces of the test set.

    Returns
    -------
    score : ndarray (128, n_columns)
        Binary piano roll with ``beat_div`` pixels per beat. The first column is the onset of the first note.
    idx : int ndarray (n_notes, 4)
        Position of each note in the piano roll: (pitch, first column, last column + 1, is melody). The last column is
        the annotation (``soprano``) of the note array.
    """
    notes = to_partitura_note_array(note_array)
    score, pr_idx = compute_pianoroll(notes, time_unit="beat", time_div=beat_div,
                                      note_separation=True, return_idxs=True,
                                      remove_silence=True, binary=True)
    score = score.toarray().astype(np.float32)
    idx = np.column_stack([pr_idx[:, 3], pr_idx[:, 1], pr_idx[:, 2], notes["soprano"]]).astype(int)
    return score, idx


def make_pianorolls(note_array: np.ndarray, beat_div: int = BEAT_DIV):
    """
    Piano rolls of the score and of the melody, computed with partitura
    (``partitura.utils.compute_pianoroll``).

    Returns
    -------
    score : ndarray (128, n_columns)
        Binary piano roll with ``beat_div`` pixels per beat. The first column is the onset of the first note.
    melody : ndarray (128, n_columns)
        Binary piano roll with only the melody notes (``soprano == 1``)
    idx : int ndarray (n_notes, 4)
        Position of each note in the piano roll: (pitch, first column, last column + 1, is melody).
    """
    score, idx = make_score_pianoroll(note_array, beat_div)
    notes = to_partitura_note_array(note_array)

    # melody piano roll: same time axis as the score piano roll
    is_melody = notes["soprano"] == 1
    melody_notes = notes[is_melody].copy()
    melody_notes["onset_beat"] -= notes["onset_beat"].min()
    melody, _ = compute_pianoroll(melody_notes, time_unit="beat", time_div=beat_div,
                                  note_separation=True, return_idxs=True,
                                  remove_silence=False, end_time=score.shape[1] / beat_div,
                                  binary=True)
    melody = melody.toarray().astype(np.float32)
    n = min(score.shape[1], melody.shape[1])
    score, melody = score[:, :n], melody[:, :n]
    return score, melody, idx


def labels_to_pianoroll(idx: np.ndarray, values: np.ndarray, n_columns: int) -> np.ndarray:
    """
    Piano roll (128, n_columns) in which each note has the given value (e.g., a boolean label or the probability
    of the note). ``idx`` is the output of ``make_pianorolls``.
    """
    pianoroll = np.zeros((WIN_HEIGHT, n_columns), dtype=np.float32)
    for (pitch, start, end, _), value in zip(idx, values):
        pianoroll[pitch, start:end] = value
    return pianoroll


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------
def excerpt_notes(note_array: np.ndarray, start_beat: float = 0.0, n_beats: Optional[float] = None,
                  melody_only: bool = False) -> np.ndarray:
    """
    Notes of an excerpt of a piece, as a partitura-style note array (fields ``pitch``,
    ``onset_beat``, ``duration_beat`` and ``id``) with the onsets relative to ``start_beat``.
    Notes that go beyond the end of the excerpt are shortened. If ``melody_only``
    is True, only the melody notes are returned.
    """
    notes = to_partitura_note_array(note_array)
    end_beat = np.inf if n_beats is None else start_beat + n_beats
    keep = (notes["onset_beat"] >= start_beat) & (notes["onset_beat"] < end_beat)
    if melody_only:
        keep &= notes["soprano"] == 1
    notes = notes[keep]
    notes["duration_beat"] = np.minimum(notes["duration_beat"], end_beat - notes["onset_beat"])
    notes["onset_beat"] -= start_beat
    out = np.zeros(len(notes), dtype=[("pitch", "i4"), ("onset_beat", "f4"), ("duration_beat", "f4"), ("id", "U12")])
    for name in ("pitch", "onset_beat", "duration_beat"):
        out[name] = notes[name]
    out["id"] = [f"n{i}" for i in range(len(notes))]  # partitura needs note ids for the synthesis
    return out


def synthesize_notes(notes: np.ndarray, bpm: float = 120, samplerate: int = 22050) -> np.ndarray:
    """
    Synthesize a note array with partitura: with fluidsynth (sampled piano sound) if it
    is installed, and with additive synthesis otherwise.
    """
    try:
        from partitura.utils.fluidsynth import HAS_FLUIDSYNTH, synthesize_fluidsynth
        if HAS_FLUIDSYNTH:
            return synthesize_fluidsynth(notes, samplerate=samplerate, bpm=bpm)
    except Exception as error:  # e.g., fluidsynth installed but the soundfont is not available
        print(f"Fluidsynth is not available ({error}); using additive synthesis")
    from partitura.utils.synth import synthesize
    return synthesize(notes, samplerate=samplerate, bpm=bpm, envelope_fun="linear", harmonic_dist="shepard")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def split_windows(pianoroll: np.ndarray, win_width: int = WIN_WIDTH) -> np.ndarray:
    """
    Split a piano roll into windows (n_windows, 1, 128, win_width) with 50%
    overlap. The first window starts ``win_width // 2`` columns before the
    piece, so that every column of the piece is covered by two windows.
    """
    hop = win_width // 2
    n = pianoroll.shape[1]
    n_windows = int(math.ceil(n / hop)) + 1
    padded = np.zeros((pianoroll.shape[0], (n_windows + 1) * hop), dtype=pianoroll.dtype)
    padded[:, hop:hop + n] = pianoroll
    windows = np.stack([padded[:, i * hop:i * hop + win_width] for i in range(n_windows)])
    return windows[:, None]


def merge_windows(windows: np.ndarray, n_columns: int) -> np.ndarray:
    """
    Inverse of ``split_windows``: overlap-add of the windows (the average of the
    two windows that cover each column). Returns an array (128, n_columns).
    """
    win_width = windows.shape[-1]
    hop = win_width // 2
    out = np.zeros((windows.shape[2], (len(windows) + 1) * hop), dtype=np.float32)
    for i, window in enumerate(windows[:, 0]):
        out[:, i * hop:i * hop + win_width] += window / 2.0
    return out[:, hop:hop + n_columns]


@torch.no_grad()
def predict_pianoroll(model, pianoroll: np.ndarray, device=None, batch_size: int = 64) -> np.ndarray:
    """Melody probability of each pixel of a piano roll (128, n_columns)."""
    if device is None:
        device = next(model.parameters()).device
    windows = split_windows(pianoroll)
    out = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i:i + batch_size]).to(device)
        out.append(model.predict(batch).cpu().numpy())
    return merge_windows(np.concatenate(out), pianoroll.shape[1])


# ---------------------------------------------------------------------------
# From pixels to notes
# ---------------------------------------------------------------------------
def note_probabilities(prob: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Probability of each note: the median of the probabilities of its pixels."""
    return np.array([np.median(prob[p, s:e]) for p, s, e, _ in idx])


def cluster_threshold(note_probs: np.ndarray) -> float:
    """
    Threshold between the "melody" and the "not melody" notes of a piece (Section 4.1 of the paper):
    hierarchical single-linkage clustering of the note probabilities in two clusters, and the threshold
    is the largest value of the lowest cluster. In one dimension, the two clusters of single-linkage
    are separated by the largest gap between consecutive (sorted) values.
    """
    values = np.sort(np.asarray(note_probs, dtype=float))
    if len(values) < 2:
        return 0.5
    cut = np.argmax(np.diff(values))
    return float(values[cut])


def melody_graph(note_probs: np.ndarray, idx: np.ndarray, threshold: float):
    """
    Graph of Algorithm 1 of the paper: one node for each note with probability above the threshold, plus a
    start and an end node. A note is connected to the notes with the earliest onset after its end.

    Returns
    -------
    candidates : indices of the notes above the threshold, sorted by onset (the nodes of the graph)
    probs : their probabilities
    first, last : for each node, the successors are the nodes ``first[k]:last[k]`` (none if ``first[k] == n``,
        i.e., the node is connected to the end node)
    n_start : the start node is connected to the nodes ``0:n_start`` (the ones with the earliest onset)
    """
    start, end = idx[:, 1], idx[:, 2]
    candidates = np.nonzero(note_probs >= threshold)[0]
    candidates = candidates[np.argsort(start[candidates], kind="stable")]
    onsets, probs = start[candidates], note_probs[candidates]
    n = len(candidates)
    first = np.searchsorted(onsets, end[candidates], side="left")
    last = np.where(first < n, np.searchsorted(onsets, onsets[np.minimum(first, max(n - 1, 0))], side="right"), n)
    n_start = int(np.searchsorted(onsets, onsets[0], side="right")) if n > 0 else 0
    return candidates, probs, first, last, n_start


def shortest_path_steps(probs: np.ndarray, first: np.ndarray, last: np.ndarray, n_start: int):
    """
    Shortest path (with weights ``-probability``) from the start node of ``melody_graph``. All the edges go
    forward in time, so the nodes are processed in order of onset (equivalent to Bellman-Ford here).
    Generator: yields ``(k, best, previous)`` after processing node ``k`` (``k = -1`` for the initial state),
    where ``best[j]`` is the highest total probability of a path from the start node to ``j`` and
    ``previous[j]`` is the node before ``j`` in that path. The arrays are modified in place.
    """
    n = len(probs)
    best = np.full(n, -np.inf)
    previous = np.full(n, -1)
    best[:n_start] = probs[:n_start]
    yield -1, best, previous
    for k in range(n):
        if best[k] > -np.inf:  # reachable from the start node
            for j in range(first[k], last[k]):
                if best[k] + probs[j] > best[j]:
                    best[j], previous[j] = best[k] + probs[j], k
        yield k, best, previous


def monophonic_melody(note_probs: np.ndarray, idx: np.ndarray, threshold: float) -> np.ndarray:
    """
    Strictly monophonic melody (Section 4.2 of the paper, Algorithm 1): the shortest path from the start to
    the end node of ``melody_graph``, i.e., the sequence of non-overlapping notes above the threshold with the
    highest total probability. Returns a boolean array with the notes of the melody.
    """
    candidates, probs, first, last, n_start = melody_graph(note_probs, idx, threshold)
    is_melody = np.zeros(len(note_probs), dtype=bool)
    if len(candidates) == 0:
        return is_melody
    for _, best, previous in shortest_path_steps(probs, first, last, n_start):
        pass
    k = int(np.argmax(np.where(first >= len(candidates), best, -np.inf)))  # best note connected to the end node
    while k != -1:
        is_melody[candidates[k]] = True
        k = previous[k]
    return is_melody


def melody_probabilities(note_array: np.ndarray, model):
    """
    Probability that each note of a piece is part of the melody, with the CNN ``model`` (the piece does not need
    annotations). Returns the probabilities and ``idx``, the positions of the notes in the piano roll
    (see ``make_score_pianoroll``).
    """
    pianoroll, idx = make_score_pianoroll(note_array)
    return note_probabilities(predict_pianoroll(model, pianoroll), idx), idx


def identify_melody(note_array: np.ndarray, model) -> np.ndarray:
    """
    Melody of a piece with the steps of the paper: the probability of each note (``melody_probabilities``), the
    threshold of the clustering (``cluster_threshold``), and the monophonic melody (``monophonic_melody``).
    Returns a boolean array with the notes of the melody.
    """
    probabilities, idx = melody_probabilities(note_array, model)
    return monophonic_melody(probabilities, idx, cluster_threshold(probabilities))


def note_level_scores(true_labels: np.ndarray, predicted: np.ndarray):
    """Precision, recall and F1-measure of the melody notes."""
    tp = float(np.sum(true_labels & predicted))
    precision = tp / predicted.sum() if predicted.sum() > 0 else 0.0
    recall = tp / true_labels.sum() if true_labels.sum() > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
MELODY_COLOR = "#d62728"
SCORE_CMAP = LinearSegmentedColormap.from_list("score", ["white", "black"])
MELODY_CMAP = LinearSegmentedColormap.from_list("melody", ["white", MELODY_COLOR])


def plot_pianoroll(pianoroll: np.ndarray, ax=None, cmap=SCORE_CMAP, title: Optional[str] = None,
                   beat_div: int = BEAT_DIV, pitch_range=None, xlabel: Optional[str] = None,
                   vmin: float = 0.0, vmax: float = 1.0):
    """
    Plot a piano roll (128, n_columns) with the time in beats and the MIDI pitch
    on the vertical axis. Values are shown from white (0) to the color of
    ``cmap`` (1). ``pitch_range`` = (lowest, highest) pitch to show.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    low, high = pitch_range if pitch_range is not None else (0, pianoroll.shape[0] - 1)
    ax.imshow(pianoroll[low:high + 1], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest", extent=(0, pianoroll.shape[1] / beat_div, low - 0.5, high + 0.5))
    ax.set_ylabel("MIDI pitch")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title, fontsize=10)
    return ax
