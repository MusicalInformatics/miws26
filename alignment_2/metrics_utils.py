#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for the `alignment_metrics` notebook: computing, plotting and evaluating
alignments at the note level (F-score) and at the time level (beat error).
"""
import contextlib
import io
import os
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union

import ipywidgets as widgets
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import partitura as pt
from IPython.display import display
from matchmaker import Matchmaker
from matchmaker.utils.eval import (
    TOLERANCES_IN_BEATS,
    evaluate_alignment,
    resolve_gt,
    spectral_arc_length,
    transfer_positions,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from parangonar.evaluate import fscore_alignments
from partitura.performance import Performance, PerformedPart

# From the alignment_1 folder
from alignment import fast_dynamic_time_warping, greedy_note_alignment
from helper import dummy_linear_alignment, interactive_figure

PathLike = Union[str, os.PathLike]

CORRECT_COLOR = "tab:green"
WRONG_COLOR = "tab:red"
MISSED_COLOR = "tab:blue"
UNALIGNED_COLOR = "darkorange"

NOTE_LEVEL_METHODS = ["linear", "DTW (pitch class)", "DTW (piano roll)"]

# The score followers of Matchmaker
ONLINE_METHODS = {
    "PitchHMM": "pthmm",
    "OLTW (Arzt)": "arzt",
    "OLTW (Dixon)": "dixon",
    "Outer-product HMM": "outerhmm",
}
ONLINE_COLORS = {
    "PitchHMM": "tab:red",
    "OLTW (Arzt)": "tab:blue",
    "OLTW (Dixon)": "tab:green",
    "Outer-product HMM": "tab:purple",
}

_ERROR_DIR = None


def _match_fn(mec_variation_dir: PathLike, performance_name: str) -> str:
    return os.path.join(mec_variation_dir, "alignments", "match", f"{performance_name}.match")


def _midi_fn(mec_variation_dir: PathLike, performance_name: str) -> str:
    return os.path.join(mec_variation_dir, "performances", "midi", f"{performance_name}.mid")


########################################################################
# Note-level alignments
########################################################################
def alignment_to_sets(
    alignment: List[dict],
) -> Tuple[Set[Tuple[str, str]], Set[str], Set[str]]:
    """Split an alignment into the set of matched (score_id, performance_id) pairs, the deleted score notes and the inserted performance notes"""
    matches = {
        (str(a["score_id"]), str(a["performance_id"]))
        for a in alignment
        if a["label"] == "match"
    }
    deletions = {str(a["score_id"]) for a in alignment if a["label"] == "deletion"}
    insertions = {str(a["performance_id"]) for a in alignment if a["label"] == "insertion"}
    return matches, deletions, insertions


def _draw_piano_roll(ax, note_array, onset_key, duration_key, color, unaligned_ids, selected_ids):
    """Draw the notes as rectangles"""
    for note in note_array:
        if str(note["id"]) not in selected_ids:
            continue
        ax.add_patch(
            Rectangle(
                (note[onset_key], note["pitch"] - 0.4),
                max(note[duration_key], 0.05),
                0.8,
                facecolor=UNALIGNED_COLOR if str(note["id"]) in unaligned_ids else color,
                edgecolor="none",
            )
        )


def plot_note_alignment(
    score_notes: np.ndarray,
    performance_notes: np.ndarray,
    prediction: List[dict],
    ground_truth: Optional[List[dict]] = None,
    beat_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Figure:
    """
    Plot a note-level alignment between the performance (top) and the score (bottom).

    If only `prediction` is given, all the pairs of matched notes are drawn in gray.
    If `ground_truth` is also given, the pairs are colored:
    green: in the prediction and in the ground truth (correct)
    red: only in the prediction (wrong)
    blue (dashed): only in the ground truth (missed)

    score_notes, performance_notes: note arrays
    beat_range: only plot the notes of the score in this range (in beats), and the
        performed notes that are close to the notes aligned to them
    """
    pred_matches, pred_del, pred_ins = alignment_to_sets(prediction)
    if ground_truth is None:
        gt_matches, gt_del, gt_ins = set(), set(), set()
    else:
        gt_matches, gt_del, gt_ins = alignment_to_sets(ground_truth)
    all_matches = pred_matches | gt_matches

    score_ids = {str(n["id"]) for n in score_notes}
    perf_ids = {str(n["id"]) for n in performance_notes}
    onset_beat = {str(n["id"]): n["onset_beat"] for n in score_notes}
    onset_sec = {str(n["id"]): n["onset_sec"] for n in performance_notes}
    pitch_of = {
        str(n["id"]): n["pitch"] for n in list(score_notes) + list(performance_notes)
    }

    # Select the notes that we plot
    if beat_range is not None:
        score_ids = {i for i in score_ids if beat_range[0] <= onset_beat[i] < beat_range[1]}
        partners = {p for s, p in all_matches if s in score_ids}
        t0 = min(onset_sec[p] for p in partners) - 0.3
        t1 = max(onset_sec[p] for p in partners) + 0.3
        perf_ids = {i for i in perf_ids if t0 <= onset_sec[i] <= t1}
        beat_lim = beat_range
        sec_lim = (t0, t1)
    else:
        beat_lim = (0, score_notes["onset_beat"].max() + 1)
        sec_lim = (0, performance_notes["onset_sec"].max() + 0.5)

    fig = Figure(figsize=figsize)
    ax_p, ax_s = fig.subplots(2, 1, gridspec_kw={"hspace": 0.55})
    _draw_piano_roll(
        ax_s, score_notes, "onset_beat", "duration_beat", "0.45",
        pred_del | gt_del, score_ids,
    )
    _draw_piano_roll(
        ax_p, performance_notes, "onset_sec", "duration_sec", "0.45",
        pred_ins | gt_ins, perf_ids,
    )

    # Draw the lines between the aligned notes
    for s, p in sorted(all_matches):
        if s not in score_ids or p not in perf_ids:
            continue
        if ground_truth is None:
            color, style, width, zorder = "0.2", "-", 0.7, 2
        elif (s, p) in pred_matches and (s, p) in gt_matches:
            color, style, width, zorder = CORRECT_COLOR, "-", 0.8, 2
        elif (s, p) in pred_matches:
            color, style, width, zorder = WRONG_COLOR, "-", 1.6, 4
        else:
            color, style, width, zorder = MISSED_COLOR, "--", 1.6, 3
        fig.add_artist(
            ConnectionPatch(
                xyA=(onset_sec[p], pitch_of[p] - 0.4), coordsA=ax_p.transData,
                xyB=(onset_beat[s], pitch_of[s] + 0.4), coordsB=ax_s.transData,
                color=color, linestyle=style, linewidth=width, zorder=zorder,
            )
        )

    pitches = [pitch_of[i] for i in score_ids | perf_ids]
    for ax, xlim, xlabel in [
        (ax_s, beat_lim, "Score time (beats)"),
        (ax_p, sec_lim, "Performance time (s)"),
    ]:
        ax.set_xlim(xlim)
        ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
        ax.set_xlabel(xlabel)
    ax_s.set_ylabel("Score\nMIDI pitch")
    ax_p.set_ylabel("Performance\nMIDI pitch")
    # performance axis on top, so that the labels are not crossed by the lines between the notes
    ax_p.xaxis.tick_top()
    ax_p.xaxis.set_label_position("top")

    handles = [Line2D([], [], color="0.2", label="aligned") if ground_truth is None else
               Line2D([], [], color=CORRECT_COLOR, label="correct (in prediction and ground truth)")]
    if ground_truth is not None:
        handles += [
            Line2D([], [], color=WRONG_COLOR, linewidth=1.6, label="wrong (only in prediction)"),
            Line2D([], [], color=MISSED_COLOR, linestyle="--", linewidth=1.6, label="missed (only in ground truth)"),
        ]
    handles += [mpatches.Patch(color=UNALIGNED_COLOR, label="note without partner (deletion / insertion)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title if title is not None else "")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.14)
    return fig


def interactive_note_alignment(
    score_notes: np.ndarray,
    performance_notes: np.ndarray,
    alignments: Dict[str, List[dict]],
    ground_truth: Optional[List[dict]] = None,
) -> None:
    """
    Sliders to look at a window of the note alignment. If there are several alignments, a
    dropdown selects which one is plotted. Without a ground truth, the plotted alignment is
    the ground truth.
    """
    max_beat = score_notes["onset_beat"].max()

    def figure(start_beat, length, method=None):
        name = next(iter(alignments)) if method is None else method
        return plot_note_alignment(
            score_notes,
            performance_notes,
            alignments[name],
            ground_truth,
            beat_range=(start_beat, start_beat + length),
            title="Ground truth alignment" if ground_truth is None else f"{name} vs. ground truth",
            figsize=(12, 5.5),
        )

    controls = {}
    if len(alignments) > 1:
        controls["method"] = widgets.Dropdown(
            options=list(alignments), value=list(alignments)[-1], description="Method"
        )
    controls["start_beat"] = widgets.FloatSlider(
        value=0, min=0, max=max_beat, step=1, description="Beat", continuous_update=False
    )
    controls["length"] = widgets.FloatSlider(
        value=8, min=2, max=16, step=1, description="Length", continuous_update=False
    )
    display(widgets.VBox(list(controls.values())), interactive_figure(figure, controls))


def compute_alignment(method: str, score, performance) -> List[dict]:
    """
    Compute a note-level alignment between a score and a performance with DTW, as in the
    `Symbolic_Music_Alignment` notebook.

    method: "linear", "DTW (pitch class)" or "DTW (piano roll)"
    """
    snote_array = score.note_array()
    pnote_array = performance.note_array()
    if method == "DTW (pitch class)":
        score_feat, sidx = pt.utils.music.compute_pitch_class_pianoroll(
            score, time_unit="beat", time_div=8, return_idxs=True,
            binary=True, note_separation=True,
        )
        perf_feat, pidx = pt.utils.music.compute_pitch_class_pianoroll(
            performance, time_unit="sec", time_div=8, return_idxs=True,
            binary=True, note_separation=True,
        )
        reference_features, performance_features = score_feat.T, perf_feat.T
    else:
        score_feat, sidx = pt.utils.music.compute_pianoroll(
            score, time_unit="beat", time_div=8, return_idxs=True,
            piano_range=True, binary=True, note_separation=True,
        )
        perf_feat, pidx = pt.utils.music.compute_pianoroll(
            performance, time_unit="sec", time_div=8, return_idxs=True,
            piano_range=True, binary=True, note_separation=True,
        )
        reference_features = score_feat.toarray().T
        performance_features = perf_feat.toarray().T

    if method == "linear":
        warping_path = dummy_linear_alignment(reference_features, performance_features)
    else:
        warping_path = fast_dynamic_time_warping(
            X=reference_features, Y=performance_features, metric="cityblock"
        )

    return greedy_note_alignment(
        warping_path=warping_path,
        idx1=sidx, note_array1=snote_array,
        idx2=pidx, note_array2=pnote_array,
    )


########################################################################
# Note-level metrics
########################################################################
def _draw_tp_fp_fn(n_tp: int, n_fp: int, n_fn: int, ax, title: str = "") -> None:
    # ground truth = TP + FN, prediction = TP + FP
    ax.barh(1, n_tp, color=CORRECT_COLOR, label="TP (correct)")
    ax.barh(1, n_fn, left=n_tp, color=MISSED_COLOR, label="FN (missed)")
    ax.barh(0, n_tp, color=CORRECT_COLOR)
    ax.barh(0, n_fp, left=n_tp, color=WRONG_COLOR, label="FP (wrong)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Prediction", "Ground truth"])
    ax.set_xlabel("Number of pairs")
    ax.set_title(title)
    # legend below the axes (in the space that tight_layout leaves free with rect)
    handles, labels = ax.get_legend_handles_labels()
    ax.figure.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0), ncol=3, frameon=False)


def tp_fp_fn_figure(n_tp: int, n_fp: int, n_fn: int, title: str = "") -> Figure:
    """Bars with the pairs of the ground truth (TP + FN) and of the prediction (TP + FP)"""
    fig = Figure(figsize=(8, 2.8))
    _draw_tp_fp_fn(n_tp, n_fp, n_fn, fig.subplots(), title)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def fscore_table(prediction: List[dict], ground_truth: List[dict]) -> pd.DataFrame:
    """Precision, recall and F-score of each type (and of all types together)"""
    rows = {}
    for name, types in [
        ("match", ["match"]),
        ("insertion", ["insertion"]),
        ("deletion", ["deletion"]),
        ("all", ["match", "insertion", "deletion"]),
    ]:
        p, r, f, n_pred, n_gt = fscore_alignments(
            prediction, ground_truth, types, return_numbers=True
        )
        rows[name] = {
            "predicted": n_pred, "ground truth": n_gt,
            "precision": p, "recall": r, "F-score": f,
        }
    return pd.DataFrame(rows).T


def perturb_alignment(
    ground_truth: List[dict], fraction_wrong: float, fraction_missed: float, seed: int = 0
) -> List[dict]:
    """
    Simulate errors on an alignment:
    - wrong matches: the partners of a fraction of the matches are rotated
    - missed matches: a fraction of the matches become a deletion and an insertion
    """
    rng = np.random.RandomState(seed)
    matches = [a for a in ground_truth if a["label"] == "match"]
    others = [a for a in ground_truth if a["label"] != "match"]
    order = rng.permutation(len(matches))
    n_wrong = int(round(fraction_wrong * len(matches)))
    n_missed = int(round(fraction_missed * len(matches)))
    wrong = [matches[i] for i in order[:n_wrong]]
    missed = [matches[i] for i in order[n_wrong : n_wrong + n_missed]]
    correct = [matches[i] for i in order[n_wrong + n_missed :]]

    # rotate the performed notes of the wrong matches
    partners = [a["performance_id"] for a in wrong]
    partners = partners[1:] + partners[:1]
    wrong = [{**a, "performance_id": p} for a, p in zip(wrong, partners)]
    # missed matches become a deletion and an insertion
    missed_alignment = []
    for a in missed:
        missed_alignment.append({"label": "deletion", "score_id": a["score_id"]})
        missed_alignment.append({"label": "insertion", "performance_id": a["performance_id"]})
    return others + correct + wrong + missed_alignment


def interactive_perturbed_metrics(ground_truth: List[dict]) -> None:
    """Sliders to choose the fraction of wrong and missed matches, and plot the resulting metrics"""

    def figure(fraction_wrong, fraction_missed) -> Figure:
        perturbed = perturb_alignment(ground_truth, fraction_wrong, fraction_missed)
        table = fscore_table(perturbed, ground_truth)
        fig = Figure(figsize=(12, 3.8))
        axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6]})

        n_tp = len(alignment_to_sets(perturbed)[0] & alignment_to_sets(ground_truth)[0])
        n_pred = len(alignment_to_sets(perturbed)[0])
        n_gt = len(alignment_to_sets(ground_truth)[0])
        _draw_tp_fp_fn(n_tp, n_pred - n_tp, n_gt - n_tp, axes[0], title="Pairs of type match")

        x = np.arange(len(table))
        for k, metric in enumerate(["precision", "recall", "F-score"]):
            axes[1].bar(x + (k - 1) * 0.27, table[metric], 0.27, label=metric)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(table.index)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc="lower right", bbox_to_anchor=(1, -0.32), ncol=3, frameon=False)
        axes[1].set_title("Metrics by type")
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return fig

    controls = {
        "fraction_wrong": widgets.FloatSlider(value=0.1, min=0, max=0.5, step=0.05, description="Wrong", continuous_update=False),
        "fraction_missed": widgets.FloatSlider(value=0.0, min=0, max=0.5, step=0.05, description="Missed", continuous_update=False),
    }
    display(widgets.VBox(list(controls.values())), interactive_figure(figure, controls))


def evaluate_note_alignments(
    score,
    performance_names: List[str],
    mec_variation_dir: PathLike,
    methods: List[str] = NOTE_LEVEL_METHODS,
) -> pd.DataFrame:
    """F-score of the methods on several performances of the mec-variation dataset"""
    results = []
    for performance_name in performance_names:
        perf, gt = pt.load_match(
            _match_fn(mec_variation_dir, performance_name),
            pedal_threshold=127,
            first_note_at_zero=True,
        )
        labels = Counter(a["label"] for a in gt)
        for method in methods:
            pred = compute_alignment(method, score, perf)
            table = fscore_table(pred, gt)
            results.append(
                {
                    "performance": performance_name,
                    "method": method,
                    "insertions (gt)": labels["insertion"],
                    "deletions (gt)": labels["deletion"],
                    **{f"{t} F-score": table.loc[t, "F-score"] for t in ["match", "insertion", "deletion", "all"]},
                    "match precision": table.loc["match", "precision"],
                    "match recall": table.loc["match", "recall"],
                }
            )
    return pd.DataFrame(results)


def plot_note_alignment_results(
    results_df: pd.DataFrame,
    performance_names: List[str],
    methods: List[str] = NOTE_LEVEL_METHODS,
) -> Figure:
    """Precision, recall and F-score of the matches, for each method and performance"""
    fig = Figure(figsize=(14, 3.8))
    axes = fig.subplots(1, 3, sharey=True)
    x = np.arange(len(performance_names))
    for ax, metric in zip(axes, ["match precision", "match recall", "match F-score"]):
        for k, method in enumerate(methods):
            values = results_df[results_df["method"] == method][metric].values
            ax.bar(x + (k - 1) * 0.27, values, 0.27, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(performance_names, rotation=30)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    return fig


########################################################################
# Time-level alignments
########################################################################
def run_follower(method: str, performance_fn: PathLike, score_fn: PathLike) -> np.ndarray:
    """
    Run a Matchmaker score follower on a MIDI performance (the performance is
    "streamed" as if it were played in real time).

    Returns the alignment path: an array with shape (2, T), where the first row is the
    performance time (in seconds) and the second row the position in the score (in beats)
    reported by the follower.
    """
    # Matchmaker is quite verbose
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mm = Matchmaker(str(score_fn), str(performance_fn), input_type="midi", method=method)
        generator = mm.run(verbose=False)
        try:
            while True:
                next(generator)
        except StopIteration as stop:
            path = stop.value
    path = np.array(path, dtype=float)
    return path if path.shape[0] == 2 else path.T


def load_ground_truth(
    performance_name: str, score_notes: np.ndarray, mec_variation_dir: PathLike
) -> Tuple[np.ndarray, np.ndarray]:
    """Performed onset times (in s) and the corresponding score positions (in beats)"""
    return resolve_gt(
        _match_fn(mec_variation_dir, performance_name),
        score_notes,
        performance=_midi_fn(mec_variation_dir, performance_name),
    )


def make_performance_with_errors(
    performance_name: str,
    fraction_errors: float,
    mec_variation_dir: PathLike,
    seed: int = 0,
) -> str:
    """
    Add errors to a performance: a fraction of the notes get a wrong pitch (a few
    semitones away) and the same fraction of random notes are added.
    Returns the path to the MIDI file of the new performance (in a temporary folder).
    """
    global _ERROR_DIR
    if _ERROR_DIR is None:
        _ERROR_DIR = tempfile.mkdtemp()

    rng = np.random.RandomState(seed)
    perf = pt.load_performance(_midi_fn(mec_variation_dir, performance_name))
    notes = [dict(n) for n in perf.performedparts[0].notes]
    for note in notes:
        if rng.rand() < fraction_errors:
            note["pitch"] = int(np.clip(note["pitch"] + rng.choice([-3, -2, -1, 1, 2, 3]), 21, 108))
    t_max = max(n["note_off"] for n in notes)
    for k in range(int(round(fraction_errors * len(notes)))):
        extra = dict(notes[rng.randint(len(notes))])
        onset = rng.uniform(0, t_max - 0.5)
        extra.update(
            pitch=int(np.clip(extra["pitch"] + rng.randint(-12, 13), 21, 108)),
            note_on=onset, note_off=onset + 0.1, sound_off=onset + 0.1, id=f"extra{k}",
        )
        notes.append(extra)
    notes.sort(key=lambda n: n["note_on"])
    fn = os.path.join(_ERROR_DIR, f"{performance_name}_errors{fraction_errors:.2f}_{seed}.mid")
    pt.save_performance_midi(Performance(PerformedPart(notes, id="p", part_name="p")), fn)
    return fn


def run_followers(
    performance_fn: PathLike, score_fn: PathLike, methods: Dict[str, str] = ONLINE_METHODS
) -> Dict[str, np.ndarray]:
    """Alignment path of each of the followers"""
    return {name: run_follower(method, performance_fn, score_fn) for name, method in methods.items()}


########################################################################
# Time-level metrics
########################################################################
def evaluate_online(path: np.ndarray, gt_perf_sec: np.ndarray, gt_score_beat: np.ndarray) -> dict:
    """Matchmaker's metrics for the alignment path of a follower"""
    return evaluate_alignment(
        aligned_score_beats=path[1],
        aligned_perf_sec=path[0],
        gt_score_beats=gt_score_beat,
        gt_perf_sec=gt_perf_sec,
    )


def beat_errors(path: np.ndarray, gt_perf_sec: np.ndarray, gt_score_beat: np.ndarray):
    """Positions reported by the follower at the time of the ground truth onsets, and the signed beat errors"""
    predicted = transfer_positions(path, gt_perf_sec, frame_rate=1, domain="score")
    return predicted, gt_score_beat - predicted


def paths_figure(
    paths: Dict[str, np.ndarray],
    gt_perf_sec: np.ndarray,
    gt_score_beat: np.ndarray,
    zoom: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Alignment paths of the followers and ground truth. With `zoom` (a time interval), there is also a zoomed plot"""
    fig = Figure(figsize=(14, 4) if zoom is not None else (10, 4))
    axes = np.atleast_1d(fig.subplots(1, 2 if zoom is not None else 1))
    for ax, xlim in zip(axes, [None, zoom]):
        for name, path in paths.items():
            ax.step(path[0], path[1], where="post", label=name, color=ONLINE_COLORS[name], linewidth=1.5)
        ax.plot(gt_perf_sec, gt_score_beat, "k.", markersize=5, label="ground truth", zorder=5)
        ax.set_xlabel("Performance time (s)")
        ax.set_ylabel("Score position (beats)")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend()
    if title is not None:
        axes[0].set_title(title)
        if zoom is not None:
            axes[1].set_title("Zoom")
    fig.tight_layout()
    return fig


def beat_errors_figure(
    path: np.ndarray,
    gt_perf_sec: np.ndarray,
    gt_score_beat: np.ndarray,
    window: float = 4.0,
) -> Figure:
    """The position reported by a follower and the ground truth, around the largest beat error"""
    predicted, errors = beat_errors(path, gt_perf_sec, gt_score_beat)
    center = gt_perf_sec[np.nanargmax(np.abs(errors))]
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.step(path[0], path[1], where="post", color="0.3", label="position reported by the follower")
    ax.plot(gt_perf_sec, gt_score_beat, "k.", markersize=7, label="ground truth", zorder=5)
    ax.vlines(gt_perf_sec, predicted, gt_score_beat, color="tab:red", linewidth=2, label="beat error", zorder=4)
    ax.set_xlim(center - window / 2, center + window / 2)
    in_window = (gt_perf_sec > center - window / 2) & (gt_perf_sec < center + window / 2)
    ax.set_ylim(np.nanmin(np.r_[predicted[in_window], gt_score_beat[in_window]]) - 0.5,
                np.nanmax(np.r_[predicted[in_window], gt_score_beat[in_window]]) + 0.5)
    ax.set_xlabel("Performance time (s)")
    ax.set_ylabel("Score position (beats)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def error_distribution_figure(
    paths: Dict[str, np.ndarray],
    gt_perf_sec: np.ndarray,
    gt_score_beat: np.ndarray,
    histogram_of: str,
) -> Figure:
    """Signed beat errors over time for each follower, and the histogram of the errors of one of them"""
    fig = Figure(figsize=(13, 3.8))
    axes = fig.subplots(1, 2)
    for name, p in paths.items():
        _, err = beat_errors(p, gt_perf_sec, gt_score_beat)
        axes[0].plot(gt_perf_sec, err, label=name, color=ONLINE_COLORS[name], linewidth=1.2)
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_xlabel("Performance time (s)")
    axes[0].set_ylabel("Signed beat error (beats)\n(> 0: follower behind)")
    axes[0].legend()

    results = evaluate_online(paths[histogram_of], gt_perf_sec, gt_score_beat)
    _, errors = beat_errors(paths[histogram_of], gt_perf_sec, gt_score_beat)
    bins = np.linspace(-3, 3, 61)
    axes[1].hist(np.clip(errors, -3, 3), bins=bins, color=ONLINE_COLORS[histogram_of])
    axes[1].set_yscale("log")
    axes[1].set_xlabel(f"Signed beat error (beats, clipped to ±3), {histogram_of}")
    axes[1].set_ylabel("Number of onsets (log scale)")
    axes[1].set_title(f"skewness = {results['skewness']:.2f}, kurtosis = {results['kurtosis']:.2f}")
    fig.tight_layout()
    return fig


def accuracy_curves_figure(
    paths: Dict[str, np.ndarray], gt_perf_sec: np.ndarray, gt_score_beat: np.ndarray
) -> Figure:
    """Cumulative distribution of the absolute beat error of each follower, with the tolerances of Matchmaker"""
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    for name, p in paths.items():
        _, err = beat_errors(p, gt_perf_sec, gt_score_beat)
        abs_err = np.sort(np.abs(err[~np.isnan(err)]))
        ax.step(abs_err, np.arange(1, len(abs_err) + 1) / len(gt_perf_sec), where="post",
                label=name, color=ONLINE_COLORS[name])
    for tau in TOLERANCES_IN_BEATS:
        ax.axvline(tau, color="0.7", linestyle=":", zorder=0)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks([0] + TOLERANCES_IN_BEATS)
    ax.set_xticklabels(["0"] + [f"{t:g}" for t in TOLERANCES_IN_BEATS])
    ax.set_xlabel("Tolerance (beats)")
    ax.set_ylabel("Accuracy (proportion of onsets\nwith error $\\leq$ tolerance)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def speed_profile(path: np.ndarray, gt_perf_sec: np.ndarray, fs: float = 50.0):
    """Speed of the reported position (resampled at `fs` Hz) in the span of the ground truth onsets"""
    times = np.arange(gt_perf_sec.min(), gt_perf_sec.max(), 1.0 / fs)
    idx = np.searchsorted(path[0], times, side="right") - 1
    position = path[1][np.clip(idx, 0, None)]
    return times, position, np.abs(np.gradient(position, 1.0 / fs))


def smoothness_figure(
    paths: Dict[str, np.ndarray], gt_perf_sec: np.ndarray, fs: float = 50.0
) -> Figure:
    """Position, speed and spectrum of the speed (and its spectral arc length) of each follower"""
    fig = Figure(figsize=(14, 3 * len(paths)))
    axes = fig.subplots(len(paths), 3, gridspec_kw={"width_ratios": [1.3, 1.3, 1]}, squeeze=False)
    for row, (name, path) in enumerate(paths.items()):
        times, position, speed = speed_profile(path, gt_perf_sec, fs)
        n_fft = int(2 ** (np.ceil(np.log2(len(speed))) + 4))
        spectrum = np.abs(np.fft.rfft(speed, n=n_fft))
        spectrum /= spectrum.max()
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        mask = freqs <= 10
        axes[row, 0].plot(times, position, color=ONLINE_COLORS[name])
        axes[row, 0].set_ylabel(f"{name}\nposition (beats)")
        axes[row, 1].plot(times, speed, color=ONLINE_COLORS[name])
        axes[row, 1].set_ylabel("speed (beats/s)")
        axes[row, 2].plot(freqs[mask], spectrum[mask], color=ONLINE_COLORS[name])
        axes[row, 2].set_ylabel("normalized spectrum")
        axes[row, 2].set_title(f"SPARC = {spectral_arc_length(speed, fs):.2f}")
    for ax in axes[-1]:
        ax.set_xlabel("Performance time (s)" if ax is not axes[-1, 2] else "Frequency (Hz)")
    axes[0, 0].set_title("Position")
    axes[0, 1].set_title("Speed")
    fig.tight_layout()
    return fig


def interactive_online_comparison(
    performance_names: List[str], score_fn: PathLike, score_notes: np.ndarray, mec_variation_dir: PathLike
) -> None:
    """Sliders to choose the performance and the fraction of errors, and compare the followers"""

    def figure(performance_name, fraction_errors, seed) -> Figure:
        if fraction_errors == 0:
            fn = _midi_fn(mec_variation_dir, performance_name)
        else:
            fn = make_performance_with_errors(performance_name, fraction_errors, mec_variation_dir, seed=seed)
        gt_t, gt_b = load_ground_truth(performance_name, score_notes, mec_variation_dir)
        local_paths = run_followers(fn, score_fn)
        table = pd.DataFrame({name: evaluate_online(p, gt_t, gt_b) for name, p in local_paths.items()})

        fig = Figure(figsize=(15, 4.2))
        axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.4, 1, 1]})
        for name, p in local_paths.items():
            _, err = beat_errors(p, gt_t, gt_b)
            axes[0].plot(gt_t, err, color=ONLINE_COLORS[name], label=name, linewidth=1.2)
        axes[0].axhline(0, color="k", linewidth=0.8)
        axes[0].set_xlabel("Performance time (s)")
        axes[0].set_ylabel("Signed beat error (beats)")
        axes[0].legend(fontsize=8)

        x = np.arange(len(TOLERANCES_IN_BEATS))
        for name in local_paths:
            acc = [table.loc[f"accuracy_{t:g}b", name] for t in TOLERANCES_IN_BEATS]
            axes[1].plot(x, acc, marker="o", color=ONLINE_COLORS[name], label=name)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{t:g}" for t in TOLERANCES_IN_BEATS])
        axes[1].set_xlabel("Tolerance (beats)")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.02)

        axes[2].bar(range(len(local_paths)), table.loc["smoothness"].astype(float),
                    color=[ONLINE_COLORS[n] for n in local_paths])
        axes[2].set_xticks(range(len(local_paths)))
        axes[2].set_xticklabels(list(local_paths), rotation=30, ha="right")
        axes[2].set_ylabel("Smoothness (SPARC)")
        mean_errors = "   ".join(f"{name}: {table.loc['mean_b', name]:.3f}" for name in local_paths)
        fig.suptitle(f"Mean absolute beat error (beats)   {mean_errors}", fontsize=10)
        fig.tight_layout()
        return fig

    controls = {
        "performance_name": widgets.Dropdown(options=performance_names, value=performance_names[0], description="Performance"),
        "fraction_errors": widgets.FloatSlider(value=0.1, min=0, max=0.3, step=0.05, description="Errors", continuous_update=False),
        "seed": widgets.IntSlider(value=0, min=0, max=9, description="Seed", continuous_update=False),
    }
    display(widgets.VBox(list(controls.values())), interactive_figure(figure, controls))


def evaluate_online_followers(
    performance_names: List[str],
    score_fn: PathLike,
    score_notes: np.ndarray,
    mec_variation_dir: PathLike,
    fraction_errors: float = 0.1,
) -> pd.DataFrame:
    """Metrics of the followers on the performances of the dataset, as they are (clean) and with errors"""
    condition_errors = f"{100 * fraction_errors:.0f}% errors"
    results = []
    for performance_name in performance_names:
        gt_t, gt_b = load_ground_truth(performance_name, score_notes, mec_variation_dir)
        for condition, fraction in [("clean", 0.0), (condition_errors, fraction_errors)]:
            fn = (
                _midi_fn(mec_variation_dir, performance_name)
                if fraction == 0
                else make_performance_with_errors(performance_name, fraction, mec_variation_dir)
            )
            for name, path in run_followers(fn, score_fn).items():
                results.append(
                    {
                        "performance": performance_name,
                        "condition": condition,
                        "follower": name,
                        **evaluate_online(path, gt_t, gt_b),
                    }
                )
    return pd.DataFrame(results)


def plot_online_results(online_df: pd.DataFrame, performance_names: List[str]) -> Figure:
    """Bar plots of some of the metrics for each performance and each follower, with and without errors"""
    conditions = ["clean"] + [c for c in online_df["condition"].unique() if c != "clean"]
    fig = Figure(figsize=(15, 3.25 * len(conditions)))
    axes = fig.subplots(len(conditions), 3, squeeze=False)
    x = np.arange(len(performance_names))
    for row, condition in enumerate(conditions):
        for ax, metric in zip(axes[row], ["mean_b", "accuracy_0.3b", "smoothness"]):
            for k, name in enumerate(ONLINE_METHODS):
                values = online_df[(online_df["condition"] == condition) & (online_df["follower"] == name)][metric]
                ax.bar(x + (k - 1.5) * 0.2, values.astype(float), 0.2, label=name, color=ONLINE_COLORS[name])
            ax.set_xticks(x)
            ax.set_xticklabels(performance_names, rotation=30)
            ax.set_title(f"{metric} ({condition})")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig
