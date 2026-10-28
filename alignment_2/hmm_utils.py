#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for the `score_follower` notebook.

Most of the building blocks (tempo models, feature processors, etc.) come
from Matchmaker. This module contains

* `BaseHMM`: a simplified base class for the HMMs that we implement in the notebook
* functions to compute the features of a performance
* functions to plot and evaluate the score followers
"""
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import ipywidgets as widgets
import numpy as np
import pandas as pd
import partitura as pt
from hiddenmarkov import HiddenMarkovModel, ObservationModel, TransitionModel
from IPython.display import display
from matchmaker.features.midi import compute_features_from_symbolic
from matchmaker.prob.hmm import compute_chord_matrix, compute_ioi_matrix
from matchmaker.utils.eval import (
    TOLERANCES_IN_BEATS,
    evaluate_alignment,
    resolve_gt,
    transfer_positions,
)
from matchmaker.utils.misc import interleave_with_constant
from matchmaker.utils.tempo_models import (  # noqa: F401 (used in the notebook)
    KalmanTempoModel,
    LinearTempoModel,
    MovingAverageTempoModel,
    ReactiveTempoModel,
    TempoModel,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray
from partitura.utils.music import (
    midi_pitch_to_pitch_spelling,
    pitch_spelling_to_note_name,
)

from helper import interactive_figure  # from the alignment_1 folder

# Alias for typing arrays
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]

MODEL_COLORS = {"PitchHMM": "firebrick", "PitchIOIHMM": "navy"}


class BaseHMM(HiddenMarkovModel):
    """
    Base class for Hidden Markov Model alignment methods.

    (Matchmaker has a `BaseHMM` that also handles the input from a stream, which
    we do not need here.)

    Parameters
    ----------
    observation_model: ObservationModel
        An observation (data) model for computing the observation probabilities.

    transition_model: TransitionModel
        A transition model for computing the transition probabilities.

    state_space: np.ndarray
        The hidden states (positions in reference time).

    tempo_model: Optional[TempoModel]
        A tempo model instance

    has_insertions: bool
        A boolean indicating whether the state space consider inserted notes.
    """

    observation_model: ObservationModel
    transition_model: TransitionModel
    state_space: Union[NDArrayFloat, NDArrayInt]
    tempo_model: Optional[TempoModel]
    has_insertions: bool
    _warping_path: List[Tuple[int, int]]

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        state_space: Optional[Union[NDArrayFloat, NDArrayInt]] = None,
        tempo_model: Optional[TempoModel] = None,
        has_insertions: bool = False,
        **kwargs,
    ) -> None:
        HiddenMarkovModel.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            state_space=state_space,
        )
        self.tempo_model = tempo_model
        self.has_insertions = has_insertions
        self.input_index = 0
        self._warping_path = []
        self.current_state = 0

    @property
    def warping_path(self) -> NDArrayInt:
        return (np.array(self._warping_path).T).astype(np.int32)

    def __call__(self, input: NDArrayFloat) -> float:
        current_state = self.forward_algorithm_step(
            observation=input,
            log_probabilities=False,
        )

        self._warping_path.append((current_state, self.input_index))
        self.input_index += 1
        self.current_state = current_state

        return current_state


## Features
def compute_pitch_ioi_features(
    performance: pt.performance.PerformanceLike,
    polling_period: float = 0.01,
    piano_range: bool = True,
) -> List[Optional[Tuple[NDArrayFloat, float]]]:
    """
    Simulate the real-time input of a performance.

    The performance is split into frames of `polling_period` seconds. For each frame
    this returns None if there is no note onset, or the tuple (pitch_obs, ioi_obs) with
    the pitches of the onsets in the frame (a binary vector) and the
    inter-onset interval (in seconds) with respect to the previous onset.
    """
    frames = compute_features_from_symbolic(
        ref_info=performance,
        processor_name="pitch",
        processor_kwargs={"piano_range": piano_range},
        polling_period=polling_period,
    )
    features = []
    previous_time = None
    for frame in frames:
        if frame is None:
            features.append(None)
            continue
        pitch_obs, onset_time = frame
        # There is no IOI for the first observed note
        ioi_obs = 0.0 if previous_time is None else onset_time - previous_time
        previous_time = onset_time
        features.append((pitch_obs, ioi_obs))
    return features


def run_score_follower(
    model: BaseHMM,
    features: List[Optional[Tuple[NDArrayFloat, float]]],
    polling_period: float = 0.01,
) -> np.ndarray:
    """
    Run a score follower on the features of a performance.

    Returns an array with shape (N, 3) with the columns [frame index, time (s), score position (beats)]
    """
    position = model.current_state
    rows = []
    for i, feat in enumerate(features):
        if feat is not None:
            position = model(feat)
        rows.append((i, i * polling_period, model.state_space[position]))
    return np.array(rows)


## Plots
def interactive_pitch_profiles(
    pitch_profiles: NDArrayFloat, score_onsets: NDArrayFloat
) -> None:
    """Slider to look at the pitch profile of each score position"""

    def figure(i):
        fig = Figure(figsize=(8, 4))
        ax = fig.subplots()
        ax.plot(pitch_profiles[i])
        ax.set_xlabel("Piano key")
        ax.set_ylabel("Pitch likelihood")
        ax.set_title(f"Score Position {score_onsets[i]:.2f} beats")
        return fig

    controls = {
        "i": widgets.IntSlider(
            value=0,
            min=0,
            max=len(pitch_profiles) - 1,
            step=1,
            description="Position",
            continuous_update=False,  # avoids lag
        )
    }
    display(widgets.VBox(list(controls.values())), interactive_figure(figure, controls))


def interactive_observation_probabilities(observation_model: ObservationModel) -> None:
    """Slider to choose a piano key and plot its observation probability for each score position"""

    def figure(key):
        pitch_obs = np.zeros(88)
        pitch_obs[key] = 1

        fig = Figure(figsize=(10, 4))
        ax = fig.subplots()
        ax.plot(observation_model(pitch_obs))
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Observation probability")
        # Piano keys start at A0 (MIDI pitch 21)
        ax.set_title(
            f"Active piano key: "
            f"{pitch_spelling_to_note_name(*midi_pitch_to_pitch_spelling(key + 21))}"
        )
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig

    controls = {
        "key": widgets.IntSlider(
            value=0, min=0, max=87, step=1, description="Key", continuous_update=False
        )
    }
    display(widgets.VBox(list(controls.values())), interactive_figure(figure, controls))


def interactive_features(
    perf_features: List[Optional[Tuple[NDArrayFloat, float]]],
    observation_probability: Callable[[NDArrayFloat, float], np.ndarray],
    ylim: Tuple[float, float] = (0, 1),
) -> None:
    """
    Slider to look at the features of a performance (the onsets), and the
    observation probability for each score position of each of them.

    observation_probability: function of the pitch observation and the IOI
    """
    # indices of valid (non-None) features
    valid_idx = [i for i, feat in enumerate(perf_features) if feat is not None]

    # accumulated onsets
    perf_onsets = np.cumsum([perf_features[i][1] for i in valid_idx])

    def figure(j):
        i = valid_idx[j]
        pitch_obs, ioi_obs = perf_features[i]

        fig = Figure(figsize=(10, 6))
        axes = fig.subplots(2, 1)

        # ---- top: observation probability
        axes[0].plot(observation_probability(pitch_obs, ioi_obs))
        axes[0].set_ylabel("Observation probability")
        axes[0].set_ylim(ylim)
        axes[0].set_title(f"perf_features[{i}]")

        # ---- bottom: accumulated onset
        axes[1].scatter(j, perf_onsets[j])
        axes[1].plot(perf_onsets[: j + 1], marker="o")
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("Performed onset (s)")
        axes[1].grid(True)

        fig.tight_layout()
        return fig

    controls = {
        "j": widgets.IntSlider(
            value=0,
            min=0,
            max=len(valid_idx) - 1,
            step=1,
            description="Feature",
            continuous_update=False,
        )
    }
    display(widgets.VBox(list(controls.values())), interactive_figure(figure, controls))


def interactive_alignment(alignment: np.ndarray) -> None:
    """
    Slider to look at the alignment step by step.

    Parameters
    ----------
    alignment : np.ndarray
        Shape (N, 3): columns [i, time, position]
    """
    alignment = np.asarray(alignment)
    idxs = alignment[:, 0].astype(int)
    times = alignment[:, 1].astype(float)
    positions = alignment[:, 2].astype(float)

    def figure(step):
        fig = Figure(figsize=(10, 4))
        ax = fig.subplots()
        ax.plot(times[: step + 1], positions[: step + 1], lw=2)
        ax.plot([times[step]], [positions[step]], marker="o")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Aligned position")
        ax.set_title(f"Step {step} (i={idxs[step]}, t={times[step]:.3f}s)")
        fig.tight_layout()
        return fig

    controls = {
        "step": widgets.IntSlider(
            value=0,
            min=0,
            max=len(times) - 1,
            step=1,
            description="Step",
            continuous_update=False,
        )
    }
    display(widgets.VBox(list(controls.values())), interactive_figure(figure, controls))


def plot_alignments(alignments: Dict[str, np.ndarray]) -> Figure:
    """Plot the alignments (position in beats vs. time) of several score followers"""
    fig = Figure(figsize=(6.4, 4.8))
    ax = fig.subplots()
    for name, alignment in alignments.items():
        ax.plot(
            alignment[:, 1],
            alignment[:, 2],
            color=MODEL_COLORS[name],
            label=name,
            linewidth=2,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (beats)")
    ax.legend()
    return fig


def plot_evaluation(
    results: Dict[str, dict],
    alignments: Dict[str, np.ndarray],
    gt_perf_sec: np.ndarray,
    gt_score_beat: np.ndarray,
) -> Figure:
    """Plot the accuracy for each tolerance and the beat error of the alignments of several score followers"""
    fig = Figure(figsize=(12, 4))
    axes = fig.subplots(1, 2)

    # Accuracy for each tolerance
    width = 0.35
    x = np.arange(len(TOLERANCES_IN_BEATS))
    for k, name in enumerate(results):
        acc = [results[name][f"accuracy_{tau:g}b"] for tau in TOLERANCES_IN_BEATS]
        axes[0].bar(x + (k - 0.5) * width, acc, width, label=name, color=MODEL_COLORS[name])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{tau:g}" for tau in TOLERANCES_IN_BEATS])
    axes[0].set_xlabel("Tolerance (beats)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right")

    # Beat error at each ground truth onset
    for name, alignment in alignments.items():
        predicted = transfer_positions(
            np.array([alignment[:, 1], alignment[:, 2]]),
            gt_perf_sec,
            frame_rate=1,
            domain="score",
        )
        axes[1].plot(
            gt_perf_sec,
            np.abs(gt_score_beat - predicted),
            marker="o",
            markersize=3,
            label=name,
            color=MODEL_COLORS[name],
        )
    axes[1].set_xlabel("Performance time (s)")
    axes[1].set_ylabel("Absolute beat error (beats)")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_evaluation_all_performances(
    results_df: pd.DataFrame, performance_names: List[str]
) -> Figure:
    """Bar plots of some of the metrics for each performance and each score follower"""
    fig = Figure(figsize=(14, 4))
    axes = fig.subplots(1, 3)
    x = np.arange(len(performance_names))
    width = 0.35
    for ax, metric, ylabel in zip(
        axes,
        ["mean_b", "accuracy_0.05b", "smoothness"],
        ["Mean beat error (beats)", "Accuracy (0.05 beats)", "Smoothness (SPARC)"],
    ):
        for k, name in enumerate(MODEL_COLORS):
            values = results_df[results_df["model"] == name][metric]
            ax.bar(x + (k - 0.5) * width, values, width, label=name, color=MODEL_COLORS[name])
        ax.set_xticks(x)
        ax.set_xticklabels(performance_names, rotation=30)
        ax.set_ylabel(ylabel)
    axes[0].legend()
    fig.tight_layout()
    return fig


## Evaluation
def evaluate_followers(
    model_factories: Dict[str, Callable[[], BaseHMM]],
    performance_names: List[str],
    score_note_array: np.ndarray,
    mec_variation_dir: Union[str, os.PathLike],
    polling_period: float = 0.01,
) -> pd.DataFrame:
    """
    Run score followers on several performances of the mec-variation dataset and evaluate them
    with Matchmaker's evaluation code.

    model_factories: dictionary of functions that create a (new) score follower
    """
    results = []
    for performance_name in performance_names:
        match_fn = os.path.join(
            mec_variation_dir, "alignments", "match", f"{performance_name}.match"
        )
        performance, _ = pt.load_match(match_fn)
        features = compute_pitch_ioi_features(performance, polling_period)
        gt_perf_sec, gt_score_beat = resolve_gt(
            match_fn, score_note_array, performance=performance
        )
        for name, factory in model_factories.items():
            alignment = run_score_follower(factory(), features, polling_period)
            results.append(
                {
                    "performance": performance_name,
                    "model": name,
                    **evaluate_alignment(
                        alignment[:, 2], alignment[:, 1], gt_score_beat, gt_perf_sec
                    ),
                }
            )
    return pd.DataFrame(results)
