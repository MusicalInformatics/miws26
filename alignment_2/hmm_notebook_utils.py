#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for the `hidden_markov_models` notebook: example data, functions to compare and
plot the alignments of the Viterbi algorithm and DTW, and the step-by-step visualizations of the
Viterbi and forward algorithms.
"""
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib import colormaps
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from helper import RNG, generate_example_sequences, interactive_figure  # from the alignment_1 folder


def make_data(
    len_x: int = 15, noise: float = 0.1, seed: int = 1984
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Example sequences of the DTW tutorial (with a seed, to always get the same sequences)"""
    RNG.seed(seed)
    np.random.seed(seed)
    return generate_example_sequences(
        lenX=len_x, centers=3, n_features=5, maxreps=4, minreps=1, noise_scale=noise
    )


def states_per_observation(warping_path: np.ndarray, n_observations: int) -> List[Set[int]]:
    """For each observation (second column of the path), the states (first column) that it is aligned with"""
    states: List[Set[int]] = [set() for _ in range(n_observations)]
    for k, n in warping_path:
        states[n].add(int(k))
    return states


def viterbi_state_in_dtw(viterbi_path: np.ndarray, warping_path: np.ndarray) -> float:
    """Fraction of observations where the state of the Viterbi path is one of the states of the DTW alignment"""
    states = states_per_observation(warping_path, len(viterbi_path))
    return float(np.mean([viterbi_path[n] in states[n] for n in range(len(viterbi_path))]))


def identical_alignments(viterbi_path: np.ndarray, warping_path: np.ndarray) -> bool:
    """True if DTW aligns each observation with exactly the state of the Viterbi path"""
    states = states_per_observation(warping_path, len(viterbi_path))
    return all(states[n] == {int(viterbi_path[n])} for n in range(len(viterbi_path)))


def plot_paths(
    ax: Axes,
    C: np.ndarray,
    viterbi_path: np.ndarray,
    dtw_paths: Dict[str, np.ndarray],
) -> None:
    """Cost matrix (columns: observations, rows: states), with the Viterbi path and DTW warping paths"""
    ax.imshow(C, cmap="gray_r", origin="lower", aspect="auto")
    ax.plot(np.arange(len(viterbi_path)), viterbi_path, color="red", linewidth=6, alpha=0.5, label="Viterbi")
    for (name, path), (color, style) in zip(dtw_paths.items(), [("royalblue", "--"), ("darkorange", ":")]):
        ax.plot(path[:, 1], path[:, 0], color=color, linestyle=style, linewidth=2, label=name)
    ax.set_xlabel("Observations (sequence $\\mathbf{Y}$)")
    ax.set_ylabel("States (sequence $\\mathbf{X}$)")
    ax.legend(loc="upper left", fontsize=8)


## Step-by-step visualizations of the Viterbi and forward algorithms
def _fmt(x: float, decimals: int = 2) -> str:
    return "-inf" if np.isneginf(x) else f"{x:.{decimals}f}"


def _trellis_axes(
    n_states: int,
    n_obs: int,
    state_names: Sequence[str],
    observation_labels: Sequence[str],
    n_text_lines: int,
) -> Tuple[Figure, Axes, Axes]:
    """Figure with the axes of the matrix (states x observations) and an axes for the description"""
    fig = Figure(figsize=(13, 0.55 * n_states + 2.2 + 0.2 * n_text_lines))
    ax, ax_text = fig.subplots(
        2, 1,
        gridspec_kw={"height_ratios": [0.55 * n_states + 1.4, 0.2 * n_text_lines + 0.4], "hspace": 0.6},
    )
    ax.set_xticks(np.arange(-0.5, n_obs, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_states, 1), minor=True)
    ax.grid(which="minor", color="0.6", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(n_obs))
    ax.set_xticklabels([f"{n + 1}\n{label}" for n, label in enumerate(observation_labels)])
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(state_names)
    ax.set_xlim(-0.5, n_obs - 0.5)
    ax.set_ylim(-0.5, n_states - 0.5)
    ax.set_xlabel("Observation $n$")
    ax.set_ylabel("State $k$")
    ax_text.axis("off")
    return fig, ax, ax_text


def _draw_matrix(ax: Axes, values: np.ndarray, filled: int, vmin: float, vmax: float, decimals: int = 2) -> None:
    """Draw the first `filled` columns of the matrix (rows: states) with the values as text"""
    cmap = LinearSegmentedColormap.from_list("trellis", colormaps["Blues"](np.linspace(0.05, 0.55, 256)))
    shown = np.full(values.shape, np.nan)
    shown[:filled] = values[:filled]
    ax.imshow(
        np.ma.masked_invalid(shown).T, origin="lower", aspect="auto", cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    for n in range(filled):
        for k in range(values.shape[1]):
            ax.text(n, k, _fmt(values[n, k], decimals), ha="center", va="center", fontsize=9)


def _highlight_column(ax: Axes, column: int, n_states: int) -> None:
    ax.add_patch(
        Rectangle((column - 0.5, -0.5), 1, n_states, fill=False, edgecolor="darkorange", linewidth=3, zorder=4)
    )


def _mark_states(ax: Axes, observations_idx: np.ndarray, states: np.ndarray, line: bool) -> None:
    """Red circles around the selected states (connected with a line)"""
    ax.plot(
        observations_idx, states, color="red", alpha=0.6, linewidth=3 if line else 0, marker="o",
        markersize=22, markerfacecolor="none", markeredgewidth=3, zorder=5,
    )


def _write_description(ax_text: Axes, lines: Sequence[str]) -> None:
    ax_text.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace",
                 transform=ax_text.transAxes)


def _play_slider(max_step: int) -> Tuple[widgets.Play, widgets.IntSlider]:
    slider = widgets.IntSlider(value=0, min=0, max=max_step, step=1, description="Step")
    play = widgets.Play(value=0, min=0, max=max_step, step=1, interval=700)
    widgets.jslink((play, "value"), (slider, "value"))
    return play, slider


## Viterbi
class ViterbiTrellis(NamedTuple):
    """The quantities of the Viterbi algorithm (all in log probabilities)"""

    log_obs: np.ndarray  # (N, K): log p(x_n | k)
    log_trans: np.ndarray  # (K, K): log A[i, k], from state i to state k
    log_init: np.ndarray  # (K,): log pi_k
    omega: np.ndarray  # (N, K): omega[n, k]
    backpointers: np.ndarray  # (N, K): best previous state for each state (row 0 is not used)


def viterbi_trellis(hmm, observations: Sequence) -> ViterbiTrellis:
    """
    Compute the matrix omega of the Viterbi algorithm, and the backpointers, in the same
    way as the `viterbi_algorithm` of the notebook (with log probabilities).
    """
    hmm.transition_model.use_log_probabilities = True
    hmm.observation_model.use_log_probabilities = True
    n_states = hmm.n_states
    log_obs = np.array([hmm.observation_model(obs) for obs in observations])
    log_trans = np.array(
        [[hmm.transition_model(i, k) for k in range(n_states)] for i in range(n_states)]
    )
    log_init = np.array(hmm.transition_model.init_probabilities)

    omega = np.zeros((len(observations), n_states))
    backpointers = np.zeros((len(observations), n_states), dtype=int)
    omega[0] = log_obs[0] + log_init
    for n in range(1, len(observations)):
        candidates = omega[n - 1][:, None] + log_trans  # candidates[i, k]
        backpointers[n] = candidates.argmax(axis=0)
        omega[n] = log_obs[n] + candidates.max(axis=0)
    return ViterbiTrellis(log_obs, log_trans, log_init, omega, backpointers)


def viterbi_backtracking(trellis: ViterbiTrellis) -> np.ndarray:
    """The most likely sequence of states (indices)"""
    n_obs = len(trellis.omega)
    states = np.zeros(n_obs, dtype=int)
    states[-1] = trellis.omega[-1].argmax()
    for n in range(n_obs - 1, 0, -1):
        states[n - 1] = trellis.backpointers[n, states[n]]
    return states


def viterbi_step_figure(
    step: int,
    trellis: ViterbiTrellis,
    state_names: Sequence[str],
    observation_labels: Sequence[str],
) -> Figure:
    """
    Figure of the matrix omega at a step of the Viterbi algorithm.

    step 0: empty omega
    step 1: initialization
    steps 2, ..., N: recursion (one observation per step)
    step N + 1: choose the best final state (start of the backtracking)
    steps N + 2, ..., 2 N: backtracking (one observation per step)
    """
    omega, backpointers = trellis.omega, trellis.backpointers
    n_obs, n_states = omega.shape
    filled = min(step, n_obs)  # number of columns of omega that have been computed
    n_back = max(step - n_obs, 0)  # number of states of the best sequence that are known
    best_sequence = viterbi_backtracking(trellis)

    fig, ax, ax_text = _trellis_axes(n_states, n_obs, state_names, observation_labels, n_text_lines=n_states)
    finite = omega[np.isfinite(omega)]
    _draw_matrix(ax, omega, filled, finite.min(), finite.max())

    recursion_column = step - 1 if 2 <= step <= n_obs else None
    # Backpointers (arrows to the best previous state)
    for n in range(1, filled):
        for k in range(n_states):
            current = n == recursion_column
            ax.annotate(
                "", xy=(n - 0.4, k), xytext=(n - 1 + 0.4, backpointers[n, k]),
                arrowprops=dict(arrowstyle="->", color="black" if current else "0.55",
                                linewidth=1.8 if current else 0.9),
                zorder=3,
            )
    if step == 1:
        _highlight_column(ax, 0, n_states)
    if recursion_column is not None:
        _highlight_column(ax, recursion_column, n_states)
    if n_back >= 1:
        known = np.arange(n_obs - 1, n_obs - 1 - min(n_back, n_obs), -1)
        _mark_states(ax, known, best_sequence[known], line=True)

    # Description of the current step
    lines = []
    if step == 0:
        title = "Step 0: empty $\\omega$"
        lines = [
            "$\\omega_{nk}$ is the log-probability of the most likely sequence of states that ends in state $k$ at observation $n$.",
            "It has one cell for each state (rows) and each observation (columns). We fill it from left to right.",
        ]
    elif step == 1:
        title = "Step 1: initialization, $\\omega_{1k} = \\log p(x_1 \\mid k) + \\log \\pi_k$"
        for k in range(n_states):
            lines.append(
                f"{state_names[k]}: log p(x_1 | {state_names[k]}) + log pi = "
                f"{_fmt(trellis.log_obs[0, k])} + {_fmt(trellis.log_init[k])} = {_fmt(omega[0, k])}"
            )
    elif step <= n_obs:
        n = step - 1
        title = (f"Step {step}: recursion for observation {n + 1} ({observation_labels[n]}), "
                 "$\\omega_{nk} = \\log p(x_n \\mid k) + \\max_i \\{\\log A_{ik} + \\omega_{n-1,i}\\}$")
        for k in range(n_states):
            candidates = [trellis.log_trans[i, k] + omega[n - 1, i] for i in range(n_states)]
            candidate_text = ", ".join(f"{_fmt(trellis.log_trans[i, k])} + {_fmt(omega[n - 1, i])}" for i in range(n_states))
            lines.append(
                f"{state_names[k]}: {_fmt(trellis.log_obs[n, k])} + max({candidate_text})"
                f" = {_fmt(trellis.log_obs[n, k])} + {_fmt(max(candidates))} = {_fmt(omega[n, k])}"
                f"   (best previous state: {state_names[backpointers[n, k]]})"
            )
    elif step == n_obs + 1:
        title = "Step %d: backtracking, start from the most likely final state" % step
        last = best_sequence[-1]
        lines = [
            f"$\\arg\\max_k\\, \\omega_{{Nk}}$ = {state_names[last]}, with log-probability {_fmt(omega[-1, last])} "
            "(this is the log-probability of the best sequence).",
        ]
    else:
        n = n_obs - 1 - (n_back - 1)  # index of the observation whose state we just found
        title = f"Step {step}: backtracking, follow the backpointers"
        lines = [
            f"The best previous state of {state_names[best_sequence[n + 1]]} at observation {n + 2} is "
            f"{state_names[best_sequence[n]]} (at observation {n + 1})."
        ]
        if n_back == n_obs:
            lines.append("The red path is the most likely sequence of states!")
    ax.set_title(title, fontsize=11)
    _write_description(ax_text, lines)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.03)
    return fig


def interactive_viterbi(
    hmm,
    observations: Sequence,
    state_names: Sequence[str],
    observation_labels: Optional[Sequence[str]] = None,
) -> None:
    """Slider (and play button) to go through the steps of the Viterbi algorithm"""
    trellis = viterbi_trellis(hmm, observations)
    if observation_labels is None:
        observation_labels = [str(obs) for obs in observations]
    play, slider = _play_slider(2 * len(observations))

    def figure(step: int) -> Figure:
        return viterbi_step_figure(step, trellis, state_names, observation_labels)

    display(widgets.HBox([play, slider]), interactive_figure(figure, {"step": slider}))


## Forward algorithm
class ForwardTrellis(NamedTuple):
    """The quantities of the forward algorithm (in probabilities, with normalization)"""

    obs_prob: np.ndarray  # (N, K): p(x_n | k)
    trans: np.ndarray  # (K, K): A[i, k], from state i to state k
    init: np.ndarray  # (K,): pi_k
    alpha_unnormalized: np.ndarray  # (N, K): before the normalization (given the normalized previous step)
    normalization: np.ndarray  # (N,): sum of the unnormalized forward variable at each step
    alpha: np.ndarray  # (N, K): the normalized forward variable

    @property
    def log_likelihood(self) -> float:
        """log p(X | theta)"""
        return float(np.sum(np.log(self.normalization)))


def forward_trellis(hmm, observations: Sequence) -> ForwardTrellis:
    """
    Compute the (normalized) forward variable in the same way as the `forward_algorithm_step`
    of the notebook.
    """
    hmm.transition_model.use_log_probabilities = False
    hmm.observation_model.use_log_probabilities = False
    n_states = hmm.n_states
    obs_prob = np.array([hmm.observation_model(obs) for obs in observations])
    trans = np.array([[hmm.transition_model(i, k) for k in range(n_states)] for i in range(n_states)])
    init = np.array(hmm.transition_model.init_probabilities)

    alpha_unnormalized = np.zeros((len(observations), n_states))
    alpha = np.zeros((len(observations), n_states))
    normalization = np.zeros(len(observations))
    for n in range(len(observations)):
        predicted = init if n == 0 else trans.T @ alpha[n - 1]
        alpha_unnormalized[n] = obs_prob[n] * predicted
        normalization[n] = alpha_unnormalized[n].sum()
        alpha[n] = alpha_unnormalized[n] / normalization[n]
    return ForwardTrellis(obs_prob, trans, init, alpha_unnormalized, normalization, alpha)


def forward_step_figure(
    step: int,
    trellis: ForwardTrellis,
    state_names: Sequence[str],
    observation_labels: Sequence[str],
) -> Figure:
    """
    Figure of the forward variable at a step of the forward algorithm.

    step 0: empty forward variable
    step 1: initialization
    steps 2, ..., N: recursion (one observation per step)
    step N + 1: termination (probability of the observations)
    The red circles are the most likely state for each observation. This is a causal
    decision, so there is no backtracking.
    """
    alpha = trellis.alpha
    n_obs, n_states = alpha.shape
    filled = min(step, n_obs)

    fig, ax, ax_text = _trellis_axes(n_states, n_obs, state_names, observation_labels, n_text_lines=n_states + 2)
    _draw_matrix(ax, alpha, filled, 0.0, 1.0)

    recursion_column = step - 1 if 2 <= step <= n_obs else None
    if step == 1:
        _highlight_column(ax, 0, n_states)
    if recursion_column is not None:
        n = recursion_column
        _highlight_column(ax, n, n_states)
        # Arrows from the previous states, with a width proportional to their contribution
        contributions = trellis.alpha[n - 1][:, None] * trellis.trans  # [i, k]
        shares = contributions / contributions.sum(axis=0, keepdims=True)
        for i in range(n_states):
            for k in range(n_states):
                ax.annotate(
                    "", xy=(n - 0.3, k), xytext=(n - 1 + 0.3, i),
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=0.3 + 3.0 * shares[i, k]),
                    zorder=3,
                )
    if filled >= 1:
        _mark_states(ax, np.arange(filled), alpha[:filled].argmax(axis=1), line=False)

    lines = []
    if step == 0:
        title = "Step 0: empty forward variable $\\alpha$"
        lines = [
            "$\\alpha_n(k)$ is the probability of the first $n$ observations and being in state $k$ at observation $n$.",
            "(After each step we normalize it, so that the values of each column add up to 1.)",
        ]
        lines += [""] * (n_states)
    elif step == 1:
        title = "Step 1: initialization, $\\alpha_1(k) = \\pi_k \\; p(x_1 \\mid k)$"
        for k in range(n_states):
            lines.append(
                f"{state_names[k]}: pi * p(x_1 | {state_names[k]}) = {_fmt(trellis.init[k])} * {_fmt(trellis.obs_prob[0, k])}"
                f" = {_fmt(trellis.alpha_unnormalized[0, k], 3)}"
            )
        lines.append(f"Normalization: divide by the sum, {_fmt(trellis.normalization[0], 3)}")
        lines.append("")
    elif step <= n_obs:
        n = step - 1
        title = (f"Step {step}: recursion for observation {n + 1} ({observation_labels[n]}), "
                 "$\\alpha_n(k) = p(x_n \\mid k) \\sum_i \\alpha_{n-1}(i) A_{ik}$")
        for k in range(n_states):
            terms = " + ".join(f"{_fmt(alpha[n - 1, i])}*{_fmt(trellis.trans[i, k])}" for i in range(n_states))
            lines.append(
                f"{state_names[k]}: {_fmt(trellis.obs_prob[n, k])} * ({terms}) = {_fmt(trellis.alpha_unnormalized[n, k], 3)}"
            )
        lines.append(f"Normalization: divide by the sum, {_fmt(trellis.normalization[n], 3)}")
        lines.append("The arrows show how much each previous state contributes. Red: most likely state.")
    else:
        title = f"Step {step}: termination, probability of the observations"
        lines = [
            "Without normalization, p(X | theta) is the sum of the last forward variable.",
            "With normalization, it is the product of the normalization constants of all the steps:",
            f"log p(X | theta) = sum_n log(c_n) = {_fmt(trellis.log_likelihood)}",
        ]
        lines += [""] * (n_states - 1)
    ax.set_title(title, fontsize=11)
    _write_description(ax_text, lines)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.03)
    return fig


def interactive_forward(
    hmm,
    observations: Sequence,
    state_names: Sequence[str],
    observation_labels: Optional[Sequence[str]] = None,
) -> None:
    """Slider (and play button) to go through the steps of the forward algorithm"""
    trellis = forward_trellis(hmm, observations)
    if observation_labels is None:
        observation_labels = [str(obs) for obs in observations]
    play, slider = _play_slider(len(observations) + 1)

    def figure(step: int) -> Figure:
        return forward_step_figure(step, trellis, state_names, observation_labels)

    display(widgets.HBox([play, slider]), interactive_figure(figure, {"step": slider}))
