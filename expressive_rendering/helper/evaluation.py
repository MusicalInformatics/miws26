"""
Comparison of performances in the space of the expressive parameters (06_evaluation.ipynb).

All the performances are described with the performance codec of partitura
(`partitura.musicanalysis.encode_performance`). The durations of the real performances are the ones of
the keys (without the sustain pedal, `pedal_threshold=127`), as in the Basis Mixer.
"""
import functools
import html
import io
import os
import traceback

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio, display
from partitura import load_match, load_musicxml
from partitura.musicanalysis import decode_performance, encode_performance
from partitura.utils.fluidsynth import HAS_FLUIDSYNTH, synthesize_fluidsynth
from partitura.musicanalysis.performance_codec import TEMPO_NORMALIZATION

import helper.data as data

SAMPLE_RATE = 44100

# The random performance of section 4 (see `random_parameters`)
N_SEGMENTS = 4  # segments of equal length (in score beats)
ALPHA = 0.8  # the random values are in [mean - ALPHA * std, mean + ALPHA * std] of their segment

PARAMETERS = ["velocity", "beat_period", "timing", "articulation_log"]
TEMPO_DEFINITIONS = list(TEMPO_NORMALIZATION)
LABELS = {
    "velocity": "velocity (MIDI / 127)",
    "beat_period": "beat_period (s)",
    "timing": "timing of the top note (s)",
    "articulation_log": "articulation_log",
}


# ---------------------------------------------------------------------------------------------------
# Interactive displays
#
# The explorers do not capture printed output (`widgets.interact`, `widgets.Output`): some notebook front
# ends then show everything twice. Instead, every figure, table and audio is a widget, and a change of
# the controls replaces the widgets of one container.


def _figure_widget(fig):
    """A figure as an image widget (the figure is closed, so that it is not shown again by matplotlib)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return widgets.Image(value=buffer.getvalue(), format="png", layout=widgets.Layout(max_width="100%"))


def _table_widget(table):
    return widgets.HTML(table.to_html(float_format=lambda v: f"{v:.4f}"))


def _text_widget(text):
    return widgets.HTML(f"<b>{html.escape(text)}</b>")


def _audio_widget(audio):
    """An `IPython.display.Audio` as an audio widget (it does not start by itself)."""
    return widgets.Audio(value=audio.data, format="wav", autoplay=False, loop=False)


def _live_view(controls, render):
    """
    A container that shows the widgets returned by `render(**values of the controls)`, and shows the new
    ones when a control changes.
    """
    area = widgets.VBox()

    def update(change=None):
        values = {name: control.value for name, control in controls.items()}
        area.children = [_text_widget("computing ...")]
        try:
            area.children = render(**values)
        except Exception:  # errors in the callbacks of widgets are otherwise not visible
            area.children = [widgets.HTML(f"<pre>{html.escape(traceback.format_exc())}</pre>")]

    for control in controls.values():
        control.observe(update, names="value")
    update()
    return area


# ---------------------------------------------------------------------------------------------------
# Loading and encoding


@functools.lru_cache(maxsize=None)
def score_part(piece):
    """Score part of a piece of the Vienna 4x22 dataset (with the note ids of the file)."""
    fn = os.path.join(data.DATASET_DIR, "musicxml", f"{piece}.musicxml")
    return load_musicxml(fn, force_note_ids="keep").parts[0]


@functools.lru_cache(maxsize=None)
def score_notes(piece):
    """Dictionary note id -> (score onset in beats, MIDI pitch, is grace note)."""
    na = score_part(piece).note_array(include_grace_notes=True)
    return {
        str(n["id"]): (float(n["onset_beat"]), int(n["pitch"]), bool(n["is_grace"]))
        for n in na
    }


def match_file(piece, performer):
    return os.path.join(data.DATASET_DIR, "match", f"{piece}_{performer}.match")


@functools.lru_cache(maxsize=None)
def real_performance(piece, performer):
    """Performed part and alignment of a performance of the dataset."""
    performance, alignment = load_match(match_file(piece, performer), pedal_threshold=127)
    return performance[0], alignment


def encode(piece, ppart, alignment, tempo_definition="beat_period"):
    """Expressive parameters of a performance: dictionary note id -> record of the codec."""
    parameters, ids = encode_performance(
        score_part(piece), ppart, alignment, beat_normalization=tempo_definition
    )
    return {str(i): p for i, p in zip(ids, parameters)}


@functools.lru_cache(maxsize=None)
def encoded_performance(piece, performer, tempo_definition="beat_period"):
    return encode(piece, *real_performance(piece, performer), tempo_definition)


# ---------------------------------------------------------------------------------------------------
# 1. Comparing the performances of a piece


@functools.lru_cache(maxsize=None)
def onset_curves(piece, performer, tempo_definition="beat_period"):
    """
    Expressive parameters of a performance, one value per score onset.

    Velocity and articulation are the mean over the notes of the onset, the tempo is the same for all of
    them, and the timing is the one of the highest note (the mean timing of an onset is zero by
    definition). Grace notes are left out.

    Returns a dictionary with an array per parameter (same length as `score_onsets(piece)`, NaN where
    the performance has no matched note).
    """
    return _curves(piece, encoded_performance(piece, performer, tempo_definition))


def _curves(piece, parameters):
    """`onset_curves` of an encoded performance (dictionary note id -> parameters)."""
    notes = score_notes(piece)
    onsets = score_onsets(piece)
    row = {o: k for k, o in enumerate(onsets)}
    groups = [[] for _ in onsets]
    for i, values in parameters.items():
        onset, pitch, grace = notes[i]
        if not grace:
            groups[row[onset]].append((pitch, values))
    first = next(iter(parameters.values()))
    names = list(first.keys() if isinstance(first, dict) else first.dtype.names)
    curves = {n: np.full(len(onsets), np.nan) for n in names}
    for k, group in enumerate(groups):
        if group:
            top = max(group, key=lambda g: g[0])[1]
            for name in names:
                if name == "timing":
                    curves[name][k] = top[name]
                else:
                    curves[name][k] = np.mean([values[name] for _, values in group])
    return curves


@functools.lru_cache(maxsize=None)
def score_onsets(piece):
    return sorted({onset for onset, _, grace in score_notes(piece).values() if not grace})


def _matrix(piece, parameter, performers, tempo_definition="beat_period"):
    return np.array(
        [onset_curves(piece, p, tempo_definition)[parameter] for p in performers]
    )


def plot_performances(piece, highlight=("p01", "p02"), performers=None, figsize=(11, 9)):
    """
    The four expressive parameters of all performances of a piece: every pianist in gray, their
    average in black, and the pianists in `highlight` in color.
    """
    performers = list(performers or data.PERFORMERS)
    x = score_onsets(piece)
    fig, axs = plt.subplots(len(PARAMETERS), 1, sharex=True, figsize=figsize)
    for ax, parameter in zip(axs, PARAMETERS):
        values = _matrix(piece, parameter, performers)
        ax.plot(x, values.T, color="0.8", linewidth=0.6)
        ax.plot(x, np.nanmean(values, axis=0), color="k", linewidth=1.5, label="average")
        for color, performer in zip(plt.rcParams["axes.prop_cycle"].by_key()["color"], highlight):
            ax.plot(x, values[performers.index(performer)], color=color, linewidth=1, label=performer)
        ax.set_ylabel(LABELS[parameter], fontsize=8)
        # a few extreme values would flatten all the curves
        low, high = np.nanpercentile(values, [0.5, 99.5])
        ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    axs[0].legend(loc="upper right", ncol=len(highlight) + 1, fontsize=8, frameon=False)
    axs[0].set_title(f"{piece}: {len(performers)} performances")
    axs[-1].set_xlabel("score onset (beats)")
    fig.tight_layout()
    return fig


def performance_explorer(piece=None):
    """Interactive: the performances of a piece, with two pianists highlighted, and their correlations."""
    controls = dict(
        piece=widgets.Dropdown(options=data.PIECES, value=piece or data.PIECES[0], description="Piece:"),
        pianist_a=widgets.Dropdown(options=data.PERFORMERS, value="p01", description="Pianist A:"),
        pianist_b=widgets.Dropdown(options=data.PERFORMERS, value="p02", description="Pianist B:"),
    )

    def render(piece, pianist_a, pianist_b):
        return [
            _figure_widget(plot_performances(piece, highlight=(pianist_a, pianist_b))),
            _figure_widget(plot_correlations(piece)),
        ]

    display(widgets.VBox([widgets.HBox(list(controls.values())), _live_view(controls, render)]))


def correlation_matrix(values):
    """Pearson correlation between the rows of `values`, using the onsets that both have."""
    n = len(values)
    result = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ok = np.isfinite(values[i]) & np.isfinite(values[j])
            result[i, j] = result[j, i] = np.corrcoef(values[i][ok], values[j][ok])[0, 1]
    return result


def plot_correlations(piece, performers=None, figsize=(11, 9)):
    """For each parameter, the correlation between every pair of pianists."""
    performers = list(performers or data.PERFORMERS)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for ax, parameter in zip(axs.ravel(), PARAMETERS):
        matrix = correlation_matrix(_matrix(piece, parameter, performers))
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        mean = (matrix.sum() - len(matrix)) / (len(matrix) * (len(matrix) - 1))
        ax.set_title(f"{parameter} (average r = {mean:.2f})", fontsize=10)
        ax.set_xticks(range(len(performers)), performers, rotation=90, fontsize=6)
        ax.set_yticks(range(len(performers)), performers, fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"{piece}: correlation between pianists")
    fig.tight_layout()
    return fig


def plot_all_pieces(performers=None, figsize=(11, 6)):
    """Distribution over the pianists of the average value of each parameter, for every piece."""
    performers = list(performers or data.PERFORMERS)
    spread = {"timing": "spread (std) of the timing of the top note (s)"}
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for ax, parameter in zip(axs.ravel(), PARAMETERS):
        summary = np.nanstd if parameter in spread else np.nanmean
        values = [summary(_matrix(piece, parameter, performers), axis=1) for piece in data.PIECES]
        ax.boxplot(values)
        ax.set_xticks(range(1, len(data.PIECES) + 1), [p.replace("_", "\n", 1) for p in data.PIECES], fontsize=7)
        ax.set_title(spread.get(parameter, f"average {LABELS[parameter]}"), fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------
# 2. Definitions of the tempo parameters


def plot_tempo_definitions(piece, definitions, performers):
    """
    The same performances with different definitions of the tempo: the curve of each definition (left)
    and the values of its additional parameters, one per performance (right).
    """
    x = score_onsets(piece)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(
        len(definitions), 2, figsize=(12, 2.4 * len(definitions)), squeeze=False,
        gridspec_kw=dict(width_ratios=[4, 1]),
    )
    for row, definition in zip(axs, definitions):
        curve_ax, constant_ax = row
        curves = [onset_curves(piece, p, definition) for p in performers]
        for k, (performer, curve) in enumerate(zip(performers, curves)):
            curve_ax.plot(x, curve[definition], color=colors[k % len(colors)], linewidth=1, label=performer)
        curve_ax.set_ylabel(definition, fontsize=8)
        extras = [n for n in TEMPO_NORMALIZATION[definition]["param_names"] if n != definition]
        if extras:
            width = 0.8 / len(extras)
            for e, name in enumerate(extras):
                constant_ax.bar(
                    np.arange(len(performers)) + (e - (len(extras) - 1) / 2) * width,
                    [np.nanmean(c[name]) for c in curves], width,
                    color=[colors[k % len(colors)] for k in range(len(performers))],
                    alpha=1.0 if e == 0 else 0.45, edgecolor="k", linewidth=0.5,
                )
            constant_ax.set_title(" (dark), ".join(extras) + (" (light)" if len(extras) > 1 else ""), fontsize=7)
            constant_ax.set_xticks(range(len(performers)), performers, rotation=90, fontsize=7)
        else:
            constant_ax.axis("off")
            constant_ax.text(0.5, 0.5, "no additional\nparameters", ha="center", va="center", fontsize=8)
    axs[0, 0].legend(ncol=6, fontsize=7, frameon=False, loc="upper right")
    axs[0, 0].set_title(piece)
    axs[-1, 0].set_xlabel("score onset (beats)")
    fig.tight_layout()
    return fig


def pianist_agreement(piece, definitions=TEMPO_DEFINITIONS, performers=None):
    """
    How similar the pianists are according to each definition of the tempo: MSE and correlation of the
    tempo curves, averaged over all pairs of pianists.
    """
    performers = list(performers or data.PERFORMERS)
    rows = []
    for definition in definitions:
        values = _matrix(piece, definition, performers, definition)
        mse = [
            np.nanmean((values[i] - values[j]) ** 2)
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
        r = correlation_matrix(values)[np.triu_indices(len(values), 1)]
        rows.append((definition, np.mean(mse), np.mean(r)))
    return pd.DataFrame(rows, columns=["tempo definition", "MSE", "r"]).set_index("tempo definition")


def tempo_explorer(piece=None):
    """Interactive comparison of the definitions of the tempo."""
    piece_menu = widgets.Dropdown(options=data.PIECES, value=piece or data.PIECES[0], description="Piece:")
    definition_menu = widgets.SelectMultiple(
        options=TEMPO_DEFINITIONS, value=("beat_period", "beat_period_standardized"),
        description="Tempo:", rows=len(TEMPO_DEFINITIONS),
    )
    performer_menu = widgets.SelectMultiple(
        options=data.PERFORMERS, value=tuple(data.PERFORMERS[:3]), description="Pianists:", rows=8
    )

    def render(piece, definitions, performers):
        if not definitions or not performers:
            return [_text_widget("Select at least one definition and one pianist (ctrl/cmd + click for several).")]
        return [
            _figure_widget(plot_tempo_definitions(piece, definitions, performers)),
            _text_widget("Average difference between all pairs of pianists:"),
            _table_widget(pianist_agreement(piece, definitions)),
        ]

    controls = dict(piece=piece_menu, definitions=definition_menu, performers=performer_menu)
    display(widgets.VBox([widgets.HBox(list(controls.values())), _live_view(controls, render)]))


# ---------------------------------------------------------------------------------------------------
# 3. Evaluation of the models


def compare_parameters(reference, other, note_ids):
    """MSE and Pearson correlation of each parameter on the notes `note_ids` (encoded performances).
    If one of the two is constant, the correlation is not defined (its standard deviation is zero); we
    report 0."""
    result = {}
    for parameter in PARAMETERS:
        a = np.array([reference[i][parameter] for i in note_ids], dtype=float)
        b = np.array([other[i][parameter] for i in note_ids], dtype=float)
        constant = np.std(a) < 1e-6 or np.std(b) < 1e-6
        r = 0.0 if constant else np.corrcoef(a, b)[0, 1]
        result[parameter] = (np.mean((a - b) ** 2), r)
    return result


def common_notes(piece, encoded_performances):
    """Ids of the notes (no grace notes) that all the encoded performances have."""
    ids = set.intersection(*(set(e) for e in encoded_performances))
    return sorted(i for i in ids if not score_notes(piece)[i][2])


def evaluate_systems(systems, test_performers=("p21", "p22"), pieces=None):
    """
    Compare generated performances with the test performances.

    Parameters
    ----------
    systems : dict
        Name of the system -> function `f(piece)` that returns a list of encoded performances
        (`encode`, with `beat_period`), e.g. one per random seed; their metrics are averaged.
    test_performers : sequence
        Pianists of the test set (at least two). Besides the systems, the "human" row compares each of
        them with the others: how different two pianists are.

    Returns
    -------
    results : pandas.DataFrame
        One row per piece, system, test performance and generated performance (or other pianist).
    """
    rows = []
    for piece in pieces or data.PIECES:
        targets = {p: encoded_performance(piece, p) for p in test_performers}
        generated = {name: f(piece) for name, f in systems.items()}
        # all the rows of a piece are computed on the same notes
        notes = common_notes(
            piece, list(targets.values()) + [e for outputs in generated.values() for e in outputs]
        )
        for performer, target in targets.items():
            candidates = [(name, e) for name, outputs in generated.items() for e in outputs]
            candidates += [("another pianist", e) for p, e in targets.items() if p != performer]
            for name, candidate in candidates:
                for parameter, (mse, r) in compare_parameters(target, candidate, notes).items():
                    rows.append(
                        dict(piece=piece, system=name, target=performer, parameter=parameter,
                             MSE=mse, r=r, notes=len(notes))
                    )
    return pd.DataFrame(rows)


def results_table(results, by_piece=False):
    """MSE and correlation of each system and parameter: average over the test performances of each
    piece and then over the pieces. With `by_piece` one row per piece and system."""
    order = list(dict.fromkeys(results["system"]))
    per_piece = results.groupby(["piece", "system", "parameter"])[["MSE", "r"]].mean()
    index = ["piece", "system"] if by_piece else ["system"]
    values = per_piece if by_piece else per_piece.groupby(["system", "parameter"]).mean()
    table = values.unstack("parameter")
    table.columns = table.columns.swaplevel()
    table = table[[(p, m) for p in PARAMETERS for m in ("MSE", "r")]]
    if by_piece:
        return table.reindex(order, level=index[1])
    return table.reindex(order)


# ---------------------------------------------------------------------------------------------------
# 4. A random performance with good numbers


@functools.lru_cache(maxsize=None)
def average_performance(piece, performers):
    """
    Average of the expressive parameters of several pianists, note by note.

    Parameters
    ----------
    piece : str
    performers : tuple

    Returns
    -------
    ids : list
        Notes (without grace notes) that at least one of the pianists played.
    average : dict
        An array (same order as `ids`) for each parameter. The beat period is the same for all notes of
        an onset.
    """
    encoded = [encoded_performance(piece, p) for p in performers]
    notes = score_notes(piece)
    ids = sorted(
        {i for e in encoded for i in e if not notes[i][2]}, key=lambda i: (notes[i][0], notes[i][1])
    )
    average = {
        name: np.array([np.mean([e[i][name] for e in encoded if i in e]) for i in ids])
        for name in PARAMETERS
    }
    onsets = np.array([notes[i][0] for i in ids])
    for onset in np.unique(onsets):
        average["beat_period"][onsets == onset] = average["beat_period"][onsets == onset].mean()
    return ids, average


def _mse_to_average(piece, encoded, performers):
    """MSE of each parameter between a performance and the average of `performers`, on the notes that
    all of them played."""
    ids, average = average_performance(piece, performers)
    played = common_notes(piece, [encoded_performance(piece, p) for p in performers])
    row = {i: k for k, i in enumerate(ids)}
    reference = {i: {n: average[n][row[i]] for n in PARAMETERS} for i in played}
    return compare_parameters(reference, encoded, played)


@functools.lru_cache(maxsize=None)
def segment_statistics(piece, performers):
    """
    Split the piece into `N_SEGMENTS` segments of equal length (in score beats). In each segment, take
    the mean and the standard deviation of every parameter of each pianist, and average them over the
    pianists: the mean and standard deviation of a typical pianist in that segment.

    Parameters
    ----------
    piece : str
    performers : tuple

    Returns
    -------
    edges : np.ndarray
        Score onsets (beats) of the `N_SEGMENTS + 1` limits of the segments.
    segment : np.ndarray
        Index of the segment of each note of `average_performance(piece, performers)`.
    statistics : dict
        For each parameter a tuple `(mean, std)` of arrays with one value per segment (NaN if a segment
        has no notes). For the beat period each onset counts once (not once per note).
    """
    ids, _ = average_performance(piece, performers)
    onsets = np.array([score_notes(piece)[i][0] for i in ids])
    edges = np.linspace(onsets.min(), onsets.max(), N_SEGMENTS + 1)
    segment = np.clip(np.searchsorted(edges, onsets, side="right") - 1, 0, N_SEGMENTS - 1)
    encoded = [encoded_performance(piece, p) for p in performers]
    statistics = {}
    for name in PARAMETERS:
        means = np.full((len(performers), N_SEGMENTS), np.nan)
        stds = np.full((len(performers), N_SEGMENTS), np.nan)
        for j, performance in enumerate(encoded):
            for k in range(N_SEGMENTS):
                notes = [i for i, s in zip(ids, segment) if s == k and i in performance]
                if name == "beat_period":
                    notes = list({score_notes(piece)[i][0]: i for i in notes}.values())
                if notes:
                    values = np.array([performance[i][name] for i in notes])
                    means[j, k], stds[j, k] = values.mean(), values.std()
        # (a segment without notes gives NaN, and is never used)
        count = np.isfinite(means).sum(axis=0)
        with np.errstate(invalid="ignore"):
            statistics[name] = (np.nansum(means, axis=0) / count, np.nansum(stds, axis=0) / count)
    return edges, segment, statistics


def random_parameters(piece, seed=0, performers=None, parameters=None, alpha=None):
    """
    A performance in which some expressive parameters are random.

    The piece is split into `N_SEGMENTS` segments of equal length. In each segment, with `mu` and `std`
    the mean and standard deviation of a typical pianist in that segment (`segment_statistics`), the
    values of a parameter are drawn from a uniform distribution between `mu - ALPHA * std` and
    `mu + ALPHA * std`: one per onset for the tempo, and one per note for the other parameters.

    With `ALPHA = 0.8`, for the Schubert piece a random beat period or timing is always closer to the
    average performance (smaller MSE) than at least one pianist (checked with 1000 seeds: at least 4
    pianists for both). This does not hold for the velocity and the articulation, nor for every
    parameter of the other pieces.

    `parameters` are the random ones (all of `PARAMETERS` by default); the others are the ones of the
    average performance. `alpha` is the weight of the standard deviation (`ALPHA` by default).

    Returns a dictionary note id -> parameters, like `encode`.
    """
    alpha = ALPHA if alpha is None else alpha
    performers = tuple(performers or data.PERFORMERS)
    ids, average = average_performance(piece, performers)
    onsets = np.array([score_notes(piece)[i][0] for i in ids])
    unique, onset_index = np.unique(onsets, return_inverse=True)
    _, segment, statistics = segment_statistics(piece, performers)

    rng = np.random.RandomState(seed)
    values = {}
    for name in PARAMETERS:
        # uniform in [-1, 1] (drawn for all parameters, so selecting one does not change the others)
        if name == "beat_period":
            uniform = rng.uniform(-1, 1, len(unique))[onset_index]
        else:
            uniform = rng.uniform(-1, 1, len(ids))
        values[name] = average[name]
        if parameters is None or name in parameters:
            mean, std = statistics[name]
            values[name] = mean[segment] + alpha * uniform * std[segment]
    values["beat_period"] = np.maximum(values["beat_period"], 0.2 * average["beat_period"])
    values["velocity"] = np.clip(values["velocity"], 1 / 127, 1)
    return {i: {name: values[name][k] for name in PARAMETERS} for k, i in enumerate(ids)}


def constant_parameters(piece, performers=None, parameters=None):
    """
    A performance that "predicts" a constant value for some expressive parameters: the mean of that
    parameter in the average performance (for the beat period, over the onsets). The other parameters are
    the ones of the average performance. `parameters` are the constant ones (all by default).

    Returns a dictionary note id -> parameters, like `encode`.
    """
    performers = tuple(performers or data.PERFORMERS)
    ids, average = average_performance(piece, performers)
    onsets = np.array([score_notes(piece)[i][0] for i in ids])
    first_of_onset = np.unique(onsets, return_index=True)[1]
    values = {}
    for name in PARAMETERS:
        values[name] = average[name]
        if parameters is None or name in parameters:
            counted = average[name][first_of_onset] if name == "beat_period" else average[name]
            values[name] = np.full(len(ids), counted.mean())
    return {i: {name: values[name][k] for name in PARAMETERS} for k, i in enumerate(ids)}


def average_parameters(piece, performers=None):
    """The average performance in the format of `encode`."""
    ids, average = average_performance(piece, tuple(performers or data.PERFORMERS))
    return {i: {name: average[name][k] for name in PARAMETERS} for k, i in enumerate(ids)}


def distances_to_average(piece, random, performers=None, constant=None):
    """
    MSE and correlation with the average performance, for every pianist and for the random performance.
    Everybody is compared with the same average (of all pianists, which includes each of them).
    """
    performers = tuple(performers or data.PERFORMERS)
    candidates = {p: encoded_performance(piece, p) for p in performers}
    candidates["random"] = random
    if constant is not None:
        candidates["constant"] = constant
    rows = []
    for name, encoded in candidates.items():
        for parameter, (mse, r) in _mse_to_average(piece, encoded, performers).items():
            rows.append(dict(performance=name, parameter=parameter, MSE=mse, r=r))
    return pd.DataFrame(rows)


class _NoteArray(object):
    def __init__(self, note_array):
        self._note_array = note_array

    def note_array(self, *args, **kwargs):
        return self._note_array


def to_performance(piece, parameters):
    """Performance (`PerformedPart`) with the given parameters (format of `encode`), e.g. to listen to it.
    Notes without parameters (grace notes) are left out."""
    note_array = score_part(piece).note_array()
    note_array = note_array[[str(i) in parameters for i in note_array["id"]]]
    note_array = note_array[np.lexsort((note_array["pitch"], note_array["onset_div"]))]
    values = np.zeros(len(note_array), dtype=[(name, "f4") for name in PARAMETERS])
    for k, i in enumerate(note_array["id"]):
        for name in PARAMETERS:
            values[name][k] = parameters[str(i)][name]
    return decode_performance(_NoteArray(note_array), values)


def excerpt_notes(piece, seconds):
    """Ids of the notes of the first `seconds` seconds of the average performance."""
    performance = to_performance(piece, average_parameters(piece))
    return {str(n["id"]) for n in performance.notes if n["note_on"] <= seconds}


def synthesize(piece, parameters, note_ids=None):
    """Audio (with FluidSynth) of the performance with the given parameters, optionally only of some notes."""
    assert HAS_FLUIDSYNTH, "FluidSynth is not available (see the installation instructions of partitura)"
    if note_ids is not None:
        parameters = {i: v for i, v in parameters.items() if i in note_ids}
    audio = synthesize_fluidsynth(note_info=to_performance(piece, parameters), samplerate=SAMPLE_RATE)
    return Audio(audio, rate=SAMPLE_RATE)


def pianist_to_compare(distances, parameters):
    """The pianist who is farther from the average than the random performance in as many of the
    parameters as possible (and by the largest factor). If nobody is farther, the pianist who is farthest
    relative to the random performance."""
    mse = distances.pivot(index="performance", columns="parameter", values="MSE")[list(parameters)]
    ratio = mse.drop(["random", "constant"], errors="ignore") / mse.loc["random"]
    wins = (ratio > 1).sum(axis=1)
    return ratio.loc[wins == wins.max()].mean(axis=1).idxmax()


def plot_random_parameters(piece, random, parameters, pianist, note_ids=None, performers=None, alpha=None):
    """
    For each random parameter: its curve in the excerpt given by `note_ids` (average, random and one
    pianist) with the segments and the range from which the random values are drawn (red area), and the MSE to the average of all pianists and of the random performance (whole piece).
    """
    alpha = ALPHA if alpha is None else alpha
    constant = constant_parameters(piece, performers, parameters)
    distances = distances_to_average(piece, random, performers, constant)
    onsets = score_onsets(piece)
    if note_ids is None:
        shown = np.ones(len(onsets), dtype=bool)
    else:
        last = max(score_notes(piece)[i][0] for i in note_ids)
        shown = np.array(onsets) <= last
    x = np.array(onsets)[shown]
    curves = [
        ("average", _curves(piece, average_parameters(piece, performers)), dict(color="k", linewidth=1.5)),
        (pianist, onset_curves(piece, pianist), dict(color="C0", linewidth=1)),
        ("random", _curves(piece, random), dict(color="#d62728", linewidth=1)),
        ("constant", _curves(piece, constant), dict(color="#2ca02c", linewidth=1.5)),
    ]
    fig, axs = plt.subplots(
        len(parameters), 2, figsize=(12, 3 * len(parameters)), squeeze=False,
        gridspec_kw=dict(width_ratios=[3, 1.3]),
    )
    for (curve_ax, bar_ax), parameter in zip(axs, parameters):
        edges, _, statistics = segment_statistics(piece, tuple(performers or data.PERFORMERS))
        mean, std = statistics[parameter]
        segment = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_SEGMENTS - 1)
        low, high = [mean[segment] + side * alpha * std[segment] for side in (-1, 1)]
        # (the curves show one value per onset: the mean over its notes, the top note for the timing)
        curve_ax.fill_between(x, low, high, step="mid", color="#d62728", alpha=0.12, linewidth=0)
        for edge in edges[edges <= x[-1]]:
            curve_ax.axvline(edge, color="0.6", linestyle=":", linewidth=0.8)
        for label, curve, style in curves:
            curve_ax.plot(x, curve[parameter][shown], label=label, **style)
        curve_ax.set_ylabel(LABELS[parameter], fontsize=8)
        curve_ax.legend(ncol=4, fontsize=8, frameon=False, loc="best")
        curve_ax.set_title(parameter, fontsize=10)

        values = distances[distances["parameter"] == parameter].sort_values("MSE")
        names = list(values["performance"])
        palette = {"random": "#d62728", "constant": "#2ca02c", pianist: "C0"}
        bar_ax.bar(range(len(names)), values["MSE"], color=[palette.get(n, "0.7") for n in names])
        bar_ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        mse = values.set_index("performance")["MSE"]
        pianists = mse.drop(["random", "constant"])
        bar_ax.set_title(
            f"MSE to the average. Pianists farther than\nrandom: {(pianists > mse['random']).sum()}, "
            f"than constant: {(pianists > mse['constant']).sum()}",
            fontsize=8,
        )
    axs[-1, 0].set_xlabel("score onset (beats)")
    fig.suptitle(f"{piece} (alpha = {alpha:g})", fontsize=10)
    fig.tight_layout()
    return fig


def random_explorer(piece="Schubert_D783_no15", seconds=30):
    """
    Interactive: choose which expressive parameters are random, see them, and listen to the result. All
    the other parameters are the ones of the average performance.
    """
    piece_menu = widgets.Dropdown(options=data.PIECES, value=piece, description="Piece:")
    boxes = {
        name: widgets.Checkbox(value=name == "beat_period", description=name, indent=False)
        for name in PARAMETERS
    }
    pianist_menu = widgets.Dropdown(
        options=["automatic"] + list(data.PERFORMERS), value="automatic", description="Pianist:"
    )
    alpha_slider = widgets.FloatSlider(
        value=ALPHA, min=0.0, max=3.0, step=0.1, description="Alpha:", continuous_update=False,
        readout_format=".1f",
    )
    button = widgets.Button(description="New random values", icon="random")
    seed = widgets.IntText(value=0, description="Seed:", layout=widgets.Layout(width="160px"))
    button.on_click(lambda _: setattr(seed, "value", seed.value + 1))

    def render(piece, pianist, alpha, seed, **selected):
        parameters = [name for name in PARAMETERS if selected[name]]
        if not parameters:
            return [_text_widget("Select at least one parameter to make it random.")]
        random = random_parameters(piece, seed=seed, parameters=parameters, alpha=alpha)
        if pianist == "automatic":
            pianist = pianist_to_compare(distances_to_average(piece, random), parameters)
        note_ids = excerpt_notes(piece, seconds)
        constant = constant_parameters(piece, parameters=parameters)
        table = distances_to_average(piece, random, constant=constant)
        table = table[
            table["performance"].isin([pianist, "random", "constant"]) & table["parameter"].isin(parameters)
        ]
        result = [
            _figure_widget(plot_random_parameters(piece, random, parameters, pianist, note_ids, alpha=alpha)),
            _table_widget(table.set_index(["parameter", "performance"]).sort_index()),
        ]
        versions = [
            ("1. Average performance", average_parameters(piece)),
            (f"2. Average performance with random {', '.join(parameters)}", random),
            (f"3. Average performance with constant {', '.join(parameters)}", constant),
            (f"4. Pianist {pianist}", encoded_performance(piece, pianist)),
        ]
        for title, values in versions:
            result += [_text_widget(title), _audio_widget(synthesize(piece, values, note_ids))]
        return result

    controls = dict(piece=piece_menu, pianist=pianist_menu, alpha=alpha_slider, seed=seed)
    menus = widgets.VBox([
        widgets.HBox([piece_menu, pianist_menu]),
        widgets.HBox([widgets.Label("Random:")] + list(boxes.values())),
        widgets.HBox([alpha_slider, seed, button]),
    ])
    display(widgets.VBox([menus, _live_view({**controls, **boxes}, render)]))
