"""
Plots, animation and audio of the fingerprinting notebook: piano roll of a sequence, animation of the tokenization, and
the scatter plot and the histogram of the matching tokens of a query (Figure 3 of Arzt, Böck and Widmer, 2012).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle
from partitura.utils import compute_pianoroll

from fingerprinting import _iter_triplets, generate_triplets

FIRST_COLOR, SECOND_COLOR, THIRD_COLOR = "#d62728", "#1f77b4", "#2ca02c"
NOTE_COLOR = "#7f7f7f"


def synthesize_notes(notes: np.ndarray, samplerate: int = 22050) -> np.ndarray:
    """
    Synthesize a note array (with ``onset_sec`` and ``duration_sec``) with partitura: with fluidsynth (sampled piano
    sound) if it is available, and with additive synthesis otherwise.
    """
    try:
        from partitura.utils.fluidsynth import (
            DEFAULT_SOUNDFONT,
            HAS_FLUIDSYNTH,
            synthesize_fluidsynth,
        )

        if HAS_FLUIDSYNTH:
            if (
                os.path.getsize(DEFAULT_SOUNDFONT) < 5e6
            ):  # fluidsynth does not fail with an incomplete soundfont file
                raise RuntimeError(
                    f"the soundfont {DEFAULT_SOUNDFONT} is incomplete; delete it to download it again"
                )
            return synthesize_fluidsynth(notes, samplerate=samplerate)
    except (
        Exception
    ) as error:  # e.g., fluidsynth installed but the soundfont is not available
        print(f"Fluidsynth is not available ({error}); using additive synthesis")
    from partitura.utils.synth import synthesize

    return synthesize(
        notes, samplerate=samplerate, envelope_fun="linear", harmonic_dist="shepard"
    )


def excerpt(notes: np.ndarray, start: float = 0.0, end: float = None) -> np.ndarray:
    """Notes of a note array that start between ``start`` and ``end`` seconds (the times are not shifted)."""
    end = np.inf if end is None else end
    return notes[(notes["onset_sec"] >= start) & (notes["onset_sec"] < end)]


def plot_sequence(
    notes: np.ndarray,
    segments: np.ndarray = None,
    start: float = 0.0,
    end: float = None,
    time_div: int = 10,
    ax=None,
):
    """
    Piano roll (``partitura.utils.compute_pianoroll``, ``time_div`` pixels per second) of the notes of a sequence
    between ``start`` and ``end`` seconds. The segments (start, end, piece) are marked with dashed lines and the
    names of the pieces.
    """
    end = float(notes["onset_sec"].max() + 1) if end is None else end
    pianoroll = compute_pianoroll(
        notes, time_unit="sec", time_div=time_div, remove_silence=False, binary=True
    )
    columns = slice(int(start * time_div), int(end * time_div))
    pianoroll = pianoroll[:, columns].toarray()
    active = np.flatnonzero(pianoroll.any(axis=1))
    low, high = active.min() - 2, active.max() + 2
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.imshow(
        pianoroll[low : high + 1] > 0,
        origin="lower",
        aspect="auto",
        cmap="Greys",
        interpolation="nearest",
        extent=[start, start + pianoroll.shape[1] / time_div, low - 0.5, high + 0.5],
    )
    if segments is not None:
        for segment in segments:
            if start <= segment["start"] < end:
                ax.axvline(
                    segment["start"], color=FIRST_COLOR, linestyle="--", linewidth=1
                )
            if segment["end"] > start and segment["start"] < end:
                ax.text(
                    max(segment["start"], start) + 0.3,
                    high - 0.5,
                    segment["piecename"],
                    color=FIRST_COLOR,
                    rotation=0,
                    fontsize=8,
                    va="top",
                    ha="left",
                    clip_on=True,
                )
    ax.set_xlim(start, start + pianoroll.shape[1] / time_div)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI pitch")
    return ax


PLAYBACK_TEMPLATE = """
<div id="{uid}" style="max-width: 900px">
  <div style="position: relative; line-height: 0">
    <img src="data:image/png;base64,{image}" style="width: 100%; cursor: pointer">
    <div class="playhead" style="position: absolute; top: {top:.3f}%; height: {height:.3f}%; left: {left:.3f}%;
         width: 2px; background: #1f77b4; pointer-events: none"></div>
  </div>
  <audio controls style="width: 100%" src="data:audio/wav;base64,{audio}"></audio>
  <div class="piece" style="font-family: monospace; margin-top: 4px">&nbsp;</div>
</div>
<script>
(function() {{
  const root = document.getElementById("{uid}");
  const audio = root.querySelector("audio"), playhead = root.querySelector(".playhead");
  const label = root.querySelector(".piece"), image = root.querySelector("img");
  const segments = {segments};              // [start, end, piece] in seconds of the audio
  const x0 = {x0}, x1 = {x1}, axLeft = {ax_left}, axWidth = {ax_width};  // time range and position of the axes
  function update() {{
    const t = audio.currentTime;
    playhead.style.left = (100 * (axLeft + axWidth * (t + x0 - x0) / (x1 - x0))) + "%";
    const current = segments.find(s => t >= s[0] && t < s[1]);
    label.textContent = current ? "Playing: " + current[2] : "\u00a0";
    if (!audio.paused) requestAnimationFrame(update);
  }}
  ["play", "seeked", "timeupdate", "pause", "loadedmetadata"].forEach(e => audio.addEventListener(e, update));
  image.addEventListener("click", ev => {{      // click on the piano roll: go to that time
    const box = image.getBoundingClientRect();
    const fraction = ((ev.clientX - box.left) / box.width - axLeft) / axWidth;
    audio.currentTime = Math.min(Math.max(fraction, 0), 1) * (x1 - x0);
  }});
  update();
}})();
</script>
"""


def playback(
    notes: np.ndarray,
    segments: np.ndarray = None,
    start: float = 0.0,
    end: float = 60.0,
    samplerate: int = 22050,
    time_div: int = 10,
):
    """
    Piano roll with audio and a playhead, as in a DAW: the audio of the excerpt (``synthesize_notes``) is played with
    the usual controls, and the playhead moves over the piano roll. The dashed lines mark the changes of piece, and
    the name of the piece that is playing is shown below. Click on the piano roll to jump to that time.

    Returns an ``IPython.display.HTML`` (it needs a notebook that runs JavaScript, e.g., Jupyter or Colab).
    """
    import base64
    import io
    import json
    import uuid
    import wave

    from IPython.display import HTML

    fig, ax = plt.subplots(figsize=(11, 4))
    plot_sequence(notes, segments, start, end, time_div, ax=ax)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.14)
    x0, x1 = ax.get_xlim()
    position = ax.get_position()
    image = io.BytesIO()
    fig.savefig(
        image, format="png", dpi=100
    )  # without "tight": the position of the axes in the image is known
    plt.close(fig)

    shifted = excerpt(notes, start, end).copy()
    shifted["onset_sec"] -= start
    audio = synthesize_notes(shifted, samplerate=samplerate)
    pcm = np.int16(
        np.clip(audio / max(float(np.abs(audio).max()), 1e-9) * 0.9, -1, 1) * 32767
    )
    wav = io.BytesIO()
    with wave.open(wav, "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(samplerate)
        f.writeframes(pcm.tobytes())

    pieces = (
        []
        if segments is None
        else [
            [
                max(float(s["start"]) - start, 0.0),
                float(s["end"]) - start,
                str(s["piecename"]),
            ]
            for s in segments
            if s["end"] > start and s["start"] < end
        ]
    )
    return HTML(
        PLAYBACK_TEMPLATE.format(
            uid="playback" + uuid.uuid4().hex[:8],
            image=base64.b64encode(image.getvalue()).decode(),
            audio=base64.b64encode(wav.getvalue()).decode(),
            segments=json.dumps(pieces),
            x0=x0,
            x1=x1,
            top=100 * (1 - position.y1),
            height=100 * position.height,
            left=100 * position.x0,
            ax_left=position.x0,
            ax_width=position.width,
        )
    )


def simple_score() -> np.ndarray:
    """A very short melody (7 notes) to illustrate the tokens."""
    onsets = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0])
    pitches = np.array([60, 64, 67, 72, 67, 64, 60])
    notes = np.zeros(
        len(onsets),
        dtype=[("pitch", "i4"), ("onset_sec", "f8"), ("duration_sec", "f8")],
    )
    notes["pitch"], notes["onset_sec"], notes["duration_sec"] = pitches, onsets, 0.4
    return notes


def animate_tokenization(
    notes: np.ndarray,
    n1: int = 2,
    n2: int = 2,
    d: float = 0.05,
    interval: int = 1200,
    figsize=(10, 5.5),
):
    """
    Animation of the generation of the tokens of a note array. Each step shows a token: a first note (red), a
    second note (blue) among the next ``n1`` notes, and a third note (green) among the ``n2`` notes after the second
    one. The token has the three pitches and the ratio of the time differences between the notes (Δt1 / Δt2,
    rounded). The tokens that have been generated are listed below the piano roll.

    Returns a ``matplotlib.animation.FuncAnimation`` (use ``HTML(anim.to_jshtml())`` to show it in a notebook).
    """
    triplets = list(_iter_triplets(notes, d, n1, n2, None))
    pitches, onsets, durations = (
        notes["pitch"],
        notes["onset_sec"],
        notes["duration_sec"],
    )

    fig, (ax, table) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1.6]}
    )
    ax.set_xlim(-0.5, onsets.max() + 1.0)
    ax.set_ylim(pitches.min() - 4, pitches.max() + 5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI pitch")
    table.axis("off")
    rects = [
        ax.add_patch(
            Rectangle((o, p - 0.4), du, 0.8, facecolor=NOTE_COLOR, edgecolor="none")
        )
        for o, p, du in zip(onsets, pitches, durations)
    ]
    title = ax.set_title("")
    decorations = []
    listing = table.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=table.transAxes,
    )
    tokens = [
        f"({pitches[i]}, {pitches[j]}, {pitches[k]}, ratio {round(a / b)})"
        for i, j, k, a, b in triplets
    ]

    def draw(frame):
        for artist in decorations:
            artist.remove()
        decorations.clear()
        for rect in rects:
            rect.set_facecolor(NOTE_COLOR)
        i, j, k, td_1, td_2 = triplets[frame]
        for index, color in ((i, FIRST_COLOR), (j, SECOND_COLOR), (k, THIRD_COLOR)):
            rects[index].set_facecolor(color)
        for a, b, color, label in (
            (i, j, SECOND_COLOR, f"Δt1 = {td_1:.1f} s"),
            (j, k, THIRD_COLOR, f"Δt2 = {td_2:.1f} s"),
        ):
            y = max(pitches[a], pitches[b]) + 2.5
            decorations.append(
                ax.annotate(
                    "",
                    xy=(onsets[b], y),
                    xytext=(onsets[a], y),
                    arrowprops=dict(arrowstyle="<->", color=color),
                )
            )
            decorations.append(
                ax.text(
                    (onsets[a] + onsets[b]) / 2,
                    y + 0.4,
                    label,
                    color=color,
                    ha="center",
                    fontsize=9,
                )
            )
        title.set_text(
            f"Token {frame + 1} of {len(triplets)}: first note {pitches[i]}, second note {pitches[j]}, "
            f"third note {pitches[k]}, Δt1 / Δt2 = {td_1 / td_2:.2f} → {round(td_1 / td_2)}"
        )
        listing.set_text(
            "Tokens (pitch 1, pitch 2, pitch 3, ratio):\n"
            + "\n".join(tokens[max(0, frame - 5) : frame + 1])
        )

    draw(0)  # so that the layout has the space of the title
    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, draw, frames=len(triplets), interval=interval, blit=False
    )
    plt.close(fig)
    return anim


def query_matches(
    notes: np.ndarray,
    hash_table,
    d: float = 0.05,
    n1: int = 5,
    n2: int = 5,
    tolerance: float = 0.25,
):
    """
    Score tokens that match the tokens of a query (a note array), with the arrays ``piece`` (index of the piece in
    ``hash_table.pieces``), ``score_time`` and ``query_time`` (seconds from the first note of the query) of each match,
    and the tempo ``ratio`` of the score token and the query token.
    """
    triplets = np.array(generate_triplets(notes, d, n1, n2), dtype=np.float64).reshape(
        -1, 6
    )
    pitches = triplets[:, :3].astype(np.int64)
    token, entry = hash_table.match(
        pitches[:, 0],
        pitches[:, 1],
        pitches[:, 2],
        triplets[:, 4] / triplets[:, 5],
        tolerance,
    )
    return {
        "piece": hash_table.piece_id[entry],
        "score_time": hash_table.stime[entry],
        "query_time": triplets[token, 3] - float(np.min(notes["onset_sec"])),
        "ratio": hash_table.std1[entry] / triplets[token, 4],
    }


def plot_matches(
    notes: np.ndarray,
    hash_table,
    piece: str,
    bin_size: float = 1.0,
    zoom: bool = True,
    figsize=(10, 6),
    **token_args,
):
    """
    The two panels of Figure 3 of the paper, for the query ``notes`` (the correct piece is ``piece``):

    a) scatter plot of the matching tokens: score time against query time. The tokens that come from the same place of
       the score (and from the correct tempo) lie on a diagonal. The matches of the correct piece are blue, and the
       ones of the other pieces are gray.
    b) histogram of the difference between the score time and the query time (corrected for the tempo, in bins of
       ``bin_size`` seconds): the diagonal becomes a peak. The histogram of the correct piece is blue. The gray bars
       are the histogram of the other piece with the highest bin.
    """
    matches = query_matches(notes, hash_table, **token_args)
    correct = matches["piece"] == hash_table.pieces.index(piece)
    bins = (
        np.rint(
            (matches["score_time"] - matches["query_time"] * matches["ratio"])
            / bin_size
        ).astype(int)
        * bin_size
    )

    peak = None  # bin with the most tokens of the correct piece
    if correct.any():
        values, counts = np.unique(bins[correct], return_counts=True)
        peak = values[np.argmax(counts)]
    in_peak = correct & (bins == peak) if peak is not None else correct & False

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=figsize)
    ax_a.scatter(
        matches["score_time"][~correct],
        matches["query_time"][~correct],
        s=8,
        color="#cccccc",
        label="other pieces",
    )
    ax_a.scatter(
        matches["score_time"][correct & ~in_peak],
        matches["query_time"][correct & ~in_peak],
        s=10,
        color="#9ecae1",
        label=f"{piece}, other bins",
    )
    ax_a.scatter(
        matches["score_time"][in_peak],
        matches["query_time"][in_peak],
        s=18,
        color=FIRST_COLOR,
        label=f"{piece}, highest bin (the diagonal)",
    )
    if zoom and in_peak.any():
        ax_a.set_xlim(
            matches["score_time"][in_peak].min() - 1,
            matches["score_time"][in_peak].max() + 1,
        )
    ax_a.set_xlabel("Score time (seconds)")
    ax_a.set_ylabel("Query time (seconds)")
    ax_a.set_title(
        "a) Matching tokens"
        + (" (score time around the highest bin of the correct piece)" if zoom else "")
    )
    ax_a.legend(loc="upper left", fontsize=8)

    other = np.zeros(
        len(hash_table.pieces), dtype=int
    )  # highest bin of each of the other pieces
    for p in np.unique(matches["piece"][~correct]):
        other[p] = np.unique(bins[matches["piece"] == p], return_counts=True)[1].max()
    competitor = int(np.argmax(other))
    for name, selection, color in (
        (hash_table.pieces[competitor], matches["piece"] == competitor, "#bbbbbb"),
        (piece, correct, SECOND_COLOR),
    ):
        if selection.any():
            values, counts = np.unique(bins[selection], return_counts=True)
            ax_b.bar(
                values, counts, width=0.8 * bin_size, color=color, label=name, alpha=0.9
            )
    ax_b.set_xlabel("Score time - query time (seconds)")
    ax_b.set_ylabel("Token count")
    ax_b.set_title("b) Histogram for the diagonal identification")
    ax_b.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
