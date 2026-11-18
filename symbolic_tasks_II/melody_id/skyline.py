"""
The skyline algorithm, the simplest baseline for melody identification.

    A. L. Uitdenbogerd and J. Zobel (1999). "Melodic matching techniques for large music
    databases". In Proceedings of the 7th ACM International Conference on Multimedia.

The skyline algorithm assumes that the melody is always the highest voice: at each
onset, the note with the highest pitch is the melody note and all the other notes
that start at the same time are accompaniment.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Patch, Rectangle

from melody_utils import MELODY_COLOR

PENDING_COLOR = "#d9d9d9"
ACCOMPANIMENT_COLOR = "#4d4d4d"
CANDIDATE_COLOR = "#1f77b4"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_name(pitch: int) -> str:
    """Name of a MIDI pitch (e.g., 60 -> 'C4')."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def onset_groups(onset: np.ndarray, decimals: int = 3):
    """Indices of the notes that start at each (distinct) onset. Returns (onsets, list of index arrays)."""
    onset = np.round(np.asarray(onset, dtype=float), decimals)
    unique_onsets = np.unique(onset)
    return unique_onsets, [np.nonzero(onset == u)[0] for u in unique_onsets]


def skyline(notes: np.ndarray) -> np.ndarray:
    """
    Skyline algorithm.

    Parameters
    ----------
    notes : structured array
        Notes with the fields ``pitch`` and ``onset_beat``.

    Returns
    -------
    is_melody : bool array (n_notes,)
        True for the highest note of each onset (the first one, if there are several with the same pitch).
    """
    pitch = np.asarray(notes["pitch"])
    is_melody = np.zeros(len(pitch), dtype=bool)
    _, groups = onset_groups(notes["onset_beat"])
    for group in groups:
        is_melody[group[np.argmax(pitch[group])]] = True
    return is_melody


def animate_skyline(notes: np.ndarray, interval: int = 700, figsize=(10, 4.5)):
    """
    Animation of the skyline algorithm on a (short) excerpt.

    The algorithm moves from onset to onset. For each onset, it first looks at the notes that
    start at that moment (blue outline), and then selects the highest one as melody (red); the
    other notes become accompaniment (dark gray).

    Parameters
    ----------
    notes : structured array with fields ``pitch``, ``onset_beat`` and ``duration_beat``
    interval : int
        Milliseconds between frames.

    Returns
    -------
    matplotlib.animation.FuncAnimation (use ``HTML(anim.to_jshtml())`` to show it in a notebook).
    """
    pitch = np.asarray(notes["pitch"])
    onset = np.asarray(notes["onset_beat"], dtype=float)
    duration = np.asarray(notes["duration_beat"], dtype=float)
    onsets, groups = onset_groups(onset)
    melody = skyline(notes)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, (onset + duration).max())
    ax.set_ylim(pitch.min() - 2.5, pitch.max() + 2.5)
    ax.set_xlabel("Score time (beats)")
    ax.set_ylabel("MIDI pitch")
    rects = []
    for p, o, d in zip(pitch, onset, duration):
        rect = Rectangle((o, p - 0.4), d * 0.95, 0.8, facecolor=PENDING_COLOR, edgecolor="none")
        ax.add_patch(rect)
        rects.append(rect)
    cursor = ax.axvline(0, color="k", linestyle="--", linewidth=1)
    title = ax.set_title("Beat 00.0: notes that start now: " + ", ".join(["C#10"] * 4))  # placeholder, reserves space
    ax.legend(handles=[Patch(color=PENDING_COLOR, label="not processed yet"),
                       Patch(facecolor="white", edgecolor=CANDIDATE_COLOR, linewidth=2, label="notes at this onset"),
                       Patch(color=MELODY_COLOR, label="melody (highest note)"),
                       Patch(color=ACCOMPANIMENT_COLOR, label="accompaniment")],
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()

    def draw(frame):
        step, selecting = divmod(frame, 2)  # two frames per onset: look at the candidates, select
        for i, rect in enumerate(rects):
            rect.set_facecolor(PENDING_COLOR)
            rect.set_edgecolor("none")
            rect.set_linewidth(0)
        # notes of the previous onsets (already decided)
        for group in groups[:step]:
            for i in group:
                rects[i].set_facecolor(MELODY_COLOR if melody[i] else ACCOMPANIMENT_COLOR)
        group = groups[step]
        for i in group:
            if selecting:
                rects[i].set_facecolor(MELODY_COLOR if melody[i] else ACCOMPANIMENT_COLOR)
            else:
                rects[i].set_edgecolor(CANDIDATE_COLOR)
                rects[i].set_linewidth(2.5)
        cursor.set_xdata([onsets[step], onsets[step]])
        names = ", ".join(midi_name(p) for p in sorted(pitch[group], reverse=True))
        if selecting:
            title.set_text(f"Beat {onsets[step]:g}: the highest note ({midi_name(pitch[group].max())}) is the melody")
        else:
            title.set_text(f"Beat {onsets[step]:g}: notes that start now: {names}")
        return rects + [cursor, title]

    anim = animation.FuncAnimation(fig, draw, frames=2 * len(groups), interval=interval, blit=False)
    plt.close(fig)  # do not show the static figure in the notebook
    return anim
