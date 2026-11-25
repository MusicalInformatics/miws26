"""Plots for the RuMEx notebook (03_rule_based_rendering.ipynb)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap
from partitura.utils import compute_pianoroll

PIANO_LOW = 21  # lowest pitch of `compute_pianoroll(..., piano_range=True)`


def velocity_pianoroll(ppart, time_div=20, end_time=None):
    """Piano roll of a performance (88 x time frames, value = MIDI velocity), time in seconds."""
    pr = compute_pianoroll(
        ppart,
        time_unit="sec",
        time_div=time_div,
        piano_range=True,
        remove_silence=False,
        end_time=end_time,
    )
    return np.asarray(pr.toarray() if hasattr(pr, "toarray") else pr)


def animate_velocity_pianoroll(
    render, values, melody_ids, seconds=20, time_div=20, interval=700, figsize=(10, 4.5)
):
    """
    Animation of the piano roll of a performance for different values of a rule parameter.

    The colour of each note is its MIDI velocity (fixed scale 0-127). Notes with velocity 127
    (where the velocity was clipped) are drawn in red and the melody notes have a white dot.

    Parameters
    ----------
    render : callable
        `render(value)` returns a `PerformedPart` (same seed for every value!).
    values : iterable
        Values of the parameter, one per frame.
    melody_ids : iterable
        Ids of the melody notes (the ids of the performed notes are the ones of the score notes).
    seconds : float
        Only the first `seconds` of the performance are shown.
    """
    values = list(values)
    rolls, perfs = [], []
    for v in values:
        ppart = render(v)
        perfs.append(ppart)
        rolls.append(velocity_pianoroll(ppart, time_div, end_time=seconds))

    n_cols = min(r.shape[1] for r in rolls)
    rolls = [r[:, :n_cols] for r in rolls]
    shown = np.array([i for i, n in enumerate(perfs[0].notes) if n["note_on"] <= seconds])
    on = np.array([perfs[0].notes[i]["note_on"] for i in shown])
    pitch = np.array([perfs[0].notes[i]["midi_pitch"] for i in shown])
    melody_ids = set(melody_ids)
    mel = np.array([perfs[0].notes[i]["id"] in melody_ids for i in shown])
    rows = np.where(np.any([r > 0 for r in rolls], axis=(0, 2)))[0]
    lo, hi = rows.min() - 1, rows.max() + 2
    extent = (0, n_cols / time_div, hi + PIANO_LOW - 0.5, lo + PIANO_LOW - 0.5)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(
        np.ma.masked_equal(rolls[0][lo:hi], 0),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=127,
        extent=extent,
        interpolation="nearest",
    )
    clipped = ax.imshow(
        np.ma.masked_less(rolls[0][lo:hi], 127),
        aspect="auto",
        cmap=ListedColormap(["#d62728"]),
        extent=extent,
        interpolation="nearest",
    )
    ax.invert_yaxis()
    ax.scatter(on[mel], pitch[mel], s=10, c="white", edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_facecolor("#f0f0f0")
    ax.set_xlabel("Performance time (s)")
    ax.set_ylabel("MIDI pitch")
    cbar = fig.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("MIDI velocity")
    title = ax.set_title("")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    def velocity_stats(i):
        v = np.array([perfs[i].notes[j]["velocity"] for j in shown])
        return v[mel].mean(), v[~mel].mean(), int((v == 127).sum())

    def draw(frame):
        roll = rolls[frame][lo:hi]
        img.set_data(np.ma.masked_equal(roll, 0))
        clipped.set_data(np.ma.masked_less(roll, 127))
        m, a, c = velocity_stats(frame)
        title.set_text(
            f"melody_relative_loudness = {values[frame]:+.2f}\n"
            f"mean velocity: melody {m:.0f}, accompaniment {a:.0f}; notes clipped at 127: {c}"
        )
        return img, clipped, title

    anim = animation.FuncAnimation(fig, draw, frames=len(values), interval=interval, blit=False)
    plt.close(fig)
    return anim

