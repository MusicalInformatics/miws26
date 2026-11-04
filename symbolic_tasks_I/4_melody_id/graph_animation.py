"""
Animation of the graph search that selects the melody line (Section 4.2 of the paper).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from melody_utils import BEAT_DIV, MELODY_COLOR, melody_graph, shortest_path_steps

BELOW_COLOR = "#e0e0e0"
NODE_COLOR = "#6baed6"
EDGE_COLOR = "#7f7f7f"
CURRENT_COLOR = "#1f77b4"


def animate_graph_search(note_probs, idx, threshold, start_beat=0, n_beats=8, interval=700, figsize=(11, 5)):
    """
    Animation of the shortest path search on the notes of an excerpt.

    Nodes are the notes above the threshold (blue; the notes below the threshold are light gray), plus a start
    node (S) and an end node (E). The algorithm goes through the nodes in order of onset. For each node, it
    looks at the notes that can follow it (blue dashed arrows) and, for each of them, keeps the best path that
    arrives at it (solid gray arrow). The number above each note is the highest total probability of a path
    from the start to that note. At the end, the best final note is selected and the path is followed
    backwards (red) to obtain the melody.

    Parameters
    ----------
    note_probs : probability of each note of the piece
    idx : positions of the notes in the piano roll (output of ``make_pianorolls``)
    threshold : probability threshold
    """
    lo, hi = start_beat * BEAT_DIV, (start_beat + n_beats) * BEAT_DIV
    in_excerpt = np.nonzero((idx[:, 1] >= lo) & (idx[:, 1] < hi))[0]
    sub_idx, sub_probs = idx[in_excerpt], note_probs[in_excerpt]

    candidates, probs, first, last, n_start = melody_graph(sub_probs, sub_idx, threshold)
    n = len(candidates)
    pitch = sub_idx[:, 0]
    x0 = (sub_idx[:, 1] - lo) / BEAT_DIV
    x1 = (sub_idx[:, 2] - lo) / BEAT_DIV

    # states of the search after each node
    states = [(k, best.copy(), previous.copy()) for k, best, previous in shortest_path_steps(probs, first, last, n_start)]
    best_final = states[-1][1]
    terminal = np.nonzero(first >= n)[0]
    k = int(terminal[np.argmax(best_final[terminal])])
    path = []
    while k != -1:
        path.append(k)
        k = states[-1][2][k]  # path from the last note back to the first one

    fig, ax = plt.subplots(figsize=figsize)
    ymid = (pitch.min() + pitch.max()) / 2
    ax.set_xlim(-1.2, n_beats + 1.2)
    ax.set_ylim(pitch.min() - 3, pitch.max() + 4)
    ax.set_xlabel("Score time (beats)")
    ax.set_ylabel("MIDI pitch")

    rects = []
    for i in range(len(sub_idx)):
        rect = Rectangle((x0[i], pitch[i] - 0.4), x1[i] - x0[i], 0.8, facecolor=BELOW_COLOR, edgecolor="none")
        ax.add_patch(rect)
        rects.append(rect)
    node_rect = {int(candidates[k]): rects[candidates[k]] for k in range(n)}
    for k in range(n):
        node_rect[int(candidates[k])].set_facecolor(NODE_COLOR)
    ax.add_patch(plt.Circle((-0.7, ymid), 0.5, color="k", zorder=3))
    ax.add_patch(plt.Circle((n_beats + 0.7, ymid), 0.5, color="k", zorder=3))
    ax.text(-0.7, ymid, "S", color="w", ha="center", va="center", fontsize=9, zorder=4)
    ax.text(n_beats + 0.7, ymid, "E", color="w", ha="center", va="center", fontsize=9, zorder=4)

    def anchor_out(k):  # where the arrows of node k start
        return (x1[candidates[k]], pitch[candidates[k]])

    def anchor_in(k):  # where the arrows to node k arrive
        return (x0[candidates[k]], pitch[candidates[k]])

    def arrow(p, q, **kwargs):
        patch = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, shrinkA=0, shrinkB=0, zorder=2, **kwargs)
        ax.add_patch(patch)
        return patch

    # all the possible arrows, created once and shown/hidden in each frame
    start_arrows = {j: arrow((-0.7, ymid), anchor_in(j), color=EDGE_COLOR, linewidth=1) for j in range(n_start)}
    end_arrows = {k: arrow(anchor_out(k), (n_beats + 0.7, ymid), color=EDGE_COLOR, linewidth=1) for k in terminal}
    edge_arrows = {(k, j): arrow(anchor_out(k), anchor_in(j), color=EDGE_COLOR, linewidth=1)
                   for k in range(n) for j in range(first[k], last[k])}
    for a in list(start_arrows.values()) + list(end_arrows.values()) + list(edge_arrows.values()):
        a.set_visible(False)
    labels = [ax.text((x0[candidates[k]] + x1[candidates[k]]) / 2, pitch[candidates[k]] + 1.0, "",
                      ha="center", fontsize=7, color="k") for k in range(n)]
    title = ax.set_title("Step 0: start node connected to the first notes")
    ax.legend(handles=[Line2D([0], [0], color=BELOW_COLOR, lw=6, label="below the threshold"),
                       Line2D([0], [0], color=NODE_COLOR, lw=6, label="above the threshold (node)"),
                       Line2D([0], [0], color=CURRENT_COLOR, lw=1.5, linestyle="--", label="notes that can follow the current note"),
                       Line2D([0], [0], color=EDGE_COLOR, lw=1.5, label="best path to each note"),
                       Line2D([0], [0], color=MELODY_COLOR, lw=2.5, label="melody (best path)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()

    n_forward = len(states)  # initial state + one frame for each node
    n_frames = n_forward + 1 + len(path)  # + selection of the final note + backtracking

    def draw(frame):
        for k in range(n):
            node_rect[int(candidates[k])].set_edgecolor("none")
            node_rect[int(candidates[k])].set_facecolor(NODE_COLOR)
        for a in list(start_arrows.values()) + list(end_arrows.values()) + list(edge_arrows.values()):
            a.set_visible(False)
            a.set_color(EDGE_COLOR)
            a.set_linewidth(1)
            a.set_linestyle("-")
        step = min(frame, n_forward - 1)
        current, best, previous = states[step]
        for k in range(n):
            labels[k].set_text(f"{best[k]:.1f}" if best[k] > -np.inf else "")
        for j in range(n_start):
            start_arrows[j].set_visible(True)
        # best arrow arriving at each note
        for j in range(n):
            if previous[j] != -1:
                edge_arrows[(previous[j], j)].set_visible(True)
        if frame < n_forward:
            if current == -1:
                title.set_text("Start: the start node is connected to the notes with the earliest onset")
            else:
                rect = node_rect[int(candidates[current])]
                rect.set_edgecolor("k")
                rect.set_linewidth(2)
                if first[current] >= n:
                    end_arrows[current].set_visible(True)
                    end_arrows[current].set_color(CURRENT_COLOR)
                    title.set_text("This note is the last one: it is connected to the end node")
                else:
                    for j in range(first[current], last[current]):
                        a = edge_arrows[(current, j)]
                        a.set_visible(True)
                        a.set_color(CURRENT_COLOR)
                        a.set_linestyle("--")
                    title.set_text("Look at the notes that can follow the current note; keep the best path to each of them")
        else:
            for k in terminal:
                end_arrows[k].set_visible(True)
                end_arrows[k].set_color(EDGE_COLOR)
            if frame == n_forward:
                node_rect[int(candidates[path[0]])].set_edgecolor(MELODY_COLOR)
                node_rect[int(candidates[path[0]])].set_linewidth(2)
                title.set_text(f"End node: choose the final note with the highest total probability ({best[path[0]]:.1f})")
            else:
                done = frame - n_forward  # number of notes of the path already followed backwards
                for a in edge_arrows.values():
                    a.set_visible(False)
                for k in path[:done]:
                    node_rect[int(candidates[k])].set_facecolor(MELODY_COLOR)
                for k_next, k_prev in zip(path[:done - 1], path[1:done]):
                    a = edge_arrows[(k_prev, k_next)]
                    a.set_visible(True)
                    a.set_color(MELODY_COLOR)
                    a.set_linewidth(2.5)
                    a.set_linestyle("-")
                end_arrows[path[0]].set_color(MELODY_COLOR)
                if done == len(path):  # the path reached the start node
                    start_arrows[path[-1]].set_color(MELODY_COLOR)
                title.set_text("Backtracking: follow the best path backwards from the end node" if done < len(path)
                               else "The melody is the best path")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)
    return anim
