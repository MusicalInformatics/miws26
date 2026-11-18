"""
Evaluation of the piece and score position identification, with the metrics of Table 3 of

    A. Arzt, S. Böck and G. Widmer (2012). "Fast Identification of Piece and Score Position via Symbolic
    Fingerprinting". ISMIR.

The performances are sequences in which the pianist switches from one piece to another (see ``utils.load_sequences``).
The sequence is processed online, and every ``query_length`` notes the system identifies the piece (and the position
in the score), with the notes of the window as query.

* The correct piece of a query is the piece of the segment of its last note (the piece that is being played when the
  system answers).
* The position is the score time of the last note of the query (the current position). As in the paper, it is correct
  if it is in the correct piece and within ``POSITION_TOLERANCE`` seconds of a place of the score where the content
  is identical (the same notes, in the last ``max(query length, 30)`` notes) to the content at the real position,
  which is the score time of the note that is aligned to the last note of the query (the last note of the window of
  the correct piece that has an alignment and a real onset). Repeated passages are therefore not errors. The scores
  have a constant tempo, so the score time is in seconds, as in the paper.

For each query length: the percentage of queries in which the correct piece (position) is the top match, or among
the top 2 to 5 matches; the mean duration of the queries (seconds), and the number of queries.
"""

import numpy as np

from fingerprinting import OnlineFingerprinting

QUERY_LENGTHS = (5, 10, 20, 30, 40, 50, 60)
TOP_K = (1, 2, 3, 4, 5)
POSITION_TOLERANCE = 1.5  # seconds
MIN_CONTENT_NOTES = 30


def note_pieces(onsets: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Piece (of the segment) of each note, given the onsets and the segments of a sequence."""
    index = np.searchsorted(segments["start"], onsets + 1e-3, side="right") - 1
    return segments["piecename"][np.clip(index, 0, len(segments) - 1)]


class ScoreInfo(object):
    """Onsets and pitches of the notes of a score, and the content of the score at each of its notes."""

    def __init__(self, notes: np.ndarray):
        self.onsets = notes["onset_sec"]
        self.pitches = notes["pitch"]
        self.index = {note_id: k for k, note_id in enumerate(notes["id"])}
        self._signatures = {}

    def signature(self, k: int, length: int) -> tuple:
        """Content of the score at the onset of the note ``k``: the last ``length`` notes up to that onset."""
        if (k, length) not in self._signatures:
            end = np.searchsorted(self.onsets, self.onsets[k] + 1e-6, side="right")
            start = max(0, end - length)
            self._signatures[(k, length)] = tuple(
                zip(
                    self.pitches[start:end].tolist(),
                    np.round(self.onsets[start:end] - self.onsets[k], 2).tolist(),
                )
            )
        return self._signatures[(k, length)]

    def is_correct(
        self, position: float, true_note: int, length: int, strict: bool = False
    ) -> bool:
        """Whether the position is at most ``POSITION_TOLERANCE`` from the true note, or from a place with its content."""
        lo = np.searchsorted(self.onsets, position - POSITION_TOLERANCE, side="left")
        hi = np.searchsorted(self.onsets, position + POSITION_TOLERANCE, side="right")
        if strict:
            return lo <= true_note < hi
        true_signature = self.signature(true_note, length)
        return any(self.signature(k, length) == true_signature for k in range(lo, hi))


def aligned_score_notes(
    notes: np.ndarray, note_map: np.ndarray, score_info: dict
) -> np.ndarray:
    """Index in its score of the score note aligned to each note of the sequence (-1: no alignment or not a real onset)."""
    row = {i: k for k, i in enumerate(note_map["id"])}
    indices = np.full(len(notes), -1)
    for k, note_id in enumerate(notes["id"]):
        m = note_map[row[note_id]]
        if m["score_id"] != "" and not m["clipped"]:
            indices[k] = score_info[m["piecename"]].index[m["score_id"]]
    return indices


def evaluate_sequence(
    notes: np.ndarray,
    segments: np.ndarray,
    hash_table: dict,
    query_length: int,
    model=OnlineFingerprinting,
    note_map: np.ndarray = None,
    score_info: dict = None,
    strict_position: bool = False,
    **token_args,
):
    """
    Run the online fingerprinting (``model``: ``OnlineFingerprintingVoting`` or ``OnlineFingerprinting``, with its hash
    table) with windows of ``query_length`` notes on a sequence.

    Returns the rank of the correct piece (``inf`` if it has no votes), the duration (seconds), whether all notes
    come from one segment ("single piece") of each query, and the position results ``(has_truth, correct)``:
    whether the query has a true position, and whether the correct position is among the top k matches
    (``len(TOP_K)`` x queries). The last two are ``None`` if the model gives no positions or there is no ``note_map``.
    """
    fingerprinting = model(hash_table, window_size=query_length, **token_args)
    fingerprinting.run_offline(notes)
    n_queries = (
        len(notes) // query_length
    )  # the last window is not a complete query if it has fewer notes
    pieces = note_pieces(notes["onset_sec"], segments)
    with_positions = (
        hasattr(fingerprinting, "matches")
        and note_map is not None
        and score_info is not None
    )
    if with_positions:
        true_notes = aligned_score_notes(notes, note_map, score_info)

    ranks, durations, single = [], [], []
    has_truth, correct_position = [], []
    for i in range(n_queries):
        window = slice(i * query_length, (i + 1) * query_length)
        ranking = fingerprinting.rankings[i]
        correct = pieces[window][-1]
        ranks.append(ranking.index(correct) + 1 if correct in ranking else np.inf)
        durations.append(notes["onset_sec"][window][-1] - notes["onset_sec"][window][0])
        single.append(len(set(pieces[window])) == 1)
        if with_positions:
            usable = np.flatnonzero(
                (pieces[window] == correct) & (true_notes[window] >= 0)
            )
            has_truth.append(len(usable) > 0)
            matches = fingerprinting.matches[i]
            if len(usable):
                true_note = true_notes[window][usable[-1]]
                length = max(query_length, MIN_CONTENT_NOTES)
                ok = [
                    piece == correct
                    and score_info[piece].is_correct(
                        position, true_note, length, strict_position
                    )
                    for piece, position, _ in matches
                ]
            else:
                ok = []
            correct_position.append([any(ok[:k]) for k in TOP_K])
    position_results = (
        (
            np.array(has_truth, bool),
            np.array(correct_position, bool).reshape(-1, len(TOP_K)).T,
        )
        if with_positions
        else (None, None)
    )
    return (
        np.array(ranks),
        np.array(durations),
        np.array(single, bool),
    ) + position_results


def evaluation(
    sequences: list,
    hash_table: dict,
    query_lengths: tuple = QUERY_LENGTHS,
    only_single_piece: bool = False,
    model=OnlineFingerprinting,
    scores: dict = None,
    strict_position: bool = False,
    **token_args,
) -> dict:
    """
    Results for each query length (the columns of Table 3 of the paper), on a list of ``(notes, segments, note_map)``.

    With ``only_single_piece`` only the queries whose notes come from one segment are used. ``model`` is the class
    of the online fingerprinting (``OnlineFingerprintingVoting`` or ``OnlineFingerprinting``). The position results
    need a model with positions (``OnlineFingerprinting``), the note maps of the sequences and the ``scores`` (``utils.load_scores``).
    With ``strict_position`` the position must be near the real one, and not only near a place with the same content.
    """
    score_info = (
        None
        if scores is None
        else {piece: ScoreInfo(notes) for piece, notes in scores.items()}
    )
    token_args = {"n1": 5, "n2": 5, **token_args}
    results = {
        "query_lengths": tuple(query_lengths),
        "n_queries": [],
        "piece_in_top_k": [],
        "duration": [],
        "position_in_top_k": [],
        "n_position_queries": [],
    }
    for n in query_lengths:
        parts = [
            evaluate_sequence(
                notes,
                segments,
                hash_table,
                n,
                model,
                note_map,
                score_info,
                strict_position,
                **token_args,
            )
            for notes, segments, note_map in sequences
        ]
        ranks, durations, single = (np.concatenate(p) for p in list(zip(*parts))[:3])
        keep = single if only_single_piece else np.ones(len(ranks), bool)
        results["n_queries"].append(int(keep.sum()))
        results["piece_in_top_k"].append([(ranks[keep] <= k).mean() for k in TOP_K])
        results["duration"].append(durations[keep].mean())
        if parts[0][3] is not None:
            has_truth = np.concatenate([p[3] for p in parts])
            correct = np.concatenate([p[4] for p in parts], axis=1)
            keep_position = keep & has_truth
            results["n_position_queries"].append(int(keep_position.sum()))
            results["position_in_top_k"].append(correct[:, keep_position].mean(axis=1))
    results["piece_in_top_k"] = np.array(
        results["piece_in_top_k"]
    ).T  # (len(TOP_K), len(query_lengths))
    results["position_in_top_k"] = (
        np.array(results["position_in_top_k"]).T
        if results["position_in_top_k"]
        else None
    )
    return results


def format_evaluation(results: dict) -> str:
    """Text table with the rows of Table 3 of the paper."""
    rows = [("Query Length in Notes", [f"{n}" for n in results["query_lengths"]])]
    for i, k in enumerate(TOP_K):
        label = "Corr. Piece as Top Match" if k == 1 else f"Corr. Piece in Top {k}"
        rows.append((label, [f"{100 * v:.2f}%" for v in results["piece_in_top_k"][i]]))
    if results["position_in_top_k"] is not None:
        for i, k in enumerate(TOP_K):
            label = (
                "Corr. Position as Top Match"
                if k == 1
                else f"Corr. Position in Top {k}"
            )
            rows.append(
                (label, [f"{100 * v:.2f}%" for v in results["position_in_top_k"][i]])
            )
    rows.append(("Mean Query Duration", [f"{v:.2f} sec" for v in results["duration"]]))
    rows.append(("Number of Queries", [f"{n}" for n in results["n_queries"]]))
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        f"{label:{width}s}  " + "  ".join(f"{c:>9s}" for c in cells)
        for label, cells in rows
    )
