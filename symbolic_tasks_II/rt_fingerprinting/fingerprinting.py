"""
Piece identification by fingerprinting with three-note tokens.

A token is made of three notes (pitch 1, pitch 2, pitch 3) and the ratio between the time differences of the notes
(onset 2 - onset 1) / (onset 3 - onset 2), rounded. The ratio is the same for a score and for a performance of it
played at a different tempo. The hash table maps each token to the pieces whose score contains it.

Two ways of identifying the piece with the tokens of a query:

With ``intervals=True`` the pitches of a token are replaced by the intervals between them (pitch 2 - pitch 1 and
pitch 3 - pitch 2), so that a transposed query has the same tokens as the score (in both systems).

* Voting (``OnlineFingerprintingVoting``): the tokens of the query vote for the pieces in which they appear (``count_votes``).
* Histograms (``OnlineFingerprinting``, as in Arzt, Böck and Widmer, 2012): the hash table also stores the position (time) of each token in the score.
  The matching tokens of a piece are sorted in a histogram of the difference between the score time and the query
  time (corrected for the tempo): the tokens that come from the same place of the score fall in the same bin, and
  the score of a piece is the number of tokens in its highest bin (``match_histograms``).
"""

import math
from collections import Counter

import numpy as np

from utils import load_note_array, piece_name, should_unfold


def round_to(x: float, a: int = 2) -> float:
    """Round to the closest multiple of 1 / a."""
    return round(x * a) / a


def _next_note(
    onsets: np.ndarray,
    pitches: np.ndarray,
    start: int,
    reference: int,
    d: float,
    max_distance,
):
    """
    Index of the first note from ``start`` that is more than ``d`` seconds after the reference note, and at most
    ``max_distance`` semitones away from it (``None``: no limit).
    """
    for k in range(start, len(onsets)):
        if round(onsets[k] - onsets[reference], 2) > d and (
            max_distance is None
            or abs(int(pitches[k]) - int(pitches[reference])) <= max_distance
        ):
            return k
    return None


def _iter_triplets(notes: np.ndarray, d: float, n1: int, n2: int, max_pitch_distance):
    """
    Triplets as indices ``(i, j, k, time difference 1-2, time difference 2-3)`` of the notes of a note array. The time
    differences are computed in double precision whatever the type of the note array (partitura has floats of 32 bits),
    so that the tokens of a score and of a query are computed in the same way (the rounding of the ratio of the time
    differences depends on the last bits).
    """
    pitches = notes["pitch"]
    onsets = np.asarray(notes["onset_sec"], dtype=np.float64)
    for i in range(len(notes)):
        j = i + 1
        for _ in range(n1):
            j = _next_note(onsets, pitches, j, i, d, max_pitch_distance)
            if j is None:
                break
            td_1 = round(onsets[j] - onsets[i], 2)
            k = j + 1
            for _ in range(n2):
                k = _next_note(onsets, pitches, k, j, d, max_pitch_distance)
                if k is None:
                    break
                yield i, j, k, td_1, round(onsets[k] - onsets[j], 2)
                k += 1
            j += 1


def generate_triplets(
    notes: np.ndarray,
    d: float = 0.05,
    n1: int = 5,
    n2: int = 5,
    max_pitch_distance: int = None,
) -> list:
    """
    Triplets of notes of a note array (with the fields ``pitch`` and ``onset_sec``, sorted by onset).

    Parameters
    ----------
    notes : structured array
        Partitura note array of a score or a performance.
    d : float
        Minimum time difference (seconds) between the notes of a triplet (so that the notes of a chord are not used
        together).
    n1, n2 : int
        Number of notes that are tried as second note (after the first) and as third note (after the second).
    max_pitch_distance : int
        Maximum distance in semitones between the notes of the pairs (note 1 and note 2, note 2 and note 3); 24 are
        two octaves, as in the paper (Arzt et al., 2012). A note that is too far is skipped, and does not count
        for ``n1`` or ``n2``. ``None`` (default): no limit.

    Returns
    -------
    triplets : list of tuples (pitch 1, pitch 2, pitch 3, onset 1, time difference 1-2, time difference 2-3)
    """
    pitches = notes["pitch"]
    onsets = np.asarray(notes["onset_sec"], dtype=np.float64)
    return [
        (
            int(pitches[i]),
            int(pitches[j]),
            int(pitches[k]),
            float(onsets[i]),
            float(td_1),
            float(td_2),
        )
        for i, j, k, td_1, td_2 in _iter_triplets(notes, d, n1, n2, max_pitch_distance)
    ]


def generate_tokens(
    notes: np.ndarray,
    d: float = 0.05,
    n1: int = 5,
    n2: int = 5,
    round_up: str = "Low",
    intervals: bool = False,
    max_pitch_distance: int = None,
) -> list:
    """
    Three-note tokens of a note array (see ``generate_triplets`` for the parameters).

    ``round_up`` is the rounding of the time difference ratio: "Low" (integers), "Med" (multiples of 1/2) or
    "High" (multiples of 1/4). With ``intervals`` the key has the intervals between the pitches, and not the pitches.

    Returns
    -------
    tokens : list of (key, onset) tuples, where key = (pitch 1, pitch 2, pitch 3, ratio) (or (pitch 2 - pitch 1,
        pitch 3 - pitch 2, ratio)) and onset is the onset of the first note.
    """
    tokens = []
    for p1, p2, p3, onset, td_1, td_2 in generate_triplets(
        notes, d, n1, n2, max_pitch_distance
    ):
        ratio = td_1 / td_2
        if round_up == "High":
            ratio = round_to(ratio, 4)
        elif round_up == "Med":
            ratio = round_to(ratio, 2)
        else:
            ratio = round(ratio)
        tokens.append(
            (((p2 - p1, p3 - p2, ratio) if intervals else (p1, p2, p3, ratio)), onset)
        )
    return tokens


def update_hash_table(hash_table: dict, tokens: list, name: str) -> dict:
    """Add the tokens of the piece ``name`` to the hash table (a dict: key -> list of pieces)."""
    for key, _ in tokens:
        pieces = hash_table.setdefault(key, [])
        if name not in pieces:
            pieces.append(name)
    return hash_table


def build_hash_table(score_files: list, unfold=False, **token_args) -> dict:
    """
    Hash table of a list of score files. The name of each piece is the name of its file. ``unfold``: bool or
    collection of the score files that are unfolded (see ``utils.should_unfold``).
    """
    hash_table = {}
    for path in score_files:
        notes = load_note_array(path, unfold=should_unfold(path, unfold))
        update_hash_table(
            hash_table, generate_tokens(notes, **token_args), piece_name(path)
        )
    return hash_table


def count_votes(
    notes: np.ndarray,
    hash_table: dict,
    d: float = 0.05,
    n1: int = 1,
    n2: int = 1,
    round_up: str = "Low",
    intervals: bool = False,
    max_pitch_distance: int = None,
) -> Counter:
    """Number of tokens of the note array that are in each piece of the hash table."""
    votes = Counter()
    for key, _ in generate_tokens(
        notes,
        d=d,
        n1=n1,
        n2=n2,
        round_up=round_up,
        intervals=intervals,
        max_pitch_distance=max_pitch_distance,
    ):
        votes.update(hash_table.get(key, []))
    return votes


def rank_pieces(notes: np.ndarray, hash_table: dict, **token_args) -> list:
    """Pieces with at least one vote for the note array of a query, from most to least votes."""
    return [
        piece for piece, _ in count_votes(notes, hash_table, **token_args).most_common()
    ]


def identify_piece(notes: np.ndarray, hash_table: dict, **token_args):
    """Piece with most votes for the note array of a performance (``None`` if no token is in the hash table)."""
    votes = count_votes(notes, hash_table, **token_args)
    return votes.most_common(1)[0][0] if votes else None


_BUCKET_STEP = math.log(1.25)
_BUCKET_OFFSETS = np.array(
    [-2, -1, 0, 1]
)  # the buckets of the ratios within the tolerance of ``match``


def _ratio_buckets(ratios: np.ndarray) -> np.ndarray:
    """Buckets of the time difference ratios, in steps of a factor 1.25 (the tolerance of the matching)."""
    return np.floor(np.log(ratios) / _BUCKET_STEP).astype(np.int64)


def _key_codes(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, buckets: np.ndarray, intervals: bool
) -> np.ndarray:
    """Integer key of a token: the pitches (or the intervals) and the bucket of the ratio."""
    if intervals:
        pitch_code = ((p2 - p1 + 127) << 8) | (p3 - p2 + 127)
    else:
        pitch_code = (p1 << 14) | (p2 << 7) | p3
    return (pitch_code << 8) | (buckets + 128)


class Histograms(object):
    """
    Histograms of the score positions of the matching tokens: arrays with the ``piece`` (index in ``names``), the
    ``position`` (score time in seconds, the center of the bin) and the ``count`` of tokens of each bin.
    """

    def __init__(
        self, names: list, piece: np.ndarray, position: np.ndarray, count: np.ndarray
    ):
        self.names, self.piece, self.position, self.count = (
            names,
            piece,
            position,
            count,
        )

    def ranking(self) -> list:
        """Pieces with matching tokens, from the highest bin to the lowest."""
        best = np.zeros(len(self.names), dtype=np.int64)
        np.maximum.at(best, self.piece, self.count)
        pieces = np.flatnonzero(best > 0)
        return [self.names[p] for p in pieces[np.argsort(-best[pieces], kind="stable")]]

    def best_matches(self, max_matches: int = 5, suppression: float = 1.5) -> list:
        """
        The best ``(piece, position, number of tokens)``, from most to least tokens. A bin that is within
        ``suppression`` seconds of a better bin of the same piece is skipped (it is the same place of the score).
        """
        matches = []
        for k in np.lexsort((self.position, self.piece, -self.count)):
            piece, position = int(self.piece[k]), float(self.position[k])
            if all(
                p != piece or abs(position - q) > suppression for p, q, _ in matches
            ):
                matches.append((piece, position, int(self.count[k])))
                if len(matches) == max_matches:
                    break
        return [(self.names[p], q, c) for p, q, c in matches]


def histograms_from_positions(
    names: list, piece: np.ndarray, position: np.ndarray, bin_size: float = 1.0
) -> Histograms:
    """Histograms (bins of ``bin_size`` seconds) of the score positions of the matching tokens of each piece."""
    bins = np.rint(position / bin_size).astype(np.int64)
    codes, counts = np.unique(
        (piece.astype(np.int64) << 32) + (bins + (1 << 31)), return_counts=True
    )
    return Histograms(
        names, codes >> 32, ((codes & ((1 << 32) - 1)) - (1 << 31)) * bin_size, counts
    )


class HashTableV2(object):
    """
    Hash table of the scores for ``OnlineFingerprinting``, as arrays sorted by key. Each entry is a token of a score, with its piece,
    score time (of the first note), time difference 1-2 and time difference ratio. The key has the pitches of the
    token (or the intervals between them, with ``intervals``) and a bucket of the ratio, so that the tokens with a
    similar ratio can be found fast.
    """

    def __init__(
        self,
        score_files: list,
        d: float = 0.05,
        n1: int = 5,
        n2: int = 5,
        unfold=False,
        intervals: bool = False,
        max_pitch_distance: int = None,
    ):
        self.intervals = intervals
        self.pieces = [piece_name(path) for path in score_files]
        tokens = []
        for piece, path in enumerate(score_files):
            notes = load_note_array(path, unfold=should_unfold(path, unfold))
            triplets = np.array(
                generate_triplets(notes, d, n1, n2, max_pitch_distance),
                dtype=np.float64,
            )
            tokens.append(
                np.column_stack(
                    [triplets.reshape(-1, 6), np.full(len(triplets), piece)]
                )
            )
        tokens = np.concatenate(tokens)
        pitches = tokens[:, :3].astype(np.int64)
        ratio = tokens[:, 4] / tokens[:, 5]
        codes = _key_codes(
            pitches[:, 0],
            pitches[:, 1],
            pitches[:, 2],
            _ratio_buckets(ratio),
            intervals,
        )
        order = np.argsort(codes, kind="stable")
        self.codes = codes[order]
        self.piece_id = tokens[order, 6].astype(np.int32)
        self.stime = tokens[order, 3]
        self.std1 = tokens[order, 4]
        self.stdr = ratio[order]

    def __len__(self):
        return len(self.codes)

    def match(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        qtdr: np.ndarray,
        tolerance: float = 0.25,
    ):
        """
        Entries that match query tokens: same pitches (or intervals) and a ratio that differs at most ``tolerance``
        times the ratio of the query. Returns the index of the query token and of the entry of each match.
        """
        codes = (
            _key_codes(p1, p2, p3, _ratio_buckets(qtdr), self.intervals)[:, None]
            + _BUCKET_OFFSETS[None, :]
        )
        codes = codes.ravel()
        low = np.searchsorted(self.codes, codes, side="left")
        counts = np.searchsorted(self.codes, codes, side="right") - low
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        token = np.repeat(np.arange(len(codes)) // len(_BUCKET_OFFSETS), counts)
        entry = np.repeat(low - (np.cumsum(counts) - counts), counts) + np.arange(total)
        keep = np.abs(self.stdr[entry] - qtdr[token]) <= tolerance * qtdr[token]
        return token[keep], entry[keep]


def build_hash_table_v2(
    score_files: list,
    d: float = 0.05,
    n1: int = 5,
    n2: int = 5,
    unfold=False,
    intervals: bool = False,
    max_pitch_distance: int = None,
) -> HashTableV2:
    """Hash table of the scores for ``OnlineFingerprinting`` (see ``HashTableV2``). ``unfold``: see ``utils.should_unfold``."""
    return HashTableV2(score_files, d, n1, n2, unfold, intervals, max_pitch_distance)


def match_histograms(
    notes: np.ndarray,
    hash_table: HashTableV2,
    d: float = 0.05,
    n1: int = 5,
    n2: int = 5,
    bin_size: float = 1.0,
    tolerance: float = 0.25,
    intervals: bool = None,
    max_pitch_distance: int = None,
) -> Histograms:
    """
    Match the tokens of a query (note array) with the score tokens of the hash table (``build_hash_table_v2``).

    A score token matches a query token if they have the same pitches and the time difference ratios differ at most
    ``tolerance`` times the ratio of the query. The tempo ratio of the two tokens (score time difference / query
    time difference) is used to estimate the score time of the last note of the query (the current position):
    score time of the token + (query time of the last note - query time of the token) * tempo ratio.
    This time, in bins of ``bin_size`` seconds, is the bin of the histogram of the piece.
    """
    assert (
        intervals is None or intervals == hash_table.intervals
    ), "the query tokens need the type of the table"
    triplets = np.array(
        generate_triplets(notes, d, n1, n2, max_pitch_distance), dtype=np.float64
    ).reshape(-1, 6)
    if len(triplets) == 0:
        return Histograms(
            hash_table.pieces, np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
        )
    pitches = triplets[:, :3].astype(np.int64)
    token, entry = hash_table.match(
        pitches[:, 0],
        pitches[:, 1],
        pitches[:, 2],
        triplets[:, 4] / triplets[:, 5],
        tolerance,
    )
    ratio = (
        hash_table.std1[entry] / triplets[token, 4]
    )  # tempo ratio of the score and the query tokens
    position = (
        hash_table.stime[entry]
        + (np.max(notes["onset_sec"]).astype(np.float64) - triplets[token, 3]) * ratio
    )
    return histograms_from_positions(
        hash_table.pieces, hash_table.piece_id[entry], position, bin_size
    )


def rank_matches(
    histograms: Histograms, max_matches: int = 5, suppression: float = 1.5
) -> list:
    """The best ``(piece, position, number of tokens)`` of the histograms (see ``Histograms.best_matches``)."""
    return histograms.best_matches(max_matches, suppression)


class OnlineFingerprintingVoting(object):
    """
    Online piece identification with votes: every time ``window_size`` notes have arrived, the pieces are ranked by the
    number of tokens of these notes that are in their score (``rank_pieces``). The pitches of a chord count as
    separate notes.

    The observations are the ones of the ``PitchProcessor`` of matchmaker (with ``return_pitch_list=True``): tuples
    ``(pitches, time)`` with the pitches that start at ``time`` (seconds since the beginning of the stream).
    As in the score followers of matchmaker, the observations are processed one by one with ``__call__``, or
    taken from the ``queue`` of a stream with ``run``.

    Parameters
    ----------
    hash_table : dict
        Hash table of the scores (``build_hash_table``).
    window_size : int
        Number of notes of a window.
    hop : int
        Number of notes between the beginnings of two windows (by default ``window_size``: consecutive windows).
    queue : RECVQueue
        Queue of the stream (only for ``run``).
    token_args :
        Arguments of ``generate_tokens`` (``d``, ``n1``, ``n2`` and ``round_up``) for the tokens of the stream.

    Attributes
    ----------
    predictions : list of ``(start, end, piece)`` of each window.
    rankings : list with all the pieces with votes (from most to least) of each window.
    """

    def __init__(
        self,
        hash_table: dict,
        window_size: int = 50,
        hop: int = None,
        queue=None,
        **token_args
    ):
        self.hash_table = hash_table
        self.window_size = window_size
        self.hop = window_size if hop is None else hop
        self.queue = queue
        self.token_args = {"n1": 1, "n2": 1, **token_args}
        self.reset()

    def reset(self):
        self.pitches = []
        self.onsets = []
        self.predictions = []
        self.rankings = []

    def __call__(self, pitches, time):
        """
        Process an observation of the stream: the ``pitches`` that start at ``time``.

        Returns the prediction ``(start, end, piece)`` of the last window that was completed with this observation
        (``start`` and ``end`` are the onsets of its first and last note; the piece is ``None`` if no token of the
        window is in the hash table), or ``None`` if no window was completed.
        """
        completed = None
        for pitch in np.atleast_1d(pitches):
            self.pitches.append(int(pitch))
            self.onsets.append(float(time))
            if len(self.pitches) == self.window_size:
                completed = self._predict()
                del self.pitches[: self.hop], self.onsets[: self.hop]
        return completed

    def run(self):
        """Take the observations from the queue until the end of the stream, and yield the predictions of the windows."""
        from matchmaker.io.stream import STREAM_END

        while True:
            item = self.queue.get()
            if item is STREAM_END:
                break
            if item is None:
                continue
            prediction = self(*item)
            if prediction is not None:
                yield prediction
        if self.pitches:
            yield self._predict()

    def _rank(self, notes: np.ndarray) -> list:
        return rank_pieces(notes, self.hash_table, **self.token_args)

    def _predict(self):
        notes = np.zeros(
            len(self.pitches), dtype=[("onset_sec", "f8"), ("pitch", "i4")]
        )
        notes["onset_sec"] = self.onsets
        notes["pitch"] = self.pitches
        notes = notes[np.lexsort((notes["pitch"], notes["onset_sec"]))]
        ranking = self._rank(notes)
        prediction = (self.onsets[0], self.onsets[-1], ranking[0] if ranking else None)
        self.predictions.append(prediction)
        self.rankings.append(ranking)
        return prediction

    def run_offline(self, notes: np.ndarray) -> list:
        """Run on a note array (fields ``pitch`` and ``onset_sec``) as if it were a stream. Returns the predictions."""
        self.reset()
        for pitch, onset in zip(notes["pitch"], notes["onset_sec"]):
            self([pitch], onset)
        if self.pitches:
            self._predict()
        return self.predictions


class OnlineFingerprinting(OnlineFingerprintingVoting):
    """
    Online piece identification with histograms (the main system): as ``OnlineFingerprintingVoting``, but the pieces are ranked with the histograms of
    the time differences between the score and the query (``match_histograms``), and not with the number of tokens.
    The score of a piece is the number of tokens of its highest bin. The hash table is the one of
    ``build_hash_table_v2``.

    The window is sliding: when a note arrives, only the tokens that end with that note are created and looked up in
    the hash table, and the matches are kept until the first note of their token leaves the window. A window then
    only needs to compute the histograms of the stored matches, and ``hop`` can be 1 note. The result is the same as
    running ``match_histograms`` on the notes of the window.
    The notes that arrive at the same time are processed in order of pitch.

    Parameters
    ----------
    hash_table : HashTableV2
        Hash table of the scores (``build_hash_table_v2``).
    token_args :
        Arguments of the tokens (``d``, ``n1``, ``n2`` and ``max_pitch_distance``) and of the matching
        (``bin_size``, ``tolerance``, ``intervals``: it has to be the one of the hash table).

    Attributes
    ----------
    matches : list with the best matches ``(piece, position, number of tokens)`` of each window (see
        ``Histograms.best_matches``). The position is the score time (seconds) of the last note of the window.
    positions : list with the position of the best match of each window (``None`` if there are no matches).
    histograms : ``Histograms`` of the last window.
    """

    def __init__(
        self,
        hash_table: HashTableV2,
        window_size: int = 50,
        hop: int = None,
        queue=None,
        **token_args
    ):
        super().__init__(
            hash_table, window_size, hop, queue, **{"n1": 5, "n2": 5, **token_args}
        )
        assert self.token_args.get("intervals") in (None, hash_table.intervals)

    def reset(self):
        super().reset()
        self.matches = []
        self.positions = []
        self.offset = 0  # index (in the stream) of the first note of the window
        self.chunks = []  # matches of the tokens that end with each note
        self.partners = {}  # note -> the first valid notes after it
        self.parents = (
            {}
        )  # note -> the notes that have it among their first valid notes

    def __call__(self, pitches, time):
        completed = None
        for pitch in np.sort(np.atleast_1d(pitches)):
            self.pitches.append(int(pitch))
            self.onsets.append(float(time))
            self._add_tokens()
            if len(self.pitches) == self.window_size:
                completed = self._predict()
                del self.pitches[: self.hop], self.onsets[: self.hop]
                for expired in range(self.offset, self.offset + self.hop):
                    del self.partners[expired], self.parents[expired]
                self.offset += self.hop
                self.chunks = [
                    chunk for chunk in self.chunks if chunk[0] >= self.offset
                ]
        return completed

    def _add_tokens(self):
        """
        Look up the tokens that end with the last note, and store their matches.

        For each note of the window the first ``n1`` (``n2``) valid notes after it are known when they arrive: when a
        note arrives it is added to the list of the notes before it for which it is valid (its time is more than ``d``
        later, and it is at most ``max_pitch_distance`` away). The tokens that end with the note are then the triplets
        (i, j, note) where the note is one of the first ``n2`` valid notes after j, and j is one of the first ``n1``
        valid notes after i. Only these are created, as ``generate_triplets`` would do for the whole window.
        """
        args = self.token_args
        n1, n2, limit = args["n1"], args["n2"], args.get("max_pitch_distance")
        onsets = np.array(self.onsets)
        pitches = np.array(self.pitches, dtype=np.int64)
        last = len(onsets) - 1
        new = self.offset + last  # index of the note in the stream

        valid = np.round(onsets[last] - onsets[:last], 2) > args.get("d", 0.05)
        if limit is not None:
            valid &= np.abs(pitches[:last] - pitches[last]) <= limit
        ends, self.parents[new], self.partners[new] = [], [], []
        for x in np.flatnonzero(valid) + self.offset:
            following = self.partners[x]
            if len(following) < max(n1, n2):
                if len(following) < n2:
                    ends.append(x)  # the new note is one of the first n2 notes after x
                if len(following) < n1:
                    self.parents[new].append(
                        x
                    )  # and x has the new note as one of its first n1 notes
                following.append(new)
        pairs = [(i, j) for j in ends for i in self.parents[j] if i >= self.offset]
        if not pairs:
            return

        first, second = np.array(pairs).T - self.offset
        td_1 = np.round(onsets[second] - onsets[first], 2)
        td_2 = np.round(onsets[last] - onsets[second], 2)
        table = self.hash_table
        token, entry = table.match(
            pitches[first],
            pitches[second],
            np.full(len(first), pitches[last]),
            td_1 / td_2,
            args.get("tolerance", 0.25),
        )
        if len(token):
            self.chunks.append(
                (
                    new,
                    self.offset + first[token],
                    table.stime[entry],
                    table.std1[entry] / td_1[token],
                    onsets[first[token]],
                    table.piece_id[entry],
                )
            )

    def _predict(self):
        table = self.hash_table
        if self.chunks:
            first, stime, ratio, qtime, piece = (
                np.concatenate([chunk[i] for chunk in self.chunks]) for i in range(1, 6)
            )
            keep = first >= self.offset
            histograms = histograms_from_positions(
                table.pieces,
                piece[keep],
                stime[keep] + (self.onsets[-1] - qtime[keep]) * ratio[keep],
                self.token_args.get("bin_size", 1.0),
            )
        else:
            histograms = Histograms(
                table.pieces, np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
            )
        self.histograms = histograms  # of the last window
        ranking = histograms.ranking()
        matches = histograms.best_matches()
        prediction = (self.onsets[0], self.onsets[-1], ranking[0] if ranking else None)
        self.predictions.append(prediction)
        self.rankings.append(ranking)
        self.matches.append(matches)
        self.positions.append(matches[0][1] if matches else None)
        return prediction
