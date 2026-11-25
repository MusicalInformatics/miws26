"""
Grace notes and double notes in a score `Part` (modified in place).

* `expand_grace_notes_by_type`: gives the grace notes a position and a duration in the score that depends
  on the type of ornament, using `partitura.utils.music.expand_grace_notes_from_local_grace_order`:
  an *acciaccatura* (and other unspecified grace notes) starts before the beat and takes time from the
  previous note, an *appoggiatura* starts on the beat and takes time from the main note. The grace notes
  of the same type at an onset share the total duration `grace_offset_quarter`.
* `remove_double_notes`: removes notes with the same pitch and onset in different voices. They cannot be
  played twice, which is a problem, e.g., for a mechanical piano (Disklavier). Uses
  `partitura.utils.music.remove_double_notes_from_score_note_array`.
"""
import contextlib
import io
import warnings

import numpy as np
from partitura.score import Note, Part
from partitura.utils.music import (
    expand_grace_notes_from_local_grace_order,
    remove_double_notes_from_score_note_array,
)


def _last_tied(note):
    """Last note of a chain of tied notes."""
    while note.tie_next is not None:
        note = note.tie_next
    return note


def expand_grace_notes_by_type(part: Part, grace_offset_quarter: float = 0.25) -> Part:
    """
    Expand the grace notes of a part in score time according to their type (in place).

    The grace notes get a duration and a position before (acciaccatura) or after (appoggiatura) the
    beat; the previous note (acciaccatura) or the main note (appoggiatura) is shortened.

    Parameters
    ----------
    part : partitura.score.Part
    grace_offset_quarter : float
        Total duration (in quarter notes) of the grace notes of the same type at an onset.

    Returns
    -------
    part : partitura.score.Part
        The same part.
    """
    note_array = part.note_array(include_grace_notes=True)
    if not note_array["is_grace"].astype(bool).any():
        return part

    # partitura prints a message when it has to shorten the grace notes: turn it into a warning
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        expanded = expand_grace_notes_from_local_grace_order(
            part, grace_offset_quarter=grace_offset_quarter
        )
    for line in out.getvalue().splitlines():
        if line.strip():
            warnings.warn(line.strip().replace("Warning: ", ""), stacklevel=2)

    original = {nid: row for nid, row in zip(note_array["id"], note_array)}
    notes = {n.id: n for n in part.notes_tied}

    # new start and end of the notes that changed (computed before changing the part)
    changes = []
    for row in expanded:
        old = original[row["id"]]
        start_beat, end_beat = row["onset_beat"], row["onset_beat"] + row["duration_beat"]
        old_start_beat, old_end_beat = old["onset_beat"], old["onset_beat"] + old["duration_beat"]
        new_start = new_end = None
        if not np.isclose(start_beat, old_start_beat):
            new_start = int(np.round(part.inv_beat_map(start_beat)))
        if not np.isclose(end_beat, old_end_beat):
            new_end = int(np.round(part.inv_beat_map(end_beat)))
        if new_start is not None or new_end is not None:
            changes.append((notes[row["id"]], new_start, new_end))

    for note, new_start, new_end in changes:
        if new_start is not None:
            part.remove(note, "start")
            part.add(note, start=new_start)
        if new_end is not None:
            last = _last_tied(note)
            part.remove(last, "end")
            part.add(last, end=new_end)
    return part


def remove_double_notes(part: Part, keep_longer: bool = True) -> list:
    """
    Remove the double notes of a part (in place).

    A double note is a non-grace note with the same onset and pitch as another one (usually written in
    two voices). Only one note of each group is kept. Slurs and tuplets that start or end on a removed
    note are moved to the note that is kept.

    Parameters
    ----------
    part : partitura.score.Part
    keep_longer : bool
        Keep the longest note of the group (the shortest otherwise), so that the sustain of the longer
        note is not lost.

    Returns
    -------
    removed_ids : list
        Ids of the removed notes.
    """
    note_array = part.note_array(include_grace_notes=True)
    kept = remove_double_notes_from_score_note_array(
        note_array, choose_longer_note=keep_longer
    )
    kept_ids = set(kept["id"])
    removed_ids = [i for i in note_array["id"] if i not in kept_ids]
    if not removed_ids:
        return []

    notes = {n.id: n for n in part.notes_tied}
    kept_by_key = {}
    for row in kept:
        kept_by_key[(row["onset_div"], row["pitch"])] = notes[row["id"]]
    replacement = {}
    for row in note_array:
        if row["id"] in removed_ids:
            replacement[notes[row["id"]]] = kept_by_key[(row["onset_div"], row["pitch"])]

    # objects that refer to the notes (e.g., slurs)
    for obj in part.iter_all(include_subclasses=True):
        for attribute in ("start_note", "end_note"):
            if getattr(obj, attribute, None) in replacement:
                setattr(obj, attribute, replacement[getattr(obj, attribute)])

    for note in replacement:
        while note is not None:  # the note and the notes tied to it
            next_note = note.tie_next
            part.remove(note)
            note = next_note
    return removed_ids
