#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Codec of the Basis Mixer.

The computation of the expressive parameters (matching of score and performance, tempo, timing and articulation) is
done by the performance codec of partitura (`partitura.musicanalysis.performance_codec`). This module only keeps the
modular interface of the Basis Mixer, to represent a performance in different ways:

* `TimeCodec`: tempo (with the different normalizations of `TEMPO_NORMALIZATION`), timing and articulation. The
  tempo curve can be computed by "average" or "derivative", or by any function with the signature of
  `partitura.musicanalysis.performance_codec.tempo_by_average`.
* A dynamics codec: `NotewiseDynamicsCodec` (`velocity`) or `OnsetwiseDecompositionDynamicsCodec`
  (`velocity_trend` and `velocity_dev`; partitura only has the notewise velocity).
* `PerformanceCodec`: a time codec and a dynamics codec together (`encode` and `decode`).
"""
import logging

import numpy as np
import numpy.lib.recfunctions as rfn

from partitura.performance import PerformedPart
from partitura.musicanalysis.performance_codec import (  # noqa: F401 (re-exported)
    TEMPO_NORMALIZATION,
    beat_period_log_rescale,
    beat_period_log_scale,
    beat_period_ratio_log_rescale,
    beat_period_ratio_log_scale,
    beat_period_ratio_rescale,
    beat_period_ratio_scale,
    beat_period_standardized_rescale,
    beat_period_standardized_scale,
    bp_rescale,
    bp_scale,
    decode_articulation,
    decode_time,
    encode_articulation,
    encode_tempo,
    get_unique_seq,
    tempo_by_average,
    tempo_by_derivative,
    to_matched_score,
)

from basismixer.utils import clip, get_unique_onset_idxs

LOGGER = logging.getLogger(__name__)


class PerformanceCodec(object):
    """
    A Codec for decomposing a performance into expressive parameters
    """

    def __init__(self, time_codec, dynamics_codec, default_values={}):
        super().__init__()
        self.time_codec = time_codec

        self.dynamics_codec = dynamics_codec

        self.parameter_names = (
            self.dynamics_codec.parameter_names + self.time_codec.parameter_names
        )
        self.default_values = default_values

    def encode(self, part, ppart, alignment, return_u_onset_idx=False):
        """
        Encode expressive parameters from a matched performance

        Parameters
        ----------
        matched_score : structured array
            Performance matched to its score.
        snote_ids : list
            NoteID's of the matched notes in the performance.
        return_u_onset_idx : bool
            Return the indices of the unique score onsets
        """

        # The performed durations of the Basis Mixer are the ones of the keys (note off), without the
        # sustain pedal. `to_matched_score` uses `sound_off`, which equals `note_off` for the pedal
        # threshold 127 (the threshold of the performance is restored afterwards)
        pedal_threshold = ppart.sustain_pedal_threshold
        ppart.sustain_pedal_threshold = 127
        try:
            m_score, snote_ids = to_matched_score(part, ppart, alignment)
        finally:
            ppart.sustain_pedal_threshold = pedal_threshold

        # Get time-related parameters
        (time_params, unique_onset_idxs) = self.time_codec.encode(
            score_onsets=m_score["onset"],
            performed_onsets=m_score["p_onset"],
            score_durations=m_score["duration"],
            performed_durations=m_score["p_duration"],
            return_u_onset_idx=True,
        )

        # Get dynamics-related parameters
        dynamics_params = self.dynamics_codec.encode(
            pitches=m_score["pitch"],
            velocities=m_score["velocity"],
            score_onsets=m_score["onset"],
            score_durations=m_score["duration"],
            unique_onset_idxs=unique_onset_idxs,
        )

        # Concatenate parameters into a single matrix
        parameters = rfn.merge_arrays(
            [time_params, dynamics_params], flatten=True, usemask=False
        )

        if return_u_onset_idx:
            return parameters, snote_ids, unique_onset_idxs
        else:
            return parameters, snote_ids

    def decode(
        self,
        part,
        parameters,
        snote_ids=None,
        sanitize=True,
        part_id=None,
        part_name=None,
        return_alignment=False,
        *args,
        **kwargs
    ):

        snotes = part.notes_tied

        if snote_ids is None:
            snote_ids = [n.id for n in snotes]

        # sort
        snote_dict = dict((n.id, n) for n in snotes)
        snote_info = np.array(
            [
                (
                    snote_dict[i].midi_pitch,
                    snote_dict[i].start.t,
                    snote_dict[i].start.t + snote_dict[i].duration_tied,
                )
                for i in snote_ids
            ],
            dtype=[("pitch", "i4"), ("onset", "f4"), ("offset", "f4")],
        )
        sort_idx = np.lexsort((snote_info["pitch"], snote_info["onset"]))
        # the performance is computed for the notes in sorted order: the ids have to be sorted the same way
        sorted_snote_ids = [snote_ids[i] for i in sort_idx]

        bm = part.beat_map
        onsets = bm(snote_info["onset"])[sort_idx]
        durations = (bm(snote_info["offset"]) - bm(snote_info["onset"]))[sort_idx]
        pitches = snote_info["pitch"][sort_idx]

        clip(pitches, 1, 127)

        # Pre-processing
        if "beat_period_mean" in parameters.dtype.names:
            parameters["beat_period_mean"] = (
                np.ones(len(parameters)) * parameters["beat_period_mean"].mean()
            )
        if "beat_period_std" in parameters.dtype.names:
            parameters["beat_period_std"] = (
                np.ones(len(parameters)) * parameters["beat_period_std"].mean()
            )

        dynamics_params = parameters[list(self.dynamics_codec.parameter_names)][
            sort_idx
        ]
        time_params = parameters[list(self.time_codec.parameter_names)][sort_idx]

        onsets_durations = self.time_codec.decode(
            score_onsets=onsets,
            score_durations=durations,
            parameters=time_params,
            *args,
            **kwargs
        )

        velocities = self.dynamics_codec.decode(
            pitches=pitches,
            score_onsets=onsets,
            score_durations=durations,
            parameters=dynamics_params,
        )

        clip(velocities, 1, 127)

        notes = []
        for nid, (onset, duration), velocity, pitch in zip(
            sorted_snote_ids, onsets_durations, velocities, pitches
        ):
            notes.append(
                dict(
                    id=nid,
                    midi_pitch=int(pitch),
                    note_on=onset,
                    note_off=onset + duration,
                    sound_off=onset + duration,
                    velocity=int(velocity),
                )
            )
        # * rescale according to default values?
        ppart = PerformedPart(id=part_id, part_name=part_name, notes=notes)
        ppart.sustain_pedal_threshold = 127

        if return_alignment:
            alignment = []
            for pnote in ppart.notes:
                # the id of a performed note is the id of the score note it was computed from
                alignment.append(
                    dict(label="match", score_id=pnote["id"], performance_id=pnote["id"])
                )

            return ppart, alignment
        else:

            return ppart


#### Codecs for Dynamics ####


class NotewiseDynamicsCodec(object):
    """
    Performance Codec for encoding and decoding dynamics related
    expressive dymensions
    """

    parameter_names = ("velocity",)

    def encode(self, pitches, velocities, *args, **kwargs):
        return np.array(velocities / 127.0, dtype=[("velocity", "f4")])
        # return (velocities / 127.0).reshape(-1, 1)

    def decode(self, pitches, parameters, *args, **kwargs):

        velocity = parameters["velocity"]

        return np.round(velocity * 127.0)


class OnsetwiseDecompositionDynamicsCodec(object):
    """
    Performance Codec for encoding and decoding dynamics related
    expressive dimensions
    """

    parameter_names = ("velocity_trend", "velocity_dev")

    def encode(
        self,
        pitches,
        velocities,
        score_onsets,
        score_durations,
        unique_onset_idxs=None,
        return_u_onset_idx=False,
        *args,
        **kwargs
    ):

        if unique_onset_idxs is None:
            unique_onset_idxs = get_unique_onset_idxs(score_onsets)

        velocity_trend = np.array(
            [np.max(velocities[uix]) for uix in unique_onset_idxs]
        )

        parameters = np.zeros(
            len(pitches), dtype=[(pn, "f4") for pn in self.parameter_names]
        )
        for i, jj in enumerate(unique_onset_idxs):
            parameters["velocity_trend"][jj] = velocity_trend[i] / 127.0
            parameters["velocity_dev"][jj] = (
                velocity_trend[i] - velocities[jj]
            ) / 127.0

        if return_u_onset_idx:
            return parameters, unique_onset_idxs
        else:
            return parameters

    def decode(self, parameters, *args, **kwargs):

        score_onsets = kwargs.get("score_onsets", None)
        if score_onsets is None:
            velocity = parameters["velocity_trend"] - parameters["velocity_dev"]
        else:
            unique_onset_idxs = get_unique_onset_idxs(score_onsets)
            velocity = parameters["velocity_trend"]
            vel_dev = parameters["velocity_dev"]
            for uix in unique_onset_idxs:
                vd = vel_dev[uix]
                vd[vd.argmin()] = 0
                velocity[uix] -= vd
        return np.round(velocity * 127.0)


#### Codec for time-related parameters ####


class TimeCodec(object):
    """
    Performance Codec for encoding and decoding time related expressive
    dimensions (tempo, timing and articulation), with the functions of
    `partitura.musicanalysis.performance_codec`.

    Parameters
    ----------
    tempo_fun : str or callable
        How the tempo curve is computed: "average", "derivative" or a function with the signature of
        `partitura.musicanalysis.performance_codec.tempo_by_average`.
    normalization : str
        A key of `TEMPO_NORMALIZATION`: "beat_period", "beat_period_log", "beat_period_ratio",
        "beat_period_ratio_log" or "beat_period_standardized".
    """

    def __init__(self, tempo_fun="average", normalization="beat_period"):
        try:
            self.normalization = TEMPO_NORMALIZATION[normalization]
        except KeyError:
            raise KeyError(
                "Unknown normalization specification: " "{}".format(normalization)
            )
        self.normalization_name = normalization

        tempo_param_name = self.normalization["param_names"][0]

        self.parameter_names = (
            tempo_param_name,
            "timing",
            "articulation_log",
        ) + self.normalization["param_names"][1:]

        self.tempo_fun = tempo_fun

    def encode(
        self,
        score_onsets,
        performed_onsets,
        score_durations,
        performed_durations,
        return_u_onset_idx=False,
    ):
        """
        Compute time-related performance parameters from a performance
        """
        parameters, unique_onset_idxs = encode_tempo(
            score_onsets=score_onsets,
            performed_onsets=performed_onsets,
            score_durations=score_durations,
            performed_durations=performed_durations,
            return_u_onset_idx=True,
            beat_normalization=self.normalization_name,
            tempo_smooth=self.tempo_fun,
        )
        # partitura also returns the beat period and a (zero) velocity: keep the parameters of this codec
        parameters = rfn.repack_fields(parameters[list(self.parameter_names)])

        if return_u_onset_idx:
            return parameters, unique_onset_idxs
        else:
            return parameters

    def decode(self, score_onsets, score_durations, parameters, *args, **kwargs):
        """
        Decode a performance into onsets and durations in seconds
        for each note in the score.
        """
        return decode_time(
            score_onsets=score_onsets,
            score_durations=score_durations,
            parameters=parameters,
            normalization=self.normalization_name,
            *args,
            **kwargs
        )


#
def get_dynamics_codec(names):
    dynamics_codec = NotewiseDynamicsCodec

    if "velocity_trend" in names or "velocity_dev" in names:

        if "velocity" in names:

            raise ValueError("Invalid combination of dynamics parameters")

        dynamics_codec = OnsetwiseDecompositionDynamicsCodec

    return dynamics_codec()


def get_timing_codec(names):
    normalization = "beat_period"

    options = [
        ["beat_period"],
        ["beat_period_log"],
        ["beat_period_ratio", "beat_period_mean"],
        ["beat_period_ratio_log", "beat_period_mean"],
        ["beat_period_standardized", "beat_period_mean", "beat_period_std"],
    ]

    normalization = options[0][0]
    conflict = False

    for i, option in enumerate(options):

        option_set = set(option)

        if option_set.intersection(names):

            others = set([o for oo in options[:i] + options[i:] for o in oo])
            bad = others.difference(option_set)

            if bad.intersection(names):
                conflict = True
            else:
                conflict = False
                normalization = option[0]
                break

    if conflict:
        raise ValueError("Invalid combination of tempo parameters")
    else:
        return TimeCodec(normalization=normalization)


def get_performance_codec(names):
    """Given a list of parameter names, return a performance codec that includes all
    parameter names in the list. If no valid performance codec can be
    constructed because of mutually exclusive parameter names or because of
    unknown parameter names a ValueError is raised.

    Parameters
    ----------
    names: list
        List of parameter names

    Returns
    -------
    PerformanceCodec
    """

    dynamics_codec = get_dynamics_codec(names)
    time_codec = get_timing_codec(names)
    pc = PerformanceCodec(time_codec, dynamics_codec)
    unknown_names = set(names).difference(pc.parameter_names)
    if unknown_names:
        raise ValueError("Invalid parameter names {}".format(unknown_names))
    else:
        return pc
