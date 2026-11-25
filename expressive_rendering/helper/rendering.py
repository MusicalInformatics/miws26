import json
import os

import torch
import numpy as np
import subprocess
import soundfile
import tempfile
import logging

from IPython.display import display, Audio

from partitura import save_performance_midi, load_musicxml, load_performance_midi
from partitura.score import unfold_part_maximal
from partitura.utils.synth import synthesize, SAMPLE_RATE
from partitura.utils.fluidsynth import HAS_FLUIDSYNTH, synthesize_fluidsynth
from basismixer.predictive_models import FullPredictiveModel, construct_model
from basismixer.performance_codec import get_performance_codec
from basismixer.basisfunctions import make_basis
from basismixer.data import remove_grace_notes
from basismixer.score_notes import expand_grace_notes_by_type, remove_double_notes

from helper.predictions import setup_output_directory

LOGGER = logging.getLogger(__name__)

SAMPLE_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "models"
)


def path_to_trained_models(path=setup_output_directory()):
    """Folder of the models trained in notebook 04 (the sample models if there are none)."""
    has_models = os.path.isdir(path) and any(
        os.path.isfile(os.path.join(path, f, "best_model.pth")) for f in os.listdir(path)
    )
    if not has_models:
        print("Models not found! Using sample models")
        path = SAMPLE_MODELS_DIR
    return path


def render_midi(midi_fn):

    perf = load_performance_midi(midi_fn)
    if HAS_FLUIDSYNTH:
        data = synthesize_fluidsynth(
            note_info=perf,
            samplerate=SAMPLE_RATE,
        )
    else:
        data = synthesize(
            note_info=perf,
            harmonic_dist=5,
            samplerate=SAMPLE_RATE,
        )
    aw = display(Audio(data=data, rate=SAMPLE_RATE, autoplay=True), display_id=True)
    return aw
    # with tempfile.NamedTemporaryFile() as out_file:
    #     cmd = [
    #         "timidity",
    #         "-E",
    #         "F",
    #         "reverb=0",
    #         "F",
    #         "chorus=0",
    #         "--output-mono",
    #         "-Ov",
    #         "-o",
    #         out_file.name,
    #         midi_fn,
    #     ]
    #     try:
    #         ps = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #         if ps.returncode != 0:
    #             LOGGER.error(
    #                 "Command {} failed with code {} (stderr: {})".format(
    #                     cmd, ps.returncode, ps.stderr.decode("UTF8")
    #                 )
    #             )
    #             return False
    #     except FileNotFoundError as f:
    #         LOGGER.error('Executing "{}" returned  {}.'.format(" ".join(cmd), f))
    #         return False
    #     data, fs = soundfile.read(out_file.name)
    #     aw = display(Audio(data=data, rate=fs, autoplay=True), display_id=True)
    #     return aw


def load_model(models_dir):
    models = []
    for f in os.listdir(models_dir):
        path = os.path.join(models_dir, f)
        # every folder with a model (other folders, e.g. generated performances, are skipped)
        if os.path.isfile(os.path.join(path, "config.json")):
            model_config = json.load(open(os.path.join(path, "config.json")))
            params = torch.load(
                os.path.join(path, "best_model.pth"),
                map_location=torch.device("cpu"),
                weights_only=False,
            )["state_dict"]

            model = construct_model(model_config, params)
            models.append(model)

    output_names = list(
        set(
            [name for out_name in [m.output_names for m in models] for name in out_name]
        )
    )
    input_names = list(
        set([name for in_name in [m.input_names for m in models] for name in in_name])
    )
    input_names.sort()
    output_names.sort()

    default_values = dict(
        velocity_trend=64,
        velocity_dev=0,
        beat_period_standardized=0,
        timing=0,
        articulation_log=0,
        beat_period_mean=0.5,
        beat_period_std=0.1,
    )
    all_output_names = list(default_values.keys())
    full_model = FullPredictiveModel(
        models, input_names, all_output_names, default_values
    )

    not_in_model_names = set(all_output_names).difference(output_names)

    print(
        "Trained models include the following parameters:\n"
        + "\n".join(output_names)
        + "\n\n"
        "The following parameters will use default values:\n"
        + "\n"
        + "\n".join(
            ["{0}:{1:.2f}".format(k, default_values[k]) for k in not_in_model_names]
        )
    )

    full_model.eval()  # no dropout when making predictions

    return full_model, output_names


def sanitize_performed_part(ppart):
    """Avoid negative durations in notes."""
    for n in ppart.notes:

        if n["note_off"] < n["note_on"]:
            n["note_off"] = n["note_on"]

        if n["sound_off"] < n["note_off"]:
            n["sound_off"] = n["note_off"]


def post_process_predictions(predictions):
    max_articulation = 1.5
    max_bps = 1
    max_timing = 0.2
    predictions["articulation_log"] = np.clip(
        predictions["articulation_log"], -max_articulation, max_articulation
    )
    predictions["velocity_dev"] = np.clip(predictions["velocity_dev"], 0, 0.8)
    predictions["beat_period_standardized"] = np.clip(
        predictions["beat_period_standardized"], -max_bps, max_bps
    )
    predictions["timing"] = np.clip(predictions["timing"], -max_timing, max_timing)
    predictions["velocity_trend"][predictions["velocity_trend"] > 0.8] = 0.8


def compute_basis_from_xml(
    xml_fn,
    input_names,
    grace_notes="expand",
    double_notes="remove",
    grace_offset_quarter=0.25,
    unfold=True,
):
    """
    Basis functions of a MusicXML score.

    Parameters
    ----------
    xml_fn : str
        MusicXML file.
    input_names : list
        Names of the basis functions (columns of the returned matrix).
    grace_notes : "expand" or "remove"
        "expand" gives the grace notes a position and a duration in the score according to their type
        (see `basismixer.score_notes.expand_grace_notes_by_type`), "remove" removes them.
    double_notes : "remove" or "keep"
        "remove" removes the notes with the same pitch and onset in different voices (they cannot be
        played twice, e.g., on a mechanical piano), see `basismixer.score_notes.remove_double_notes`.
    grace_offset_quarter : float
        Total duration (in quarter notes) of the grace notes of the same type at an onset.
    unfold : bool
        Unfold the repetitions of the score (the notes get new ids). With False the part keeps the note
        ids of the file, e.g., to compare with the performances aligned to that score.

    Returns
    -------
    basis : np.ndarray
        Array (n_notes, len(input_names)).
    part : partitura.score.Part
        The score part (unfolded) that the rows of `basis` refer to.
    """
    if grace_notes not in ("expand", "remove"):
        raise ValueError("`grace_notes` must be 'expand' or 'remove'")
    if double_notes not in ("remove", "keep"):
        raise ValueError("`double_notes` must be 'remove' or 'keep'")

    # Load MusicXML file
    part = load_musicxml(xml_fn, force_note_ids=True if unfold else "keep")[0]
    if double_notes == "remove":
        remove_double_notes(part)
    if grace_notes == "expand":
        expand_grace_notes_by_type(part, grace_offset_quarter=grace_offset_quarter)
    else:
        remove_grace_notes(part)
    if unfold:
        part = unfold_part_maximal(part)

    # Compute basis functions
    _basis, bf_names = make_basis(
        part=part,
        basis_functions=list(set([bf.split(".")[0] for bf in input_names])),
    )
    basis = np.zeros((len(_basis), len(input_names)))
    for i, n in enumerate(input_names):
        try:
            ix = bf_names.index(n)
        except ValueError:
            continue
        basis[:, i] = _basis[:, ix]

    return basis, part
