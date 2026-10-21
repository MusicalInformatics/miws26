import io
import os
import subprocess
from base64 import b64decode
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.fftpack import fft
import numpy as np

from typing import Any, Tuple, List, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from scipy.interpolate import interp1d

from partitura.utils.music import frequency_to_midi_pitch, midi_pitch_to_frequency
from partitura.utils.synth import additive_synthesis, SAMPLE_RATE

# True if pyfluidsynth is installed and can find the FluidSynth library
# (the same check partitura uses internally)
from partitura.utils.fluidsynth import HAS_FLUIDSYNTH

import IPython.display as ipd
import ipywidgets as widgets
from ipywidgets import interactive, fixed

PATH = os.path.dirname(os.path.realpath(__file__))
MIDI_PITCH_FREQS = midi_pitch_to_frequency(np.arange(128))

# Maximum number of components in the interactive tone analyzer
MAX_COMPONENTS = 200

class VerticalLineOnClickPlot(object):
    """
    Interactive plot

    Add line: left click
    Remove last line: right click
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        figsize: Tuple[int, int] = (8, 6),
        xlabel: str = "x",
        ylabel: str = "y",
        fig: Figure = None,
        ax: Axes = None,
    ) -> None:
        self.x = x
        self.y = y

        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig = fig
            self.ax = ax
        self.ax.plot(x, y, color="navy")
        self.lines: List[plt.Line2D] = []
        # x values of the lines in the order in which they were added
        # (to be able to remove the most recently added line)
        self._line_x_values: List[float] = []
        # all x values, sorted
        self.x_values: List[float] = []
        self.interp_fun = interp1d(
            x=x,
            y=y,
            kind="linear",
            fill_value="extrapolate",
        )
        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event",
            self.onclick,
        )

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def onclick(self, event) -> None:
        if event.inaxes == self.ax:
            if event.button == 1:  # Left click
                self.add_line(event.xdata)
            elif event.button == 3:  # Right click
                self.remove_line()
            self.update_plot()

    def add_line(self, x_val: float) -> None:
        line = self.ax.axvline(x_val, color="r", linestyle="--")
        self.lines.append(line)
        self._line_x_values.append(x_val)
        self.x_values.append(x_val)
        self.x_values.sort()

    def remove_line(self) -> None:
        if self.lines:
            self.lines.pop().remove()
            self.x_values.remove(self._line_x_values.pop())

    def clear_lines(self) -> None:
        while self.lines:
            self.remove_line()

    def update_plot(self) -> None:
        plt.draw()

    @property
    def y_values(self) -> np.ndarray:
        y_interpolated = self.interp_fun(self.x_values)
        return y_interpolated

    def show(self) -> None:
        plt.show()


def load_audio(filename: str) -> Tuple[np.ndarray, float]:
    """
    Load a Wav file

    Parameters
    ----------
    filename : str
        Wav file to be loaded.

    Returns
    -------
    signal : np.ndarray
        The audio signal as a 1D numpy array. If the audio file is not mono,
        the signal will be averaged into a single channel
    sample_rate: float
        Sample rate of the audio file
    """
    # Read the audio file
    sample_rate, signal = wavfile.read(filename)

    # Ensure audio is mono
    if len(signal.shape) > 1 and signal.shape[1] > 1:
        # Convert to mono by averaging channels
        signal = signal.mean(axis=1)

    return signal, sample_rate


_COLAB_RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time));
const toDataURL = blob => new Promise(resolve => {
  const reader = new FileReader();
  reader.onloadend = e => resolve(e.srcElement.result);
  reader.readAsDataURL(blob);
});
var record = time => new Promise(async resolve => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach(track => track.stop());
    resolve(await toDataURL(new Blob(chunks)));
  };
  recorder.start();
  await sleep(time);
  recorder.stop();
});
"""


def _record_audio_local(duration: float, sample_rate: int) -> np.ndarray:
    """Record from the default microphone of this computer using PyAudio"""
    try:
        import pyaudio
    except ImportError as error:
        raise ImportError(
            "Recording audio requires the package `pyaudio` (pip install pyaudio)"
        ) from error

    chunk = 1024
    audio = pyaudio.PyAudio()
    try:
        try:
            rate = sample_rate
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )
        except OSError:
            # The microphone does not support the requested sample rate:
            # record with its default sample rate and resample afterwards
            rate = int(audio.get_default_input_device_info()["defaultSampleRate"])
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )
        n_chunks = int(np.ceil(rate * duration / chunk))
        frames = [
            stream.read(chunk, exception_on_overflow=False) for _ in range(n_chunks)
        ]
        stream.stop_stream()
        stream.close()
    finally:
        audio.terminate()

    signal = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float64) / 32768
    signal = signal[: int(rate * duration)]
    if rate != sample_rate:
        divisor = gcd(rate, sample_rate)
        signal = resample_poly(signal, sample_rate // divisor, rate // divisor)
    return signal


def _record_audio_colab(duration: float, sample_rate: int) -> np.ndarray:
    """Record from the microphone of the browser in Google Colab"""
    from google.colab import output

    ipd.display(ipd.Javascript(_COLAB_RECORD_JS))
    # Colab will ask for permission to use the microphone
    data_url = output.eval_js(f"record({int(duration * 1000)})")
    with open("recording.bin", "wb") as f:
        f.write(b64decode(data_url.split(",")[1]))
    # The browser records in a compressed format (webm, ogg, mp4), so we need
    # to convert it to a wav file
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            "recording.bin",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "recording.wav",
        ],
        check=True,
    )
    _, signal = wavfile.read("recording.wav")
    return signal.astype(np.float64) / 32768


def record_audio(
    duration: float = 2.0,
    sample_rate: int = 44100,
) -> Tuple[np.ndarray, int]:
    """
    Record audio from the microphone.

    Locally, this uses the default microphone of the computer (your operating
    system may ask for permission to use it). In Google Colab, this uses the
    microphone of your browser (the browser will ask for permission).

    Parameters
    ----------
    duration : float
        Duration of the recording in seconds.
    sample_rate : int
        Sample rate of the recording (in Hz).

    Returns
    -------
    signal : np.ndarray
        The recording as a 1D array with values between -1 and 1
        (without DC offset).
    sample_rate : int
        Sample rate of the recording.
    """
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except ImportError:
        in_colab = False

    print(f"Recording {duration:g} seconds... make some sound!")
    if in_colab:
        signal = _record_audio_colab(duration, sample_rate)
    else:
        signal = _record_audio_local(duration, sample_rate)
    print("Done!")

    signal = signal - signal.mean()
    if np.abs(signal).max() < 1e-3:
        print(
            "Warning: (almost) no sound was recorded. Check that your microphone "
            "works and that you gave permission to use it."
        )
    return signal, sample_rate


def compute_magnitude_spectrogram(
    signal: np.ndarray,
    sample_rate: float,
    window_size: int = 2**14,
    hop_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectrogram of a signal using short time Fourier transform.

    Parameters
    ----------
    signal: np.ndarray
        Audio signal as a 1D array.
    sample_rate: float
        Sample rate of the audio signal (in Hz)
    window_size: int
        Size of the window (in samples)
    hop_size: int
        Step size (in samples)

    Returns
    -------
    spectrogram: np.ndarray
        A 2D magnitude spectrogram with shape (n_frequency_bins, n_steps)
    freqs: np.ndarray
        Frequency bins of the signal.
    """

    # Number of windows
    num_windows = 1 + (len(signal) - window_size) // hop_size

    # Initialize the spectrogram with zeros
    spectrogram = np.zeros((window_size // 2, num_windows))

    # Window function (Hamming window)
    window = np.hamming(window_size)

    # Compute STFT
    for w in range(num_windows):
        start = w * hop_size
        end = start + window_size
        audio_segment = signal[start:end] * window
        spectrum = fft(audio_segment)[
            : window_size // 2
        ]  # Take only the positive frequencies
        magnitude = np.abs(spectrum)
        spectrogram[:, w] = magnitude

    # Compute frequency bins
    freqs = np.linspace(0, sample_rate / 2, window_size // 2)

    return spectrogram, freqs


def signal_to_wav_bytes(signal: np.ndarray, sample_rate: float) -> bytes:
    """
    Encode a float signal (in [-1, 1]) as 16-bit PCM WAV bytes, e.g., to
    be used as the value of an `ipywidgets.Audio` widget.
    """
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, int(sample_rate), pcm)
    return buffer.getvalue()


class InteractiveToneAnalyzer(VerticalLineOnClickPlot):
    def __init__(
        self,
        signal: np.ndarray,
        sample_rate: float,
        window_size: int = 1024,
        hop_size: int = 256,
        figsize: Tuple[int, int] = (8, 6),
        waveform_start: int = 1000,
        fundamental_freq: Optional[float] = None,
    ) -> None:
        """
        waveform_start: index of the first sample of the (1000 samples long)
        part of the signal that is compared to the synthesized signal.
        fundamental_freq: fundamental frequency (in Hz) of the signal. By
        default, it is the strongest peak of the spectrum, rounded to the
        closest MIDI pitch.
        """
        spectrogram, freqs = compute_magnitude_spectrogram(
            signal=signal,
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
        )

        spectrogram_dist = spectrogram.sum(axis=1)
        spectrogram_dist /= max(spectrogram_dist.sum(), 1e-10)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=figsize,
        )
        super().__init__(
            x=freqs,
            y=spectrogram_dist,
            figsize=figsize,
            xlabel="Frequency (Hz)",
            ylabel="Magnitude",
            fig=fig,
            ax=axes[0],
        )
        self.signal_ax = axes[1]
        self.signal = signal
        self.waveform_slice = slice(waveform_start, waveform_start + 1000)
        self.spectrogram = spectrogram
        # A single audio player that is updated in place (instead of
        # creating a new player every time the plot changes)
        self.audio_widget = widgets.Audio(
            value=b"",
            format="wav",
            autoplay=False,
            loop=False,
        )
        self.synth_signal = None
        self.time_axis = np.linspace(0, 1, len(signal))

        if fundamental_freq is None:
            self.fundamental_freq_raw = freqs[spectrogram_dist.argmax()]
            self.fundamental_freq = MIDI_PITCH_FREQS[
                frequency_to_midi_pitch(self.fundamental_freq_raw)
            ]
        else:
            self.fundamental_freq_raw = float(fundamental_freq)
            self.fundamental_freq = float(fundamental_freq)
        self.x_values.append(self.fundamental_freq)

        line = self.ax.axvline(
            self.fundamental_freq_raw,
            color="firebrick",
            linestyle="--",
            linewidth=3,
        )
        # Option to pick N equally spaced components between the fundamental
        # frequency and the highest frequency in the spectrum (the Nyquist
        # frequency)
        self.max_freq = freqs.max()
        # Logarithmic slider: from 1 (10^0) to 200 components
        self.n_components_widget = widgets.FloatLogSlider(
            value=1,
            base=10,
            min=0,
            max=np.log10(MAX_COMPONENTS),
            step=0.05,
            description="Components",
            readout_format=".0f",
            continuous_update=False,
            tooltip="Number of equally spaced components between the "
            "fundamental frequency and the Nyquist frequency",
        )
        self.n_components_widget.observe(self._on_n_components_change, names="value")

        self.update_plot()

    def _on_n_components_change(self, change: dict) -> None:
        self.set_n_components(int(round(change["new"])))

    def set_n_components(self, n_components: int) -> None:
        """
        Use n_components components, equally spaced between the fundamental
        frequency and the Nyquist frequency (the highest frequency in the
        spectrum). With one component, only the fundamental frequency is used.
        This replaces the lines that have been added so far, and draws one
        line per component (the fundamental has its own line).
        """
        n_components = int(np.clip(n_components, 1, MAX_COMPONENTS))
        self.clear_lines()
        components = np.linspace(self.fundamental_freq, self.max_freq, n_components)
        # The fundamental frequency is always present
        for freq in components[1:]:
            self.add_line(freq)
        self.update_plot()

    def harmonics_dist(self, freq: int) -> Tuple[np.ndarray, np.ndarray]:
        overtones = np.array(self.x_values) / self.fundamental_freq
        weights = self.y_values
        weights /= 2 * weights.sum()

        return overtones * freq, weights

    def __call__(self, arg: Any) -> None:
        pass

    def show(self) -> None:
        plt.show()
        self.synthesize()
        # The widgets are displayed only once; afterwards the audio widget
        # is updated in place
        ipd.display(self.n_components_widget)
        ipd.display(self.audio_widget)

    def synthesize(self) -> None:
        freqs, weights = self.harmonics_dist(self.fundamental_freq)
        audio_signal = additive_synthesis(
            freqs=freqs,
            duration=1,
            samplerate=SAMPLE_RATE,
            weights=weights,
        )
        self.synth_signal = audio_signal
        self.audio_widget.value = signal_to_wav_bytes(audio_signal, SAMPLE_RATE)
        self.signal_ax.cla()
        self.signal_ax.plot(
            self.signal[self.waveform_slice] / abs(self.signal).max(),
            color="firebrick",
            label="Original signal",
        )
        self.signal_ax.plot(
            audio_signal[self.waveform_slice] / abs(audio_signal).max(),
            color="black",
            label="Fourier approximation",
        )
        self.signal_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    def update_plot(self) -> None:
        self.synthesize()
        plt.draw()
