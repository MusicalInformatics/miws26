import io
import os
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np

from typing import Any, Tuple, List

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
    ) -> None:
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

        self.fundamental_freq_raw = freqs[spectrogram_dist.argmax()]
        self.fundamental_freq = MIDI_PITCH_FREQS[
            frequency_to_midi_pitch(self.fundamental_freq_raw)
        ]
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
            self.signal[1000:2000] / abs(self.signal).max(),
            color="firebrick",
            label="Original signal",
        )
        self.signal_ax.plot(
            audio_signal[1000:2000] / abs(audio_signal).max(),
            color="black",
            label="Fourier approximation",
        )
        self.signal_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    def update_plot(self) -> None:
        self.synthesize()
        plt.draw()
