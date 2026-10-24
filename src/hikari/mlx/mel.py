"""Log-mel front end matching ``whisper.log_mel_spectrogram`` (80 bins, 16 kHz)."""

import os
from functools import lru_cache

import mlx.core as mx
import numpy as np

N_FFT = 400
HOP = 160
N_MELS = 80
N_FRAMES = 3000  # 30 s


@lru_cache(maxsize=1)
def _filters() -> mx.array:
    import whisper

    path = os.path.join(os.path.dirname(whisper.__file__), "assets", "mel_filters.npz")
    with np.load(path) as f:
        return mx.array(f[f"mel_{N_MELS}"].astype(np.float32))  # (80, 201)


@lru_cache(maxsize=1)
def _window() -> mx.array:
    n = np.arange(N_FFT)
    return mx.array((0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32))


def log_mel(audio: np.ndarray, pad_to: int = N_FRAMES, dtype=mx.float16) -> mx.array:
    """audio: float32 [N] in [-1, 1]. Returns [1, pad_to, 80] (zero padded after normalisation, or cut to
    `pad_to` frames; the normalisation always uses the whole clip)."""
    x = np.pad(audio.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = (len(x) - N_FFT) // HOP + 1
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(N_FFT)[None, :]
    frames = mx.array(x)[mx.array(idx)] * _window()  # (n_frames, 400)
    spec = mx.fft.rfft(frames, axis=-1)
    power = spec.real * spec.real + spec.imag * spec.imag  # (n_frames, 201)
    power = power[:-1]  # whisper drops the last STFT frame
    mel = power @ _filters().T  # (T, 80)
    log_spec = mx.log10(mx.maximum(mel, 1e-10))
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    t = log_spec.shape[0]
    if pad_to is not None:
        # zero-pad after normalisation (as the reference does), or cut frames that are not needed
        log_spec = mx.pad(log_spec, [(0, pad_to - t), (0, 0)]) if t < pad_to else log_spec[:pad_to]
    return log_spec.astype(dtype)[None]
