# Copyright (c) 2026, SB Intuitions Corp
#
# Unless otherwise stated, this project is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator

import fastrtc
import librosa
import numpy as np


@dataclass
class VADEvent:
    interrupt_signal: bool | None = None
    full_audio: tuple[int, np.ndarray] | None = None


class RealtimeVAD:
    def __init__(
        self,
        src_sr: int = 24000,
        hop_size: int = 256,
        start_threshold: float = 0.8,
        end_threshold: float = 0.7,
        pad_start_s: float = 0.6,
        min_positive_s: float = 0.4,
        min_silence_s: float = 1.2,
    ):
        self.src_sr = src_sr
        self.vad_sr = 16000
        self.hop_size = hop_size
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.pad_start_s = pad_start_s
        self.min_positive_s = min_positive_s
        self.min_silence_s = min_silence_s

        self.vad_model = TenVad(hop_size=hop_size)

        self.vad_buffer = np.array([], dtype=np.int16)
        self.src_buffer = np.array([], dtype=np.int16)

        self.vad_buffer_offset = 0
        self.src_buffer_offset = 0

        self.active = False
        self.interrupt_signal = False
        self.sum_positive_s = 0.0
        self.silence_start_s: float | None = None

        self.vad_model.process(np.zeros(hop_size, dtype=np.int16))

    def process(self, audio_data: np.ndarray):
        if audio_data.ndim == 2:
            audio_data = audio_data[0]

        self.src_buffer = np.concatenate((self.src_buffer, audio_data))

        vad_audio_data = librosa.resample(
            audio_data.astype(np.float32) / 32768.0,
            orig_sr=self.src_sr,
            target_sr=self.vad_sr,
        )
        vad_audio_data = (vad_audio_data * 32767.0).round().astype(np.int16)
        self.vad_buffer = np.concatenate((self.vad_buffer, vad_audio_data))
        vad_buffer_size = self.vad_buffer.shape[0]

        def process_chunk(chunk_offset_s: float, vad_chunk: np.ndarray):
            speech_prob, _ = self.vad_model.process(vad_chunk)

            hop_s = self.hop_size / self.vad_sr

            if not self.active:
                if speech_prob >= self.start_threshold:
                    self.active = True
                    self.sum_positive_s = hop_s
                    print(f"[VAD] Active at {chunk_offset_s:.2f}s, {speech_prob=:.3f}")
                else:
                    new_src_offset = int((chunk_offset_s - self.pad_start_s) * self.src_sr)
                    cut_pos = new_src_offset - self.src_buffer_offset
                    if cut_pos > 0:
                        self.src_buffer = self.src_buffer[cut_pos:]
                        self.src_buffer_offset = new_src_offset
                return

            chunk_src_pos = int(chunk_offset_s * self.src_sr)

            if speech_prob >= self.end_threshold:
                self.silence_start_s = None
                self.sum_positive_s += hop_s
                if not self.interrupt_signal and self.sum_positive_s >= self.min_positive_s:
                    self.interrupt_signal = True
                    yield VADEvent(interrupt_signal=True)
                    print(f"[VAD] Interrupt signal at {chunk_offset_s:.2f}s, {speech_prob=:.3f}")
            elif self.silence_start_s is None:
                self.silence_start_s = chunk_offset_s

            if self.silence_start_s is not None and chunk_offset_s - self.silence_start_s >= self.min_silence_s:
                cut_pos = chunk_src_pos - self.src_buffer_offset
                if self.interrupt_signal:
                    webrtc_audio = self.src_buffer[np.newaxis, :cut_pos]
                    yield VADEvent(full_audio=(self.src_sr, webrtc_audio))
                    print(f"[VAD] Full audio at {chunk_offset_s:.2f}s, {webrtc_audio.shape=}")
                self.src_buffer = self.src_buffer[cut_pos:]
                self.src_buffer_offset = chunk_src_pos

                self.active = False
                self.interrupt_signal = False
                self.sum_positive_s = 0.0
                self.silence_start_s = None

        for chunk_pos in range(0, vad_buffer_size - self.hop_size, self.hop_size):
            processed_samples = chunk_pos + self.hop_size
            chunk_offset_s = (self.vad_buffer_offset + chunk_pos) / self.vad_sr
            vad_chunk = self.vad_buffer[chunk_pos : chunk_pos + self.hop_size]
            yield from process_chunk(chunk_offset_s, vad_chunk)

        self.vad_buffer = self.vad_buffer[processed_samples:]
        self.vad_buffer_offset += processed_samples


from typing import TypeAlias

StreamerGenerator: TypeAlias = Generator[fastrtc.tracks.EmitType, None, None]
StreamerFn: TypeAlias = Callable[[tuple[int, np.ndarray], str], StreamerGenerator]



class ConstantStreamHandler(fastrtc.StreamHandler):
    """StreamHandler that forwards all audio frames (no VAD), batching into chunks."""

    def __init__(
        self,
        streamer_fn: StreamerFn,
        input_sample_rate: int = 16000,
        chunk_ms: int = 80,
    ):
        super().__init__(
            "mono",
            input_sample_rate,
            None,
            input_sample_rate,
            chunk_ms,
        )
        self.streamer_fn = streamer_fn
        self.generator: StreamerGenerator | None = None
        self.buffer: list[np.ndarray] = []
        self.chunk_samples = int(input_sample_rate * (chunk_ms / 1000.0))

    def emit(self) -> fastrtc.tracks.EmitType:
        if self.generator is None:
            return None
        try:
            return next(self.generator)
        except StopIteration:
            self.generator = None
            return None

    def receive(self, frame: tuple[int, np.ndarray]):
        sr, audio = frame
        if audio.ndim == 2:
            audio = audio[0]

        self.buffer.append(audio)
        buffered = np.concatenate(self.buffer, axis=-1)

        if buffered.shape[-1] >= self.chunk_samples:
            chunk = buffered[: self.chunk_samples]
            self.buffer = [buffered[self.chunk_samples :]]

            self.wait_for_args_sync()
            self.latest_args[0] = (sr, chunk[np.newaxis, :])
            self.generator = self.streamer_fn(*self.latest_args)

    def copy(self):
        return ConstantStreamHandler(
            self.streamer_fn,
            input_sample_rate=self.input_sample_rate,
            chunk_ms=int(self.chunk_samples / self.input_sample_rate * 1000),
        )
