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

from typing import Callable, Generator

import fastrtc
import numpy as np


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
