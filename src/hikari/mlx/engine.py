"""Stateless Hikari inference engine on MLX: (audio window, token history) -> logits row."""

import json
import os
import time
from typing import List, Optional

import mlx.core as mx
import numpy as np

from hikari.mlx.convert import load_weights, resolve_checkpoint
from hikari.mlx.mel import N_FRAMES, log_mel
from hikari.mlx.model import Hikari, HikariMLXConfig

SAMPLES_PER_TOKEN = 160 * 2 * 4  # 10 ms hop * conv stride 2 * dilation 4 = 80 ms (set from config below)


class HikariMLX:
    def __init__(self, checkpoint: str = "sbintuitions/hikari-medium", dtype=mx.float16):
        ckpt_dir = resolve_checkpoint(checkpoint)
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            self.hf_config = json.load(f)
        self.cfg = HikariMLXConfig.from_hf(self.hf_config)
        self.dtype = dtype
        self.model = Hikari(self.cfg)
        self.model.load_weights(list(load_weights(ckpt_dir, dtype).items()))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.samples_per_token = 160 * 2 * self.cfg.time_dilation

    def logits_row(
        self,
        window_audio: np.ndarray,
        ids: List[int],
        window: int,
        pos: int,
        timings: Optional[dict] = None,
        trim: bool = True,
    ) -> np.ndarray:
        """Logits [vocab] (float32) at decoder row `pos` for the current window.

        window_audio: float32 audio of the current window (<= window * samples_per_token samples)
        ids: current decoder token history (len <= window)
        pos: row of the decoder output used for the next-token decision (== len(ids) - 1)

        trim=False reproduces the reference literally (30 s padded encoder, `window` decoder slots).
        trim=True (default) computes only what row `pos` depends on; this is exact because the decoder and
        encoder are causal and token i only sees encoder frames j < dilation * i:
          - decoder rows 0..pos,
          - encoder frames [0, dilation * pos), from mel frames [0, 2 * frames + 2).
        """
        sync = timings is not None
        dil = self.cfg.time_dilation
        t0 = time.perf_counter()
        if trim:
            n_frames = dil * pos
            mel = log_mel(window_audio, pad_to=2 * n_frames + 2, dtype=self.dtype)
        else:
            n_frames = None
            mel = log_mel(window_audio, pad_to=N_FRAMES, dtype=self.dtype)
        if sync:
            mx.eval(mel)
            t1 = time.perf_counter()
        enc = self.model.encoder(mel, n_out=n_frames)
        if sync:
            mx.eval(enc)
            t2 = time.perf_counter()
        if trim:
            ids_mx = mx.array(np.asarray(ids[: pos + 1], dtype=np.int32)[None])
        else:
            padded = np.zeros((1, window), dtype=np.int32)
            padded[0, : len(ids)] = ids
            ids_mx = mx.array(padded)
        h = self.model.decoder(ids_mx, enc)
        row = self.model.decoder.head(h[:, pos]).astype(mx.float32)[0]
        mx.eval(row)
        out = np.array(row)
        if sync:
            t3 = time.perf_counter()
            timings.update(mel=t1 - t0, encoder=t2 - t1, decoder=t3 - t2, total=t3 - t0)
        return out
