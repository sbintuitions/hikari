"""Opt-in approximate "stride refresh" inference (encoder-side reuse only; the decoder is always recomputed).

The exact model re-encodes the whole sliding window every 80 ms step. Here the window start advances in blocks
of K steps instead of one:

* at a *refresh* step the whole window is encoded (like the exact mode) and the per-layer encoder K/V and the
  decoder's cross-attention K/V are stored;
* in the K-1 steps after it only the newest frames (plus `margin` trailing frames that were computed with
  zero-padded lookahead) are encoded, attending to the cached K/V, and the cross K/V caches are extended.

Approximation: frames kept from earlier steps carry the mel normalisation / edge context of the step that
computed them, and the window start moves in K-step blocks (window length oscillates in [W-K, W]).
"""

from typing import List

import mlx.core as mx
import numpy as np

from hikari.mlx.engine import HikariMLX
from hikari.mlx.mel import log_mel


class StrideSession:
    def __init__(self, engine: HikariMLX, margin: int = 4, stale_tol: float = 0.05):
        """margin: trailing encoder frames re-encoded every step (they were computed with zero-padded lookahead).
        stale_tol: if the (normalised) mel that the cached frames saw differs from the current mel by more than
        this, re-encode the window (happens when a loud onset raises whisper's `max - 8` floor)."""
        self.engine = engine
        self.margin = margin
        self.stale_tol = stale_tol
        self.forced_refreshes = 0
        self.reset()

    def reset(self):
        self.mel_used = None  # mel as seen by the cached encoder frames
        self.kvs = None  # per encoder layer (k, v) heads for all cached frames
        self.cross = None  # per decoder layer cross-attention (k, v) heads for all cached frames
        self.n_frames = 0

    def logits_row(self, window_audio: np.ndarray, ids: List[int], refresh: bool) -> np.ndarray:
        """Logits [vocab] at row len(ids)-1. `window_audio` is the audio from the current window start on."""
        e = self.engine
        m = e.model
        dil = e.cfg.time_dilation
        pos = len(ids) - 1
        n_frames = dil * pos
        mel = log_mel(window_audio, pad_to=2 * n_frames + 2, dtype=e.dtype)
        n_commit = max(0, self.n_frames - self.margin)
        if not refresh and self.kvs is not None and n_frames == self.n_frames + dil:
            t_cmp = max(0, 2 * n_commit - 4)  # mel frames that the committed encoder frames fully depend on
            if t_cmp and float(mx.abs(mel[:, :t_cmp] - self.mel_used[:, :t_cmp]).max()) > self.stale_tol:
                refresh = True
                self.forced_refreshes += 1
        if refresh or self.kvs is None or n_frames != self.n_frames + dil:
            enc, kvs = m.encoder.forward_capture(mel, n_frames)
            cross = m.decoder.cross_kv(enc)
            self.mel_used = mel
        else:
            enc_new, kvs = m.encoder.forward_incremental(mel, n_commit, n_frames, self.kvs)
            cross = [
                (mx.concatenate([k0[:, :, :n_commit], k1], axis=2), mx.concatenate([v0[:, :, :n_commit], v1], axis=2))
                for (k0, v0), (k1, v1) in zip(self.cross, m.decoder.cross_kv(enc_new))
            ]
            t0 = max(0, 2 * n_commit - 2)
            self.mel_used = mx.concatenate([self.mel_used[:, :t0], mel[:, t0:]], axis=1)
        self.kvs, self.cross, self.n_frames = kvs, cross, n_frames
        ids_mx = mx.array(np.asarray(ids, dtype=np.int32)[None])
        h = m.decoder(ids_mx, cross_kv=cross)
        row = m.decoder.head(h[:, pos]).astype(mx.float32)[0]
        mx.eval(row, self.kvs, self.cross)
        return np.array(row)
