"""MLX implementation of the Hikari encoder/decoder (Whisper-medium shaped, causal encoder)."""

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


@dataclass
class HikariMLXConfig:
    d_model: int = 1024
    n_heads: int = 16
    enc_layers: int = 24
    dec_layers: int = 24
    ffn_dim: int = 4096
    n_mels: int = 80
    vocab_size: int = 51865
    max_source_positions: int = 1500
    max_target_positions: int = 375
    time_dilation: int = 4

    @classmethod
    def from_hf(cls, cfg: dict) -> "HikariMLXConfig":
        return cls(
            d_model=cfg["d_model"],
            n_heads=cfg["encoder_attention_heads"],
            enc_layers=cfg["encoder_layers"],
            dec_layers=cfg["decoder_layers"],
            ffn_dim=cfg["encoder_ffn_dim"],
            n_mels=cfg["num_mel_bins"],
            vocab_size=cfg["vocab_size"],
            max_source_positions=cfg["max_source_positions"],
            max_target_positions=cfg["max_target_positions"],
            time_dilation=cfg.get("decoder_time_dilation", 1),
        )


@mx.compile
def gelu(x):
    """Exact GELU evaluated in fp32 (fused, so storage stays fp16).

    mlx's fp16 ``nn.gelu`` loses precision for negative inputs (1 + erf cancellation) and visibly perturbs
    the encoder output; the torch reference does not.
    """
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

    def _heads(self, x):
        b, l, d = x.shape
        return x.reshape(b, l, self.n_heads, d // self.n_heads).transpose(0, 2, 1, 3)

    def kv(self, src):
        """Key/value heads [B, H, L, D] of `src` (used to precompute cross-attention K/V)."""
        return self._heads(self.k_proj(src)), self._heads(self.v_proj(src))

    def attend(self, q, k, v, mask=None, zero_rows=None):
        """q: [B,H,Lq,D] heads; k, v: [B,H,Lk,D] heads -> [B, Lq, d] after out_proj."""
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        if zero_rows is not None:
            # query rows that attend to nothing produce 0 (torch SDPA semantics), not NaN
            o = mx.where(zero_rows[None, None, :, None], 0, o)
        b, h, l, dh = o.shape
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(b, l, h * dh))

    def __call__(self, x, kv=None, mask=None, zero_rows=None, kv_heads=None):
        q = self._heads(self.q_proj(x))
        k, v = kv_heads if kv_heads is not None else self.kv(x if kv is None else kv)
        return self.attend(q, k, v, mask, zero_rows)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: HikariMLXConfig):
        super().__init__()
        d = cfg.d_model
        self.self_attn = Attention(d, cfg.n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.ffn_dim)
        self.fc2 = nn.Linear(cfg.ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def __call__(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask="causal")
        return self._ffn(x)  # includes the fp16 overflow guard of the torch reference

    @staticmethod
    def _clip(x):
        if x.dtype == mx.float16:
            lim = 65504.0 - 1000.0
            x = mx.clip(x, -lim, lim)
        return x

    def _ffn(self, x):
        x = x + self.fc2(gelu(self.fc1(self.final_layer_norm(x))))
        return self._clip(x)

    def forward_capture(self, x):
        """Full causal pass that also returns this layer's self-attention K/V heads (for incremental use)."""
        a = self.self_attn
        h = self.self_attn_layer_norm(x)
        q, (k, v) = a._heads(a.q_proj(h)), a.kv(h)
        x = x + a.attend(q, k, v, mask="causal")
        return self._ffn(x), k, v

    def forward_incremental(self, x_new, k_prev, v_prev):
        """Process rows continuing a cached sequence. k_prev/v_prev: committed K/V [1,H,nc,D] (nc may be 0)."""
        a = self.self_attn
        h = self.self_attn_layer_norm(x_new)
        q, (k_new, v_new) = a._heads(a.q_proj(h)), a.kv(h)
        k = mx.concatenate([k_prev, k_new], axis=2)
        v = mx.concatenate([v_prev, v_new], axis=2)
        nc, m = k_prev.shape[2], x_new.shape[1]
        mask = mx.arange(nc + m)[None, :] <= (nc + mx.arange(m))[:, None]  # row i sees keys <= nc + i
        x = x_new + a.attend(q, k, v, mask=mask)
        return self._ffn(x), k, v


class Encoder(nn.Module):
    def __init__(self, cfg: HikariMLXConfig):
        super().__init__()
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.n_mels, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, d)
        self.layers = [EncoderLayer(cfg) for _ in range(cfg.enc_layers)]
        self.layer_norm = nn.LayerNorm(d)

    def __call__(self, mel, n_out=None):
        """mel: [B, T, n_mels] -> [B, T/2, d]; `n_out` keeps only the first frames (exact: the encoder is causal)"""
        x = gelu(self.conv1(mel))
        x = gelu(self.conv2(x))
        if n_out is not None:
            x = x[:, :n_out]
        x = x + self.embed_positions.weight[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)

    def _embed(self, mel, start, stop):
        x = gelu(self.conv1(mel))
        x = gelu(self.conv2(x))[:, start:stop]
        return x + self.embed_positions.weight[start:stop]

    def forward_capture(self, mel, n_out):
        """Like __call__ but also returns per-layer (k, v) heads for the first n_out frames."""
        x = self._embed(mel, 0, n_out)
        kvs = []
        for layer in self.layers:
            x, k, v = layer.forward_capture(x)
            kvs.append((k, v))
        return self.layer_norm(x), kvs

    def forward_incremental(self, mel, n_commit, n_out, kvs):
        """Recompute frames [n_commit, n_out) on top of the first n_commit cached frames of `kvs`."""
        x = self._embed(mel, n_commit, n_out)
        new = []
        for layer, (k, v) in zip(self.layers, kvs):
            x, k, v = layer.forward_incremental(x, k[:, :, :n_commit], v[:, :, :n_commit])
            new.append((k, v))
        return self.layer_norm(x), new


@lru_cache(maxsize=64)
def causal_mask(n: int) -> mx.array:
    return mx.arange(n)[:, None] >= mx.arange(n)[None, :]


@lru_cache(maxsize=64)
def dilated_cross_mask(n_tok: int, n_frames: int, dilation: int):
    """Token i may attend to encoder frames j < dilation * i (row 0 attends to nothing)."""
    i = mx.arange(n_tok)[:, None]
    j = mx.arange(n_frames)[None, :]
    mask = j < dilation * i if dilation != 1 else j <= i
    empty = ~mask.any(axis=-1)  # rows with no visible key
    # keep the row computable (all True), the output of these rows is zeroed afterwards
    mask = mx.where(empty[:, None], True, mask)
    return mask, empty


class DecoderLayer(nn.Module):
    def __init__(self, cfg: HikariMLXConfig):
        super().__init__()
        d = cfg.d_model
        self.self_attn = Attention(d, cfg.n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.encoder_attn = Attention(d, cfg.n_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.ffn_dim)
        self.fc2 = nn.Linear(cfg.ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def __call__(self, x, enc, cross_mask, cross_empty, cross_kv=None):
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask="causal")
        x = x + self.encoder_attn(
            self.encoder_attn_layer_norm(x), kv=enc, mask=cross_mask, zero_rows=cross_empty, kv_heads=cross_kv
        )
        x = x + self.fc2(gelu(self.fc1(self.final_layer_norm(x))))
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: HikariMLXConfig):
        super().__init__()
        self.time_dilation = cfg.time_dilation
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, cfg.d_model)
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.dec_layers)]
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def cross_kv(self, enc):
        """Per-layer cross-attention K/V heads of encoder outputs `enc` [B, F, d]."""
        return [layer.encoder_attn.kv(enc) for layer in self.layers]

    def __call__(self, ids, enc=None, cross_kv=None):
        """ids: [B, n] int; enc: [B, F, d] or precomputed `cross_kv` (list of (k, v)) -> hidden [B, n, d]"""
        n = ids.shape[1]
        n_frames = enc.shape[1] if cross_kv is None else cross_kv[0][0].shape[2]
        x = self.embed_tokens(ids) + self.embed_positions.weight[:n]
        cross_mask, cross_empty = dilated_cross_mask(n, n_frames, self.time_dilation)
        for i, layer in enumerate(self.layers):
            x = layer(x, enc, cross_mask, cross_empty, None if cross_kv is None else cross_kv[i])
        return self.layer_norm(x)

    def head(self, h):
        return self.embed_tokens.as_linear(h)


class Hikari(nn.Module):
    def __init__(self, cfg: HikariMLXConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
