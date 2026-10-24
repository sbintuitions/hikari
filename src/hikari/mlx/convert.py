"""Load the HF Hikari checkpoint into MLX arrays."""

import os
from typing import Dict

import mlx.core as mx

DEFAULT_REPO = "sbintuitions/hikari-medium"


def resolve_checkpoint(checkpoint: str) -> str:
    """Return the directory holding config.json / model.safetensors (local path or HF Hub id)."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint, allow_patterns=["*.json", "model.safetensors"])


def load_weights(checkpoint_dir: str, dtype=mx.float16) -> Dict[str, mx.array]:
    """Read model.safetensors and return weights keyed for `hikari.mlx.model.Hikari`.

    - drops the leading ``model.`` prefix
    - casts to ``dtype`` (the CUDA server also runs in fp16)
    - Conv1d weights (O, I, K) -> (O, K, I) for MLX
    - the LM head is tied to ``decoder.embed_tokens`` (no tensor in the checkpoint)
    """
    raw = mx.load(os.path.join(checkpoint_dir, "model.safetensors"))
    out = {}
    for k, v in raw.items():
        if not k.startswith("model."):
            raise KeyError(f"Unexpected tensor in checkpoint: {k}")
        k = k[len("model.") :]
        if k in ("encoder.conv1.weight", "encoder.conv2.weight"):
            v = v.transpose(0, 2, 1)
        out[k] = v.astype(dtype)
    return out
