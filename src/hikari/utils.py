import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import whisper


class Timer:
    def __init__(self, message=None, logger=None, fstream=None, stdout=True):
        self.message = message
        self.fstream = fstream
        self.stdout = stdout
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        if self.stdout and self.message is None:
            print(f"elapsed: {self.interval:.2f} s")
        if self.message is not None:
            if self.fstream is not None:
                self.fstream.write(f"{self.message},{datetime.now().timestamp()},{self.interval}\n")
            if self.logger is not None:
                self.logger.info(f"{self.message} took {self.interval:.2f} seconds")
            if self.fstream is None and self.logger is None and self.stdout:
                print(f"{self.message} took {self.interval:.2f} seconds")
        else:
            pass


def sample_with_suppression(
    logits: torch.Tensor,
    suppress_tokens: list = [0, 10, 20],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from logits while suppressing specific token IDs.

    Args:
        logits: tensor of shape [batch_size, sequence_length, vocab_size]
        suppress_tokens: list of token IDs to suppress
        temperature: sampling temperature (higher = more random, lower = more deterministic)

    Returns:
        sampled tokens of shape [batch_size, sequence_length]
    """
    mask = torch.ones_like(logits)
    mask[..., suppress_tokens] = float("-inf")
    masked_logits = logits + mask

    if temperature != 1.0 and temperature > 0:
        masked_logits = masked_logits / temperature

    probs = F.softmax(masked_logits, dim=-1)
    samples = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
    samples = samples.view(logits.size(0), logits.size(1))
    return samples


def get_log_mel(audio_np, device="cuda:0", dtype=torch.half, pad=True):
    with torch.no_grad():
        mel = whisper.log_mel_spectrogram(torch.from_numpy(audio_np))
        if pad:
            mel = mel.unsqueeze(0).to(device=device, dtype=dtype)
            pad_len = 3000 - mel.shape[-1]
            return F.pad(mel, (0, pad_len), mode="constant", value=0)
        else:
            return mel


def tonp(x):
    """Convert torch tensor to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()
