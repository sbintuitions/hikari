"""MLX backend with the same public surface as :class:`hikari.server.model_wrapper.ModelWrapper`."""

import copy
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Union

import numpy as np
import torch
from termcolor import cprint

from hikari.mlx.engine import HikariMLX
from hikari.mlx.stride import StrideSession
from hikari.utils import Timer

logger = logging.getLogger(__name__)

WAIT_TOKEN = 93
GEN_PROMPTS = {
    ("translate", "Japanese"): [50258, 50266, 50358, 50363],
    ("translate", "German"): [50258, 50261, 50358, 50363],
    ("translate", "Russian"): [50258, 50263, 50358, 50363],
    ("translate", "English"): [50258, 50259, 50358, 50363],
    ("translate", "Chinese"): [50258, 50260, 50358, 50363],
}
TRANSCRIBE_PROMPT = [50258, 50259, 50359, 50363]


class ModelWrapperMLX:
    def __init__(self, checkpoint=None, device="mlx", debug=False, verbose=False, refresh_stride=0):
        assert checkpoint is not None, "Please provide a valid checkpoint path."
        self.SPEECH_THRESHOLD = 0.8
        self.debug = debug
        self.verbose = verbose
        self.profile = False
        self.timings = defaultdict(list)
        cprint(f"Loading MLX model from {checkpoint}", "cyan", attrs=["bold"])

        self.WAIT_PENALTY_BOOST = 0.3
        self.WAIT_PENALTY_DECAY = 0.3
        self.BASELINE_WAIT_PENALTY = 0.0
        self.temperature = 0.0
        self.SUPPRESS_TOKENS = []
        self.SUPPRESS_BY_ABS = 5.0
        self.REPETITION_PENALTY = 0.0
        self.j = -1
        self.speech_prob = 0.0

        self.REFRESH_STRIDE = refresh_stride  # 0 = exact sliding window; K > 0 = approximate stride refresh
        self.engine = HikariMLX(checkpoint)
        self.session = StrideSession(self.engine)
        self.DECODER_CONTEXT_LEN = self.engine.cfg.max_target_positions
        self.EFFECTIVE_WINDOW = self.DECODER_CONTEXT_LEN
        self.samples_per_token = self.engine.samples_per_token

        # torch is only used for Silero VAD. Its default intra-op thread pool busy-waits and slows the MLX
        # step by ~40% (measured on M4), so keep it to one thread.
        torch.set_num_threads(1)
        # Silero VAD (CPU torch); run in a worker thread so it overlaps with the GPU step.
        self.vad_model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
        self._vad_pool = ThreadPoolExecutor(max_workers=1)

        from transformers import WhisperTokenizer

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")
        self.apply_settings({"task": "transcribe", "tgt_lang": "English", "baseline_wait_penalty": 0.0})
        self.timer = Timer(logger=logger)

    # ------------------------------------------------------------------ settings
    def apply_settings(self, settings, debug=False):
        self.WAIT_PENALTY_BOOST = settings.get("wait_penalty_boost", self.WAIT_PENALTY_BOOST)
        self.WAIT_PENALTY_DECAY = settings.get("wait_penalty_decay", self.WAIT_PENALTY_DECAY)
        self.REPETITION_PENALTY = settings.get("repetition_penalty", self.REPETITION_PENALTY)
        task = settings.get("task", None)
        tgt_lang = settings.get("tgt_lang", None)
        baseline_wait_penalty = settings.get("baseline_wait_penalty", None)
        self.REFRESH_STRIDE = settings.get("refresh_stride", self.REFRESH_STRIDE)
        ctx = settings.get("decoder_context", self.DECODER_CONTEXT_LEN)
        if not (task and tgt_lang and str(baseline_wait_penalty)):
            raise ValueError("Both 'task', 'tgt_lang' and 'baseline_wait_penalty' must be provided in control settings.")
        if ctx == "max":
            ctx = self.DECODER_CONTEXT_LEN
        self.EFFECTIVE_WINDOW = min(max(ctx, 50), self.DECODER_CONTEXT_LEN)
        self.tgt_lang = tgt_lang
        if task not in ["transcribe", "translate"]:
            raise ValueError(f"Task {task} is not supported. Choose 'transcribe' or 'translate'.")
        self.TASK = task
        self.BASELINE_WAIT_PENALTY = baseline_wait_penalty
        if task == "translate":
            if (task, tgt_lang) not in GEN_PROMPTS:
                raise ValueError(f"Target language {tgt_lang} is not supported.")
            gen_prompt = GEN_PROMPTS[(task, tgt_lang)]
        else:
            gen_prompt = TRANSCRIBE_PROMPT
        self.decoder_input_ids_lst = copy.deepcopy(gen_prompt)
        cprint(
            f"Changed task to {self.TASK}. gen prompt: {self.tokenizer.decode(self.decoder_input_ids_lst)} | "
            f"Effective decoder context: {self.EFFECTIVE_WINDOW} | Baseline wait penalty: {self.BASELINE_WAIT_PENALTY} | "
            f"WP boost: {self.WAIT_PENALTY_BOOST} | WP decay: {self.WAIT_PENALTY_DECAY} | Rpt penalty: {self.REPETITION_PENALTY}"
            + (f" | APPROXIMATE stride refresh K={self.REFRESH_STRIDE}" if self.REFRESH_STRIDE else ""),
            color="cyan",
        )
        self.reset()

    def reset(self):
        self.full_audio = []
        self.wait_penalty = self.BASELINE_WAIT_PENALTY
        self.silence = []
        self.current_word_tokens = []
        self.j = -1
        self.session.reset()
        self.window_start = 0  # sample index where the encoder window starts (stride mode)

    # ------------------------------------------------------------------ VAD
    def is_speech(self, chunk: torch.Tensor, threshold: float = 0.8, sample_rate: int = 16000):
        with torch.no_grad():
            speech_prob = self.vad_model(chunk, sample_rate).item()
        return speech_prob > threshold, speech_prob

    # ------------------------------------------------------------------ one step
    def _sample(self, logits: np.ndarray) -> int:
        if self.temperature > 0:
            x = logits.astype(np.float64).copy()
            if self.SUPPRESS_TOKENS:
                x[self.SUPPRESS_TOKENS] = -np.inf
            x /= self.temperature
            x -= x.max()
            p = np.exp(x)
            p /= p.sum()
            return int(np.random.choice(len(p), p=p))
        if self.SUPPRESS_TOKENS:
            logits[self.SUPPRESS_TOKENS] -= self.SUPPRESS_BY_ABS
        return int(logits.argmax())

    def get_one_token(self, audio_chunk: np.ndarray) -> Tuple[str, Union[None, np.ndarray]]:
        self.full_audio.append(audio_chunk)
        audio_np = np.hstack(self.full_audio)
        spt = self.samples_per_token
        if audio_np.shape[0] < spt * 3:
            return "", None

        self.j += 1
        W = self.EFFECTIVE_WINDOW
        pos = min(W - 1, self.j + 3)
        t = len(audio_np)
        windowed_audio = audio_np[max(0, t - W * spt) :]

        vad_future = self._vad_pool.submit(
            self.is_speech, torch.from_numpy(windowed_audio[-512:]), self.SPEECH_THRESHOLD
        )

        timings = {} if self.profile else None
        if self.REFRESH_STRIDE:
            ids = self.decoder_input_ids_lst
            refresh = False
            if len(ids) > W:  # block-wise window advance: drop K+1 oldest generated tokens at once
                keep = max(5, W - self.REFRESH_STRIDE)
                del ids[4 : 4 + len(ids) - keep]
                refresh = True
            if refresh or self.session.kvs is None:
                self.window_start = max(0, t - (len(ids) - 1) * spt)
            t0 = time.perf_counter()
            logits = self.session.logits_row(audio_np[self.window_start :], ids, refresh)
            if timings is not None:
                self.timings["total"].append(time.perf_counter() - t0)
        else:
            if len(self.decoder_input_ids_lst) > W:
                self.decoder_input_ids_lst.pop(4)
            logits = self.engine.logits_row(windowed_audio, self.decoder_input_ids_lst, W, pos, timings)
        if timings is not None:
            for k, v in timings.items():
                self.timings[k].append(v)

        speech, self.speech_prob = vad_future.result()
        self.silence.append(not speech)

        max_logit = int(logits.argmax())
        if max_logit != WAIT_TOKEN and max_logit in self.decoder_input_ids_lst[-5:]:
            logits[max_logit] -= self.REPETITION_PENALTY
        logits[WAIT_TOKEN] -= self.wait_penalty
        last_token = self._sample(logits)
        self.decoder_input_ids_lst.append(last_token)

        if last_token == WAIT_TOKEN:
            if speech and len(self.decoder_input_ids_lst) > 10 and all(
                ii == WAIT_TOKEN for ii in self.decoder_input_ids_lst[-10:]
            ):
                self.wait_penalty += self.WAIT_PENALTY_BOOST
        else:
            self.wait_penalty -= self.WAIT_PENALTY_DECAY * (self.wait_penalty - self.BASELINE_WAIT_PENALTY)

        if self.verbose:
            print(
                f"WP: {self.wait_penalty:.3f} | tok in ctx: {len(self.decoder_input_ids_lst)} | "
                f"seen: {len(windowed_audio)} | speech_p: {self.speech_prob:.2f} | token: {last_token}"
            )

        dec_pred = None
        if last_token == WAIT_TOKEN:
            return "", dec_pred

        decoded_token = self.tokenizer.decode([last_token])
        if (not decoded_token.startswith("�")) or (decoded_token.startswith(" ")):
            if self.current_word_tokens:
                word = self.tokenizer.decode(self.current_word_tokens)
                if word:
                    self.current_word_tokens = [last_token]
                    return word, dec_pred
        self.current_word_tokens.append(last_token)
        return "", dec_pred
