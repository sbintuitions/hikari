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

import copy
import logging
import random
from collections import deque
from typing import Tuple, Union

import numpy as np
import torch
from termcolor import cprint
from tqdm import trange

from hikari.models.compilation_utils import CUDAGraphed
from hikari.models.configuration_hikari import HikariConfig
from hikari.models.hikari_model import HikariForConditionalGeneration
from hikari.utils import Timer, get_log_mel, sample_with_suppression
from transformers import  WhisperProcessor

logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(self, checkpoint=None, device="cuda:0", debug=False):
        self.SPEECH_THRESHOLD = 0.8
        self.device = device
        self.debug = debug
        assert checkpoint is not None, "Please provide a valid checkpoint path."
        cprint(f"Speech threshold: {self.SPEECH_THRESHOLD}", "cyan", attrs=["bold"])
        cprint(f"Loading model from {checkpoint}", "cyan", attrs=["bold"])
        cprint(f"Using device: {device}", "cyan", attrs=["bold"])
        whisper_config = HikariConfig.from_pretrained(checkpoint)
        whisper_config.use_cache = False  # TODO: disable for now, fix later

        attn_implementation = "sdpa"
        self.WAIT_PENALTY_TYPE = "additive"
        self.BASELINE_WAIT_PENALTY = 0.0
        self.WAIT_PENALTY_BOOST = 0.3
        self.WAIT_PENALTY_DECAY = 0.3

        self.temperature = 0.0

        self.STOP_ON_TOKENS = []
        self.SUPPRESS_TOKENS = []
        self.SUPPRESS_BY_ABS = 5.0
        self.REPETITION_PENALTY = 0.0
        self.j = -1

        self.vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.speech_prob = 0.0
        self._model = HikariForConditionalGeneration.from_pretrained(
            checkpoint,
            config=whisper_config,
            attn_implementation=attn_implementation,
            ignore_mismatched_sizes=False,
        ).to(device=self.device)
        self._model.to(torch.half)
        self._model.eval()

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.DECODER_CONTEXT_LEN = whisper_config.max_target_positions
        self.EFFECTIVE_WINDOW = whisper_config.max_target_positions
        self._model._effective_window = self.EFFECTIVE_WINDOW
        assert self.EFFECTIVE_WINDOW <= self.DECODER_CONTEXT_LEN
        self.PRED_AUDIO = []
        self.SRC_AUDIO = []

        cprint(self._model.config._attn_implementation, "yellow", "on_black", attrs=["bold"])
        print(f"max_target_positions: {self._model.config.max_target_positions}")
        cprint(f"Number of paramters: {self._model.num_parameters() / 1e6:.2f} M", color="blue")
        cprint(f"Decoder time dilation: {self._model.config.decoder_time_dilation}", color="green")
        cprint(f"{self._model.device} {self._model.dtype}", color="green")
        self.apply_settings(
            {"task": "transcribe", "tgt_lang": "English", "baseline_wait_penalty": self.BASELINE_WAIT_PENALTY},
            debug=self.debug,
        )
        self.timer = Timer(logger=logger)

    def _s2s_prep(self, debug=False):
        self.positions = deque(range(len(self.decoder_input_ids_lst)), maxlen=self._model._effective_window)
        EFFECTIVE_WINDOW = self._model._effective_window
        device = self._model.device
        dtype = self._model.dtype

        if not debug:
            compiled_encoder = CUDAGraphed(self._model.model.encoder.forward, warmup_steps=3, disable=debug)
            for i in trange(10, desc="Capturing encoder graph"):
                compiled_encoder(torch.randn(size=(1, 80, 3000)).to(device=device, dtype=dtype))

            main_graphed_decoder = CUDAGraphed(
                self._model.model.decoder.forward_graphed_noKV, warmup_steps=3, disable=debug
            )
            for i in trange(10, desc="Capturing decoder graph"):
                encoder_hidden_states = torch.randn(size=(1, 1500, 1024), device=device, dtype=dtype)
                decoder_input_ids = torch.randint(
                    high=40000,
                    size=(1, EFFECTIVE_WINDOW),
                    device=device,
                )
                rope_position_ids = torch.arange(EFFECTIVE_WINDOW, device=device).unsqueeze(0)

                _ = main_graphed_decoder(
                    decoder_input_ids,
                    encoder_hidden_states,
                    rope_position_ids,
                    None,
                )
        else:
            main_graphed_decoder = None
            compiled_encoder = self._model.model.encoder.forward


        self.compiled_encoder = compiled_encoder
        self._model.model.main_graphed_decoder = main_graphed_decoder

        cprint(
            f"rotary_emb in decoder sattn: {self._model.model.decoder.layers[0].self_attn.rotary_emb}", color="yellow"
        )

    def apply_settings(self, settings, debug=False):
        self.WAIT_PENALTY_BOOST = settings.get("wait_penalty_boost", self.WAIT_PENALTY_BOOST)
        self.WAIT_PENALTY_DECAY = settings.get("wait_penalty_decay", self.WAIT_PENALTY_DECAY)
        self.REPETITION_PENALTY = settings.get("repetition_penalty", self.REPETITION_PENALTY)
        task = settings.get("task", None)
        tgt_lang = settings.get("tgt_lang", None)
        baseline_wait_penalty = settings.get("baseline_wait_penalty", None)
        effective_decoder_context = settings.get("decoder_context", self.DECODER_CONTEXT_LEN)
        if not (task and tgt_lang and str(baseline_wait_penalty)):
            raise ValueError(
                "Both 'task', 'tgt_lang' and 'baseline_wait_penalty' must be provided in control settings."
            )

        if effective_decoder_context == "max":
            effective_decoder_context = self.DECODER_CONTEXT_LEN
        self.EFFECTIVE_WINDOW = min(max(effective_decoder_context, 50), self.DECODER_CONTEXT_LEN)
        self._model._effective_window = self.EFFECTIVE_WINDOW

        self.tgt_lang = tgt_lang
        if task not in ["transcribe", "translate"]:
            raise ValueError(f"Task {task} is not supported. Choose 'transcribe' or 'translate'.")

        self.TASK = task
        self.BASELINE_WAIT_PENALTY = baseline_wait_penalty

        if self.TASK == "translate":
            if self.tgt_lang == "Japanese":
                gen_prompt = [50258, 50266, 50358, 50363]
            elif self.tgt_lang == "German":
                gen_prompt = [50258, 50261, 50358, 50363]
            elif self.tgt_lang == "Russian":
                gen_prompt = [50258, 50263, 50358, 50363]
            elif self.tgt_lang == "English":
                gen_prompt = [50258, 50259, 50358, 50363]
            else:
                raise ValueError(f"Target language {self.tgt_lang} is not supported.")
        elif self.TASK == "transcribe":
            gen_prompt = [50258, 50259, 50359, 50363]
        self.decoder_input_ids_lst = copy.deepcopy(gen_prompt)
        cprint(
            f"Changed task to {self.TASK}. gen prompt: {self.processor.tokenizer.decode(self.decoder_input_ids_lst)} | Effective decoder context: {self.EFFECTIVE_WINDOW} | Baseline wait penalty: {self.BASELINE_WAIT_PENALTY} | WP boost: {self.WAIT_PENALTY_BOOST} | WP decay: {self.WAIT_PENALTY_DECAY} | Rpt penalty: {self.REPETITION_PENALTY}",
            color="cyan",
        )
        self.reset()
        self._s2s_prep(debug=debug)

    def reset(self):
        self.full_audio = []
        self.wait_penalty = self.BASELINE_WAIT_PENALTY
        self.silence = []
        self.current_word_tokens = []
        self.j = -1
        self.PRED_AUDIO = []
        self.SRC_AUDIO = []

    def is_speech(self, chunk: torch.Tensor, threshold: float = 0.8, sample_rate: int = 16000):
        with torch.no_grad():
            speech_prob = self.vad_model(chunk, sample_rate).item()
        return speech_prob > threshold, speech_prob

    def get_one_token(self, audio_chunk: np.ndarray) -> Tuple[str, Union[None, np.ndarray]]:
        self.full_audio.append(audio_chunk)
        audio_np = np.hstack(self.full_audio)
        if audio_np.shape[0] < 160 * 2 * self._model.config.decoder_time_dilation * 3:
            return "", None

        self.j += 1
        pos = min(self._model._effective_window - 1, self.j + 3)

        t = len(audio_np)
        st_samp = max(0, t - (self.EFFECTIVE_WINDOW * 160 * 2 * self._model.config.decoder_time_dilation))
        en_samp = t
        with torch.no_grad():
            windowed_audio = audio_np[st_samp:en_samp]
            mel = get_log_mel(windowed_audio).to(device=self.device)
            encoder_hidden_states = self.compiled_encoder(mel)

        speech, self.speech_prob = self.is_speech(
            torch.from_numpy(windowed_audio[-512:]),
            self.SPEECH_THRESHOLD,
        )

        self.silence.append(not speech)

        cprint(
            f"WP: {self.wait_penalty:.3f} | tok in ctx: {len(self.decoder_input_ids_lst)} | seen: {len(windowed_audio)}| speech_p: {self.speech_prob:.2f}",
            color="blue",
        )

        if len(self.decoder_input_ids_lst) > self.EFFECTIVE_WINDOW:
            self.decoder_input_ids_lst.pop(4)

        decoder_input_ids = torch.tensor(self.decoder_input_ids_lst, device=self._model.device).unsqueeze(0)
        decoder_input_ids = torch.nn.functional.pad(
            decoder_input_ids, (0, self.EFFECTIVE_WINDOW - decoder_input_ids.shape[1])
        )

        positions_tensor = torch.tensor(self.positions).to(device=self._model.device).unsqueeze(0)
        positions_tensor = torch.nn.functional.pad(
            positions_tensor, (0, self.EFFECTIVE_WINDOW - positions_tensor.shape[1])
        )

        with torch.no_grad():
            outputs = self._model(
                encoder_outputs=encoder_hidden_states,
                decoder_input_ids=decoder_input_ids,
                positions=positions_tensor,
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            self.positions.append(self.positions[-1] + 1)

            logits = outputs.logits.clone()
            dec_pred = None

            max_logit = logits.argmax(dim=-1)[0][pos].item()
            if max_logit != 93 and max_logit in self.decoder_input_ids_lst[-5:]:
                logits[0, pos, max_logit] -= self.REPETITION_PENALTY

            logits[:, pos, 93] -= self.wait_penalty

            if self.temperature > 0:
                last_token = sample_with_suppression(
                    logits,
                    suppress_tokens=self.SUPPRESS_TOKENS,
                    temperature=self.temperature,
                )[..., -1].item()
            else:
                if self.SUPPRESS_TOKENS:
                    logits[:, pos, self.SUPPRESS_TOKENS] -= self.SUPPRESS_BY_ABS
                last_token = logits.argmax(dim=-1)[0][pos].item()

            self.decoder_input_ids_lst.append(last_token)

            if last_token == 93:
                if speech:
                    if len(self.decoder_input_ids_lst) > 10 and all(
                        [ii == 93 for ii in self.decoder_input_ids_lst[-10:]]
                    ):
                        self.wait_penalty += self.WAIT_PENALTY_BOOST
                        self._model.suppress_repetitive_cb0 = False
            else:
                self.wait_penalty -= self.WAIT_PENALTY_DECAY * (self.wait_penalty - self.BASELINE_WAIT_PENALTY)
                self._model.suppress_repetitive_cb0 = True

        decoded_token = self.processor.tokenizer.decode([last_token])
        print(f"Decoded token: {decoded_token}")

        if last_token == 93:
            return "", dec_pred

        if (not decoded_token.startswith("�")) or (decoded_token.startswith(" ")):
            if self.current_word_tokens:
                word = self.processor.tokenizer.decode(self.current_word_tokens)
                print(word)
                if word:
                    self.current_word_tokens = [last_token]
                    return word, dec_pred

        self.current_word_tokens.append(last_token)
        return "", dec_pred
