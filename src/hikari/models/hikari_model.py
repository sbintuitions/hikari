import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from .depth_transformer import MoshiDepthConfig, MoshiDepthDecoder
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import EncoderDecoderCache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from .configuration_hikari import HikariConfig
from .modeling_hikari import (
    HikariDecoder,
    WhisperDecoder,
    WhisperEncoder,
    WhisperPreTrainedModel,
    _compute_mask_indices,
    shift_tokens_right,
)
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin

from transformers import MimiModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

warnings.filterwarnings("once", message="Shutting up the model.")


@dataclass
class HikariModelOutput(Seq2SeqModelOutput):
    """Extended Seq2SeqModelOutput with additional properties."""

    pass


@dataclass
class HikariFinalModelOutput(Seq2SeqLMOutput):
    """Extended Seq2SeqLMOutput with additional properties."""

    depth_decoder_logits: Optional[torch.FloatTensor] = None
    depth_decoder_loss: Optional[torch.FloatTensor] = None
    generated_codes: Optional[torch.LongTensor] = None
    per_codebook_losses: Optional[dict] = None
    text_token_loss: Optional[dict] = None


class HikariModel(WhisperPreTrainedModel):
    def __init__(self, config: HikariConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        if getattr(config, "s2s", False):
            self.decoder = HikariDecoder(config)
        else:
            self.decoder = WhisperDecoder(config)
        self.main_graphed_decoder = None

        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_encoder(self):
        self.encoder._freeze_parameters()

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_position_ids: Optional[torch.LongTensor] = None,
        codebooks: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], HikariModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            if isinstance(self.encoder, MimiModel):
                encoder_last_hidden_state = self.encoder.encode(input_features).last_hidden_state
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_last_hidden_state,
                    hidden_states=None,
                    attentions=None,
                )
            else:
                encoder_outputs = self.encoder(
                    input_features,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs,
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if self.main_graphed_decoder is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                position_ids=decoder_position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                rope_position_ids=rope_position_ids,
                codebooks=codebooks,
            )
        else:
            decoder_outputs = self.main_graphed_decoder(
                decoder_input_ids,
                encoder_outputs[0],
                rope_position_ids,
                codebooks,
            )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return HikariModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class CodesBuffer:
    """only needed for inference"""

    def __init__(self, start_pos=4, max_len=375):
        self.start_pos = start_pos
        self.max_len = max_len
        self.num_codebooks = 8
        self.reset()

    def update(self, new_codes):
        if self.i < self.max_len - self.start_pos:
            self.rolled[0, self.i, :] = new_codes
            self.i += 1
        elif self.i == self.max_len - self.start_pos:
            self.rolled = torch.roll(self.rolled, -1, 1)
            self.rolled[:, -1, :] = new_codes
            self.buf[0, -1, :] = new_codes
        else:
            raise

        self.buf = torch.cat([self.static, self.rolled], dim=1)

    def to(self, device="cpu"):
        self.static = self.static.to(device=device)
        self.rolled = self.rolled.to(device=device)
        self.buf = self.buf.to(device=device)

    def reset(self):
        self.static = torch.zeros(size=(1, self.start_pos, self.num_codebooks)).to(dtype=torch.long)
        self.rolled = torch.zeros(size=(1, self.max_len - self.start_pos, self.num_codebooks)).to(
            dtype=torch.long,
        )
        self.i = 0
        self.buf = torch.cat([self.static, self.rolled], dim=1)


class HikariForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = {"proj_out.weight": "model.decoder.embed_tokens.weight"}

    def __init__(self, config: HikariConfig):
        super().__init__(config)
        self.model = HikariModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions
        self.previous_codebook_0 = deque([0], maxlen=4)
        self.SILENCE_TOKENS = [752, 1926]
        self.graphed_decoder = None
        self.last_sampled_semantic_token = torch.zeros((1, 1), dtype=torch.long, device=self.model.device)
        self.codes_buffer = CodesBuffer()
        self.use_adapter = False
        self._effective_window = None

        if getattr(config, "s2s", False):
            depth_decoder_config = MoshiDepthConfig(
                input_size=1024,
                _attn_implementation="sdpa",
                vocab_size=config.vocab_size,
                cb0_loss_weight=getattr(config, "cb0_loss_weight", 1.0),
            )
            config.use_cache = False
            self.depth_decoder = MoshiDepthDecoder(depth_decoder_config)
            self.depth_decoder_generation_config = GenerationConfig(suppress_tokens=[])

        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        self.model.encoder._freeze_parameters()

    @property
    def effective_window(self):
        return self._effective_window

    @effective_window.setter
    def effective_window(self, val: int):
        for i in range(self.config.decoder_layers):
            self.model.decoder.layers[i].encoder_attn._effective_window = val

        if getattr(self, "s2t_adapter", False):
            for i in range(self.config.num_s2t_layers):
                self.s2t_adapter.layers[i].encoder_attn._effective_window = val

        self._effective_window = val
        print(f"_effective_window: {self._effective_window}")

    @staticmethod
    def get_adapter_input_codes(labels: torch.LongTensor) -> torch.LongTensor:
        input_ids = nn.functional.pad(labels[:, :-1, :], (0, 0, 1, 0, 0, 0), "constant", 0)
        return input_ids

    @staticmethod
    def get_depformer_input_ids(labels: torch.LongTensor) -> torch.LongTensor:
        input_ids = torch.cat(
            [
                nn.functional.pad(labels[:, :-1, :1], (0, 0, 1, 0, 0, 0), "constant", 0),
                labels[..., :-1],
            ],
            dim=-1,
        )
        return input_ids

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        unmasked_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        codes: Optional[torch.LongTensor] = None,
        positions: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels,
                    self.config.pad_token_id,
                    self.config.decoder_start_token_id,
                )

        if codes is not None:
            decoder_input_codes = self.get_adapter_input_codes(codes.clone())
            depformer_input_ids = self.get_depformer_input_ids(codes.clone())
        else:
            decoder_input_codes = self.codes_buffer.buf

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            rope_position_ids=positions,
            codebooks=decoder_input_codes,
        )

        lm_logits = self.proj_out(outputs[0])

        if codes is not None:
            depth_decoder_outputs = self.depth_decoder(
                last_hidden_state=outputs.last_hidden_state,
                input_ids=depformer_input_ids,
                labels=codes,
                text_tokens=unmasked_labels,
            )
            generated_codes = None
            depth_decoder_logits = depth_decoder_outputs.logits
            depth_decoder_loss = depth_decoder_outputs.loss

        else:
            if not getattr(self.config, "s2s", False):
                generated_codes = None
                depth_decoder_logits = None
                depth_decoder_loss = None
            else:
                last_sampled_semantic_token = self.last_sampled_semantic_token

                if self.suppress_repetitive_cb0:
                    _next_cb0_to_suppress = [_t for _t in self.previous_codebook_0 if _t not in self.SILENCE_TOKENS]
                else:
                    _next_cb0_to_suppress = []

                assert self._effective_window is not None, "we must know _effective_window at inference w/o KV"
                last_position_index = positions.argmax().clamp(max=self._effective_window - 1)
                if self.use_adapter:
                    raise NotImplementedError("Separate adapter is not implemented for this model.")
                else:
                    hidden_state = outputs.last_hidden_state[:, last_position_index, :].unsqueeze(1)

                if self.graphed_decoder is None:
                    self.depth_decoder_generation_config.begin_suppress_tokens = _next_cb0_to_suppress
                    generated_codes = self.depth_decoder.generate(
                        generation_config=self.depth_decoder_generation_config,
                        last_hidden_state=hidden_state,
                        input_ids=last_sampled_semantic_token,
                    )
                else:
                    generated_codes = self.graphed_decoder(
                        last_sampled_semantic_token,
                        hidden_state,
                        begin_suppress_tokens=_next_cb0_to_suppress,
                    )
                if getattr(self, "shut_up", False):
                    warnings.warn("Shutting up the model.")
                    generated_codes[0, 1:] = torch.tensor([1926, 243, 1559, 1348, 1736, 1572, 1978, 1744]).to(
                        device=generated_codes.device
                    )
                self.last_sampled_semantic_token = generated_codes[0, 1].view(1, 1)
                self.codes_buffer.update(generated_codes[:, 1:].clone())
                self.previous_codebook_0.append(generated_codes[0, 1].item())
                depth_decoder_logits = None
                depth_decoder_loss = None

        loss = None
        text_token_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            text_token_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            loss = text_token_loss
            if codes is not None:
                loss = loss + depth_decoder_loss

        per_codebook_losses = None
        if codes is not None:
            per_codebook_losses = {}
            loss_fct_monitor = CrossEntropyLoss(reduction="mean", ignore_index=-100)

            num_codebooks = depth_decoder_logits.shape[2]
            vocab_size = depth_decoder_logits.shape[3]

            for i in range(num_codebooks):
                cb_logits = depth_decoder_logits[:, :, i, :].reshape(-1, vocab_size)
                cb_codes = codes[:, :, i].reshape(-1)
                cb_loss = loss_fct_monitor(cb_logits, cb_codes)
                per_codebook_losses[f"loss_codebook_{i}"] = cb_loss.detach()

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return HikariFinalModelOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            depth_decoder_logits=depth_decoder_logits,
            depth_decoder_loss=depth_decoder_loss,
            generated_codes=generated_codes,
            per_codebook_losses=per_codebook_losses,
            text_token_loss=text_token_loss,
        )
