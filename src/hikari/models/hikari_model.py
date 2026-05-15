import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import EncoderDecoderCache
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



class HikariForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = {"proj_out.weight": "model.decoder.embed_tokens.weight"}

    def __init__(self, config: HikariConfig):
        super().__init__(config)
        self.model = HikariModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions
        self.graphed_decoder = None
        self._effective_window = None

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

        self._effective_window = val
        print(f"_effective_window: {self._effective_window}")


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
            codebooks=None,
        )

        lm_logits = self.proj_out(outputs[0])

        generated_codes = None
        depth_decoder_logits = None
        depth_decoder_loss = None
        

        loss = None
        text_token_loss = None
        per_codebook_losses = None

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
