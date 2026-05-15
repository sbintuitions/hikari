import math
from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers import initialization as init
from transformers.models.moshi.configuration_moshi import MoshiConfig, MoshiDepthConfig

from transformers.utils import logging

logger = logging.get_logger(__name__)


class MoshiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MoshiConfig, layer_idx: Optional[int] = None, use_flexible_linear=False, use_rope=True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.scaling = 1 / math.sqrt(self.head_dim)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = MoshiLinear(
            self.hidden_size, self.num_heads * self.head_dim, config.num_codebooks, use_flexible_linear
        )
        self.k_proj = MoshiLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, config.num_codebooks, use_flexible_linear
        )
        self.v_proj = MoshiLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, config.num_codebooks, use_flexible_linear
        )
        self.o_proj = MoshiLinear(
            self.num_heads * self.head_dim, self.hidden_size, config.num_codebooks, use_flexible_linear
        )

        # rotary embeddings are not used in the depth decoder
        self.rotary_emb = None
        if use_rope:
            self.rope_theta = config.rope_theta
            self.rotary_emb = MoshiRotaryEmbedding(config)

    # copied from transformers.models.gemma.modeling_gemma.GemmaAttention.forward
    # no longer copied after attention refactors
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, cache_position)  # Ignore copy
        key_states = self.k_proj(hidden_states, cache_position)  # Ignore copy
        value_states = self.v_proj(hidden_states, cache_position)  # Ignore copy

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:  # Ignore copy
            cos, sin = self.rotary_emb(value_states, position_ids)  # Ignore copy
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # Ignore copy

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = (
                {"sin": sin, "cos": cos, "cache_position": cache_position}
                if self.rotary_emb is not None
                else {"cache_position": cache_position}
            )  # Ignore copy
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output, cache_position)  # Ignore copy

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MoshiSdpaAttention(MoshiAttention):
    """
    Moshi attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MoshiAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MoshiAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MoshiModel is using MoshiSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, T, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, cache_position)  # Ignore copy
        key_states = self.k_proj(hidden_states, cache_position)  # Ignore copy
        value_states = self.v_proj(hidden_states, cache_position)  # Ignore copy

        query_states = query_states.view(bsz, T, q_len, self.num_heads, self.head_dim).transpose(2, 3)
        key_states = key_states.view(bsz, T, q_len, self.num_key_value_heads, self.head_dim).transpose(2, 3)
        value_states = value_states.view(bsz, T, q_len, self.num_key_value_heads, self.head_dim).transpose(2, 3)

        if self.rotary_emb is not None:  # Ignore copy
            cos, sin = self.rotary_emb(value_states, position_ids)  # Ignore copy
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # Ignore copy

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = (
                {"sin": sin, "cos": cos, "cache_position": cache_position}
                if self.rotary_emb is not None
                else {"cache_position": cache_position}
            )  # Ignore copy
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.view(bsz, T, q_len, -1)
        attn_output = self.o_proj(attn_output, cache_position)  # Ignore copy

        return attn_output, None, past_key_value


MOSHI_ATTENTION_CLASSES = {
    "eager": MoshiAttention,
    "sdpa": MoshiSdpaAttention,
}


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """⚠️🤯 WHY 👈👈👈👈
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from
    (batch, T, num_key_value_heads, seqlen, head_dim) to
    (batch, T, num_attention_heads, seqlen, head_dim)
    """
    batch, T, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return einops.repeat(hidden_states, "b t h s d -> b t (h n) s d", n=n_rep)


# Copied from transformers.models.gemma.modeling_gemma.GemmaRMSNorm with Gemma->Moshi
class MoshiRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Ignore copy

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # Ignore copy
    def forward(self, x):
        output = self._norm(x.float())
        output = output * self.weight.float()
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class MoshiFlexibleLinear(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        # Stack the weights for N layers into a single tensor (num_layers, output_size, input_size)
        self.weight = nn.Parameter(torch.randn(num_layers, output_size, input_size))

    def forward(self, x, layer_idx=None):
        """
        `MoshiFlexibleLinear` creates one linear layer per codebook. There's multiple ways to use it.
        In the default case, `sequence_length=num_layers`, so each element of the sequence will be matmul to the weights corresponding to its index on the sequence.

        For more advanced cases, one can specify which codebook's layer(s) to use with `layer_idx`.
        If `layer_idx` indicates a single integer, all of the element of the sequence will be matmul to this single codebook's layer.
        But if `layer_idx` is a tensor of shape `(seq_length,)`, it will matmul each i-th element of the input sequence to the corresponding layer `weight[i]`.


        Args:
            x (`torch.FloatTensor): input to the layer of shape `(batch, num_layers, embed_dim)` or of shape `(batch, seq_length, embed_dim)`
            layer_idx (`torch.Tensor`, *optional*):
                Can be used to specify which codebook's layers(s) to use.
                If it's a tensor of shape `(seq_length,)`, will matmul each element of the sequence to the corresponding weights.
                But if `layer_idx` is a tensor of shape `(seq_length,)`, it will matmul each i-th element of the input sequence to the corresponding layer `weight[i]`.
        """

        # Use torch.gather to select the corresponding weights for each sample
        # (codebooks, output_size, hidden_size)
        selected_weights = torch.index_select(self.weight, 0, layer_idx) if layer_idx is not None else self.weight

        # (1, codebooks, hidden_size, output_size)
        # selected_weights = selected_weights.transpose(1, 2)[None, None, :, :, :]

        # (batch_size, codebooks, 1, hidden_size) x (1, codebooks, hidden_size, output_size)
        # -> (batch_size, codebooks, 1, output_size)
        # x = torch.matmul(x[:, :, None, :], selected_weights)

        selected_weights = selected_weights.transpose(1, 2)
        if len(x.shape) == 3:
            x = torch.einsum("btd,cdo->btco", x, selected_weights)
        else:
            x = torch.einsum("btcd,cdo->btco", x, selected_weights)

        # (batch_size, time, codebooks, output_size)
        return x  # .squeeze(2)


class MoshiLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_codebooks, use_flexible_linear=False):
        super().__init__()

        self.use_flexible_linear = use_flexible_linear

        if not use_flexible_linear:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = MoshiFlexibleLinear(input_dim, output_dim, num_layers=num_codebooks)

    def forward(self, x, layer_idx=None):
        if self.use_flexible_linear:
            return self.linear(x, layer_idx)
        else:
            return self.linear(x)


class MoshiGatingMLP(nn.Module):
    def __init__(self, config, use_flexible_linear=False):
        super().__init__()

        self.activation_fn = ACT2FN[config.hidden_act]
        ffn_dim = config.ffn_dim
        hidden_size = config.hidden_size
        num_layers = config.num_codebooks if use_flexible_linear else 1
        if num_layers == 1:
            self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=False)
            self.fc2 = nn.Linear(ffn_dim // 2, hidden_size, bias=False)
        else:
            self.fc1 = MoshiFlexibleLinear(hidden_size, ffn_dim, num_layers)
            self.fc2 = MoshiFlexibleLinear(ffn_dim // 2, hidden_size, num_layers)

    def forward(self, hidden_states: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) if layer_idx is None else self.fc1(hidden_states, layer_idx)

        batch_size, T, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, T, sequence_length, 2, -1)
        hidden_states = self.activation_fn(hidden_states[..., 0, :]) * hidden_states[..., 1, :]
        hidden_states = self.fc2(hidden_states) if layer_idx is None else self.fc2(hidden_states, layer_idx)
        return hidden_states


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Moshi
class MoshiRotaryEmbedding(nn.Module):
    def __init__(self, config: MoshiConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MoshiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MoshiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoshiDecoderLayer", "MimiTransformerLayer"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = True
    main_input_name = "input_ids"

    def _init_weights(self, module):
        std = 0.005

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, MoshiFlexibleLinear):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, MoshiRMSNorm):
            init.ones_(module.weight)


class MoshiDecoderLayer(nn.Module):
    def __init__(self, config: MoshiConfig, layer_idx: int, use_flexible_linear: bool, use_rope=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_flexible_linear = use_flexible_linear

        self.self_attn = MOSHI_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx, use_flexible_linear=use_flexible_linear, use_rope=use_rope
        )

        self.mlp = MoshiGatingMLP(config, use_flexible_linear)
        self.input_layernorm = MoshiRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MoshiRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

        self._attn_implementation = config._attn_implementation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)  # [B, 1, 4096]

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = (
            self.mlp(hidden_states) if not self.use_flexible_linear else self.mlp(hidden_states, cache_position)
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# MoshiDepthDecoder(config.depth_decoder_config)
class MoshiDepthDecoder(MoshiPreTrainedModel, GenerationMixin):  # , GenerationMixin
    """
    Transformer depth decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MoshiTransformerLayer`]

    Args:
        config: MoshiConfig
    """

    config_class = MoshiDepthConfig

    def __init__(self, config: MoshiDepthConfig):
        super().__init__(config)

        self.embed_text_token = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_semantic_token = nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
        self.embed_acoustic_token = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size + 1, config.hidden_size) for _ in range(config.num_codebooks - 1)]
        )

        self.input_projections = MoshiFlexibleLinear(
            config.input_size,
            config.hidden_size,
            config.num_codebooks,
        )

        self.layers = nn.ModuleList(
            [
                MoshiDecoderLayer(config, layer_idx, use_flexible_linear=True, use_rope=False)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.lm_heads = MoshiFlexibleLinear(config.hidden_size, config.audio_vocab_size, config.num_codebooks)
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,  # input codes (not optional)
        last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # 8 audiocodes as labels [B, T, 8] (passed at training)
        text_tokens: Optional[
            torch.LongTensor
        ] = None,  # text tokens as the 1st token for depformer [B, T] (passed at training)
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens. The first element of the sequence must the text token associated to the audio codebooks.
                The rest of the elements must be flatten audio codebooks. The `cache_position` argument can be used to indicate to which index is associated each token.
            last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the main decoder. Used to contextualize `input_ids`
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
                `past_key_values`).

                If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
                and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
                information on the default strategy.

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
                returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                Two formats are allowed:
                - a [`~cache_utils.Cache`] instance;
                - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
                cache format.

                The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
                legacy cache format will be returned.

                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
                have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
                of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert the inputs into associated vectors than the
                model's internal embedding lookup matrix.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_ids.shape[-1],  # type: ignore
                device=input_ids.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # If inputs_embeds is provided, it has the priority over input_ids, which won't be used

        if inputs_embeds is None:  # see note A in Obsidian.Moshi
            inputs_embeds = []
            for position_idx in cache_position:
                position_idx = position_idx.item()
                if position_idx == 0:  # self.text_embed_tokens: Embedding(32001, 1024) 👈
                    semantic_embedding = self.embed_semantic_token(input_ids[..., [position_idx]])
                    text_embedding = self.embed_text_token(text_tokens.unsqueeze(-1))
                    inputs_embeds.append(semantic_embedding + text_embedding)
                else:
                    inputs_embeds.append(  # self.embed_tokens: ModuleList[7 x Embedding(2049, 1024)]
                        self.embed_acoustic_token[(position_idx - 1)](input_ids[..., [position_idx - past_seen_tokens]])
                    )

            inputs_embeds = torch.cat(inputs_embeds, dim=-2)  # [B, T, 8, 1024]
        # this is where `z_s` is added. The figure in the paper doesn't show that
        # z_s is projected for each position sepately.
        # inputs_embeds += self.input_projections(last_hidden_state, cache_position)  # input_embeds: [B, 8, 1024]
        inputs_embeds = inputs_embeds + self.input_projections(last_hidden_state, cache_position)

        # FIX: Only force mask creation if we have an explicit mask (padding)
        # OR if we are using a fixed-size cache (Static/Sliding) that requires hiding empty slots.
        causal_mask = None
        is_static_cache = isinstance(past_key_values, (StaticCache, SlidingWindowCache))
        if attention_mask is not None or is_static_cache:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
            )
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        hidden_states = inputs_embeds
        for layer_id, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,  # [B, 8, 1024]
                    attention_mask=causal_mask,
                    position_ids=position_ids,  # [0, 1, 2, 3, 4, 5, 6, 7]
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        logits = self.lm_heads(hidden_states, cache_position)  # logits.shape: [B, T, 8, 2048] 1 for each codebook
        logits = logits.squeeze(1)  # logits.shape: [B, T, 8, 2048] or [B, 1, 2048] if T=1 (during generation)
        n_codebooks = logits.shape[2]  # the depformer might have more, but n_codebooks depends on the input data

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            loss_fct = CrossEntropyLoss(reduction="none")  # 🟡 changed here from default reduction

            cloned_labels = labels.clone()  # for safety
            cloned_labels[:, :3, :] = -100  # 🟡🔴🟡 👈 NOTE: HARDCODED: don't backpropagate on the first 3 positions
            cloned_labels = cloned_labels.masked_fill(cloned_labels == self.config.audio_vocab_size, -100)

            if text_tokens is not None:
                # don't backpropagate on the audio codes corresponding to
                # a clipped sentence at the start of the contenx window
                text_token_mask = (text_tokens == -100).unsqueeze(-1)  # (B, 375, 1)
                cloned_labels = cloned_labels.masked_fill(text_token_mask, -100)

            cloned_labels = cloned_labels.reshape(-1)

            # Enable model parallelism
            cloned_labels = cloned_labels.to(logits.device)
            # loss = loss_fct(logits.reshape(-1, self.config.audio_vocab_size), labels)

            # 2. CHANGE: Calculate raw loss (B * T * 8)
            raw_loss = loss_fct(logits.reshape(-1, self.config.audio_vocab_size), cloned_labels)

            # 3. CHANGE: Reshape to [Batch*Time, 8] to access distinct codebooks
            raw_loss = raw_loss.view(-1, n_codebooks)

            # 4. CHANGE: 🟡 HARDCODED weights (e.g., 150.0 for the first codebook, 1.0 for the others)
            weights = torch.tensor([self.config.cb0_loss_weight] + [1.0] * (n_codebooks - 1), device=raw_loss.device)

            # 5. CHANGE: Apply weights and normalize by the number of valid (non-padded) tokens
            # We divide by (labels != -100).sum() so we don't dilute the loss with padding
            num_valid_tokens = (cloned_labels != -100).sum().clamp(min=1.0)
            loss = (raw_loss * weights).sum() / num_valid_tokens

        if not return_dict:
            return tuple(v for v in [loss, logits, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask with Phi3->Moshi
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Moshi. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":  # NOTE: 🚨 what is it?
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->MoshiDepth
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: MoshiDepthConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`MoshiDepthConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            if config.get_text_config().sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - config.get_text_config().sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask
