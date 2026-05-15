# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Provides some extra utilities around torch compile, in particular with a way
to fully deactivate it easily with a context manager.
Provides a simple activation checkpointing that is compatible with FSDP and torch compile.
Finally, provides some utilities for CUDA graphing functions.
"""

import os
import typing as tp
from contextlib import contextmanager
from typing import Optional

import torch
from torch import cuda
from transformers.cache_utils import StaticCache

_compile_disabled: bool = False
_in_cuda_graph = False
_disable_cuda_graph = False


def in_cuda_graph() -> bool:
    """Indicate whether we are in a function that is CUDA Graphed (or will be soon)."""
    return _in_cuda_graph


@contextmanager
def _set_in_cuda_graph():
    global _in_cuda_graph
    assert not _in_cuda_graph
    _in_cuda_graph = True
    try:
        yield
    finally:
        _in_cuda_graph = False


def _is_cuda_graph_enabled() -> bool:
    if _disable_cuda_graph:
        return False
    no_cuda_graph = os.environ.get("NO_CUDA_GRAPH", "")
    if no_cuda_graph.lower() not in {"0", "no", "n", ""}:
        return False
    return True


@contextmanager
def no_cuda_graph():
    """Deactivate CUDA Graphing for all the calls in this context manager."""
    global _disable_cuda_graph
    old_value = _disable_cuda_graph
    _disable_cuda_graph = True
    try:
        yield
    finally:
        _disable_cuda_graph = old_value


class CUDAGraphed:
    """Allow simple CUDA Graphing of a function.

    Args:
        func: callable, taking any number of arguments. Its tensors arguments should
            be top level args, not nested in structures (tuples, dicts, etc). Keyword
            arguments are NOT supported for simplicity.
        warmup_steps: how many call to make normally before CUDA Graphing. In particular, this
            allows torch.compiled functions to get properly compiled.
        disabled: if True, just call the func directly, useful to quickly deactivate on CPU.
    """

    def __init__(self, func: tp.Callable, warmup_steps: int = 1, disable: bool = False):
        self.func = func
        self.warmup_steps = warmup_steps
        self.disable = disable
        self._graph: cuda.CUDAGraph | None = None
        self._output: tuple | None = None
        self._args: tuple | None = None

    def reset(self, warmup_steps: int = 0) -> None:
        """Reset the state, meaning the next call we get CUDA Graphed again. Useful if some
        shapes have changed, or external state (e.g. KVCache) has changed."""
        self.warmup_steps = warmup_steps
        self._graph = None
        self._output = None
        self._args = None

    def __call__(self, *args, **kwargs) -> tp.Any:
        if kwargs:
            raise RuntimeError("Named arguments not supported for now.")
        if self.disable or not _is_cuda_graph_enabled() or in_cuda_graph():
            return self.func(*args, **kwargs)

        def _clone_tensors(args: tuple) -> tuple:
            out: list = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = arg.clone()
                out.append(arg)
            return tuple(out)

        def _match_values_copy_tensors(args: tuple, target_args: tuple) -> None:
            if len(args) != len(target_args):
                raise ValueError(f"Expected {len(target_args)}, but got {args} for CUDA Graphed function.")
            for idx, (source, target) in enumerate(zip(args, target_args)):
                if isinstance(target, torch.Tensor):
                    if not isinstance(source, torch.Tensor):
                        raise ValueError(f"Argument #{idx} was a tensor, and is no longer (now {source}).")
                    if source.shape != target.shape:
                        raise ValueError(
                            f"Argument #{idx} had shape {target.shape}, but got shape {source.shape}"
                            "When using CUDAGraph, every call must be done with exactly the same shapes. "
                            "Feel free to deactivate with the env variable NO_CUDA_GRAPH=1, or the decorator "
                            "`with no_cuda_graph():`"
                        )
                    target.copy_(source)
                else:
                    if isinstance(source, torch.Tensor):
                        raise ValueError(f"Argument #{idx} was not a tensor {target}, but is now one.")
                    if source is not target and source != target:
                        raise ValueError(f"Argument #{idx} changed value from {target} to {source}.")

        with _set_in_cuda_graph():
            # Prevent any one under us to try and CUDA Graph things.
            if self._graph is None:
                if self.warmup_steps <= 0:
                    self._graph = cuda.CUDAGraph()
                    # Making a copy just to ensure those are not used else where.
                    self._args = _clone_tensors(args)
                    with cuda.graph(self._graph):
                        self._output = self.func(*self._args)
                    # At this point nothing really happened, so we have to make it run for real.
                    self._graph.replay()
                    return self._output
                else:
                    self.warmup_steps -= 1
                    return self.func(*args)
            else:
                assert self._args is not None
                _match_values_copy_tensors(args, self._args)
                self._graph.replay()
                return self._output


def cuda_graph(func: tp.Callable, warmup_steps: int = 1):
    """Just calls `CUDAGraphed` on the given function."""
    if not _is_cuda_graph_enabled():
        return func
    return CUDAGraphed(func, warmup_steps)


class MoshiStaticCache(StaticCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.dim() == 5 and key_states.shape[1] == 1:
            k_in = key_states.squeeze(1)
            v_in = value_states.squeeze(1)
        else:
            k_in, v_in = key_states, value_states

        k_out, v_out = super().update(k_in, v_in, layer_idx, cache_kwargs)

        if k_out.dim() == 4:
            k_out = k_out.unsqueeze(1)
            v_out = v_out.unsqueeze(1)

        return k_out, v_out


class GraphedDepthDecoder:
    def __init__(self, model, batch_size, warmup_steps=3, disable=False):
        self.model = model
        self.config = model.config
        self.device = model.device

        self.step_tensors = [
            torch.tensor([i], device=self.device, dtype=torch.long) for i in range(self.config.num_codebooks)
        ]

        self.cache = MoshiStaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=self.config.num_codebooks,
            device=self.device,
            dtype=model.dtype,
        )

        if disable:
            self.graphed_forward = self._generation_loop
        else:
            self.graphed_forward = CUDAGraphed(self._generation_loop, warmup_steps=warmup_steps)

    def _generation_loop(
        self,
        additive_logit_suppression_mask: torch.Tensor,
        text_token: torch.Tensor,
        last_hidden_state: torch.Tensor,
    ):
        self.cache.reset()

        generated_codes = [text_token]
        current_input = text_token.unsqueeze(1)

        for step in range(self.config.num_codebooks):
            cache_pos = self.step_tensors[step]

            if step == 0:
                text_embeds = self.model.embed_text_token(current_input)
                semantic_embeds = self.model.embed_semantic_token(current_input)
                embeds = semantic_embeds + text_embeds
            else:
                embeds = self.model.embed_acoustic_token[step - 1](current_input)

            outputs = self.model(
                input_ids=None,
                inputs_embeds=embeds,
                last_hidden_state=last_hidden_state,
                past_key_values=self.cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=False,
            )

            logits = outputs[0]
            logits += additive_logit_suppression_mask[cache_pos, :]

            next_token = torch.argmax(logits[:, 0, :], dim=-1, keepdim=True)
            generated_codes.append(next_token)
            current_input = next_token

        return torch.cat(generated_codes, dim=1)

    def __call__(
        self,
        text_token: torch.Tensor,
        last_hidden_state: torch.Tensor,
        begin_suppress_tokens: Optional[list] = None,
    ):
        additive_logit_suppression_mask = torch.zeros(
            size=(
                self.model.config.num_codebooks,
                self.model.config.audio_vocab_size,
            )
        ).to(device=self.model.device, dtype=self.model.dtype)
        if begin_suppress_tokens:
            additive_logit_suppression_mask[0, begin_suppress_tokens] = -1000.0

        return self.graphed_forward(additive_logit_suppression_mask, text_token, last_hidden_state)
