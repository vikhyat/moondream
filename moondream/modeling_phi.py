# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# Licensed under the BSD 3-Clause License.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Tuple

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_moondream import PhiConfig

FusedDense = None


@dataclass
class InferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: Dict[str, Any] = field(default_factory=dict)
    lengths_per_sample: torch.Tensor = None


class Embedding(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.drop(self.wte(input_ids.view(-1, input_ids.size(-1))))


def _apply_rotary_emb(x, cos, sin):
    seqlen, rotary_dim = x.size(1), cos.size(1) * 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    c, s = cos[:seqlen].unsqueeze(1), sin[:seqlen].unsqueeze(1)
    x_rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return torch.cat([x_rot.to(x.dtype), x_pass], dim=-1)


def _apply_rotary_emb_kv(
    kv: torch.FloatTensor, cos: torch.FloatTensor, sin: torch.FloatTensor
) -> torch.FloatTensor:
    seqlen, rotary_dim = kv.shape[1], cos.shape[-1] * 2
    k_rot = kv[:, :, 0, :, :rotary_dim].chunk(2, dim=-1)
    k_pass = kv[:, :, 0, :, rotary_dim:]
    c, s = cos[:seqlen].unsqueeze(1), sin[:seqlen].unsqueeze(1)
    k_rot = torch.cat(
        [k_rot[0] * c - k_rot[1] * s, k_rot[0] * s + k_rot[1] * c], dim=-1
    )
    return torch.cat(
        [torch.cat([k_rot, k_pass], dim=-1).unsqueeze(2), kv[:, :, 1:2, :, :]], dim=2
    )


def _apply_rotary_emb_qkv(
    qkv: torch.FloatTensor, cos: torch.FloatTensor, sin: torch.FloatTensor
) -> torch.FloatTensor:
    seqlen, rotary_dim = qkv.shape[1], cos.shape[1] * 2

    c = cos[:seqlen].unsqueeze(1)
    s = sin[:seqlen].unsqueeze(1)

    qkv_rot = torch.stack(
        [
            torch.cat(
                [
                    qkv[:, :, i, :, : rotary_dim // 2] * c
                    - qkv[:, :, i, :, rotary_dim // 2 : rotary_dim] * s,
                    qkv[:, :, i, :, : rotary_dim // 2] * s
                    + qkv[:, :, i, :, rotary_dim // 2 : rotary_dim] * c,
                ],
                dim=-1,
            ).to(qkv.dtype)
            for i in range(2)
        ],
        dim=2,
    )

    qkv_pass = qkv[:, :, :2, :, rotary_dim:].unsqueeze(2)
    qkv_v = qkv[:, :, 2:3, :, :]
    return torch.cat([qkv_rot, qkv_pass, qkv_v], dim=2)


class RotaryEmbedding(nn.Module):
    # Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/pdf/2104.09864.pdf)
    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        max_position_embeddings: int = 2048,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        # fp32 is preferred since the output of `torch.arange` can be quite large and bf16 would lose a lot of precision
        self.dim, self.base, self.pos_idx_in_fp32, self.device = (
            dim,
            float(base),
            pos_idx_in_fp32,
            device,
        )
        self.max_position_embeddings = max_position_embeddings
        if scale_base is not None:
            raise NotImplementedError

        # Generate and register the non-trainable buffers
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(device), persistent=False
        )
        self.register_buffer(
            "scale", self._calculate_scale(dim, scale_base, device), persistent=False
        )
        self._update_cos_sin_cache(
            max_position_embeddings, device=device, dtype=torch.float32
        )

    def _calculate_scale(self, dim, scale_base, device):
        return (
            (
                (
                    torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    + 0.4 * dim
                )
                / (1.4 * dim)
            )
            if scale_base is not None
            else None
        )

    def _compute_inv_freq(self, device: Optional[str] = None) -> torch.FloatTensor:
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._seq_len_cached = seqlen
        t = torch.arange(
            seqlen,
            device=device,
            dtype=torch.float32 if self.pos_idx_in_fp32 else self.inv_freq.dtype,
        )
        inv_freq = (
            self._compute_inv_freq(device=device)
            if self.pos_idx_in_fp32 and self.inv_freq.dtype != torch.float32
            else self.inv_freq
        )

        freqs = torch.outer(t, inv_freq)

        def apply_scale(freqs, scale, operator, dtype):
            result = operator(freqs)
            return (result / scale).to(dtype) if scale is not None else result.to(dtype)

        if scale := self.scale:
            power = (
                torch.arange(seqlen, dtype=scale.dtype, device=scale.device)
                - seqlen // 2
            ) / self.scale_base
            scale = scale.to(device=power.device) ** power.unsqueeze(1)

        self._cos_cached = apply_scale(
            freqs, 1 / scale if scale is not None else None, torch.cos, dtype
        )
        self._sin_cached = apply_scale(
            freqs, 1 / scale if scale is not None else None, torch.sin, dtype
        )
        if scale is not None:
            self._cos_k_cached = apply_scale(freqs, scale, torch.cos, dtype)
            self._sin_k_cached = apply_scale(freqs, scale, torch.sin, dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        should_update = (
            self._seq_len_cached < qkv.shape[1] + seqlen_offset
            or self._cos_cached.device != qkv.device
            or self._cos_cached.dtype != qkv.dtype
            or (self.training and self._cos_cached.is_inference())
        )

        if should_update:
            self._update_cos_sin_cache(
                qkv.shape[1] + seqlen_offset, device=qkv.device, dtype=qkv.dtype
            )

        offset_cos = self._cos_cached[seqlen_offset:]
        offset_sin = self._sin_cached[seqlen_offset:]

        if kv is None:
            return _apply_rotary_emb_qkv(qkv, offset_cos, offset_sin)
        else:
            return _apply_rotary_emb(qkv, offset_cos, offset_sin), _apply_rotary_emb_kv(
                kv, offset_cos, offset_sin
            )


class MLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        n_inner: Optional[int] = None,
        act_fn: Optional[str] = None,
    ) -> None:
        super().__init__()
        n_inner = n_inner or getattr(config, "n_inner", None) or 4 * config.n_embd
        act_fn = act_fn or config.activation_function

        self.fc1 = nn.Linear(config.n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, config.n_embd)
        self.act = ACT2FN[act_fn]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


# Flash Attention (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py)
class SelfAttention(nn.Module):
    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    @torch.autocast("cpu", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        qkv: torch.FloatTensor,
        causal: Optional[bool] = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ):
        q, k, v = qkv.chunk(3, dim=-1)
        scale = self.softmax_scale or 1.0 / q.size(-1) ** 0.5

        scores = (
            torch.einsum("bthd,bshd->bhts", q.to(torch.float32), k.to(torch.float32))
            * scale
        )
        if causal or self.causal:
            scores.triu_(1).fill_(-10000.0)
        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask[:, None, None, :], -10000.0)

        attn = self.drop(torch.softmax(scores, dim=-1).to(v.dtype))
        return torch.einsum("bhts,bshd->bthd", attn, v)


# Flash Attention (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py)
class CrossAttention(nn.Module):
    def __init__(self, causal=True, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    @torch.autocast("cpu", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        causal: bool = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]

        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)

        q = q.to(torch.float32)
        k = k.to(torch.float32)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        # Autocast is manually disabled to avoid `torch.einsum` performing the operation using float16, which might lead to overflow
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            rows = rearrange(
                torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1"
            )
            cols = torch.arange(seqlen_k, device=k.device, dtype=torch.long)
            causal_mask = cols > rows + seqlen_k - seqlen_q
            scores = scores.masked_fill(causal_mask, -10000.0)

        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        attention = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention, v)

        return output


def _find_mha_dims(
    config: PretrainedConfig,
    n_head: Optional[int] = None,
    n_head_kv: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Tuple[int, int]:
    if n_head is None and head_dim is None:
        head_dim = config.n_embd // config.n_head
        n_head = config.n_head
    elif n_head is None or head_dim is None:
        raise ValueError("`n_head` and `head_dim` must be both specified or `None`.")
    if n_head_kv is None:
        n_head_kv = getattr(config, "n_head_kv", None) or n_head
    return n_head, n_head_kv, head_dim


def _update_kv_cache(
    kv: torch.FloatTensor, inference_params: InferenceParams, layer_idx: int
) -> torch.FloatTensor:
    num_heads, head_dim = kv.shape[-2:]
    layer_memory = inference_params.key_value_memory_dict.setdefault(
        layer_idx,
        torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        ),
    )

    batch_slice = slice(
        inference_params.batch_size_offset,
        inference_params.batch_size_offset + kv.shape[0],
    )
    seqlen_slice = slice(
        inference_params.seqlen_offset, inference_params.seqlen_offset + kv.shape[1]
    )

    if seqlen_slice.stop >= inference_params.max_seqlen:
        layer_memory = torch.cat((layer_memory, kv), dim=1)
        inference_params.key_value_memory_dict[layer_idx] = layer_memory

    layer_memory[batch_slice, seqlen_slice, ...] = kv
    return layer_memory[batch_slice, : seqlen_slice.stop, ...]


# Multi-head attention layer with rotary embeddings
class MHA(nn.Module):
    def __init__(
        self,
        config,
        dtype=None,
        device=None,
        rotary_dim=None,
        rotary_base=10000.0,
        rotary_scale_base=None,
        n_head=None,
        n_head_kv=None,
        head_dim=None,
        bias=True,
        causal=True,
        softmax_scale=None,
        layer_idx=None,
        return_residual=False,
        checkpointing=False,
    ):
        super().__init__()

        # Set rotary embedding if specified
        self.rotary_dim = rotary_dim or getattr(config, "rotary_dim", 0)
        if self.rotary_dim:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_dim,
                base=rotary_base,
                scale_base=rotary_scale_base,
                device=device,
                max_position_embeddings=config.n_positions,
            )

        # Determine MHA dims from arguments or config
        self.n_head, self.n_head_kv, self.head_dim = _find_mha_dims(
            config, n_head, n_head_kv, head_dim
        )
        op_size = self.head_dim * (self.n_head + 2 * self.n_head_kv)
        hidden_size = config.n_embd

        # Choose Linear class based on config, FusedDense is optional
        LinearClass = (
            FusedDense if config.fused_dense and FusedDense is not None else nn.Linear
        )
        self.Wqkv = LinearClass(
            hidden_size, op_size, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = LinearClass(
            hidden_size, hidden_size, bias=bias, device=device, dtype=dtype
        )

        # Initialize attention mechanisms
        attn_kwargs = {
            "causal": causal,
            "softmax_scale": softmax_scale,
            "attention_dropout": config.attn_pdrop,
        }
        self.inner_attn = SelfAttention(**attn_kwargs)
        self.inner_cross_attn = CrossAttention(**attn_kwargs)

        self.layer_idx = layer_idx
        self.return_residual = return_residual
        self.checkpointing = checkpointing

    def _forward_self_attn(
        self, x: torch.FloatTensor, key_padding_mask: Optional[torch.BoolTensor]
    ) -> torch.FloatTensor:
        qkv = rearrange(
            self.Wqkv(x), "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        if self.rotary_dim > 0:
            qkv = self.rotary_emb(qkv)
        attn_func = (
            torch.utils.checkpoint.checkpoint
            if self.checkpointing
            else lambda f, *args, **kwargs: f(*args, **kwargs)
        )
        return attn_func(self.inner_attn, qkv, key_padding_mask=key_padding_mask)

    def _forward_cross_attn(
        self,
        x: torch.FloatTensor,
        past_key_values: Optional[InferenceParams],
        key_padding_mask: Optional[torch.BoolTensor],
    ) -> torch.FloatTensor:
        qkv = self.Wqkv(x)
        q, kv = (
            qkv[..., : self.n_head * self.head_dim],
            qkv[..., self.n_head * self.head_dim :],
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        seqlen_offset = (
            past_key_values.seqlen_offset if past_key_values is not None else 0
        )
        causal = None if seqlen_offset == 0 else False
        if self.rotary_dim > 0:
            q, kv = self.rotary_emb(q, kv=kv, seqlen_offset=seqlen_offset)

        if past_key_values is not None:
            kv = _update_kv_cache(kv, past_key_values, self.layer_idx)

        attn_func = (
            torch.utils.checkpoint.checkpoint
            if self.checkpointing
            else lambda fn, *args, **kwargs: fn(*args, **kwargs)
        )

        return attn_func(
            self.inner_cross_attn,
            q,
            kv,
            key_padding_mask=key_padding_mask,
            causal=causal,
        )

    def forward(
        self,
        x: torch.FloatTensor,
        past_key_values: Optional[InferenceParams] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        attention_mask = attention_mask.bool() if attention_mask is not None else None
        use_cross_attn = self.n_head != self.n_head_kv or past_key_values is not None
        attn_output_function = (
            self._forward_cross_attn if use_cross_attn else self._forward_self_attn
        )
        attn_output = (
            attn_output_function(x, past_key_values, attention_mask)
            if use_cross_attn
            else attn_output_function(x, attention_mask)
        )
        output = self.out_proj(rearrange(attn_output, "... h d -> ... (h d)"))
        return (output, x) if self.return_residual else output


# Parallel block. This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).
class ParallelBlock(nn.Module):
    def __init__(self, config: PretrainedConfig, block_idx: Optional[int] = None):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_idx = block_idx
        self.mixer = MHA(config, layer_idx=block_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs = self.mixer(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        attn_outputs = self.resid_dropout(attn_outputs)
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        return attn_outputs + feed_forward_hidden_states + residual


class CausalLMHead(nn.Module):
    """Causal Language Modeling head. Simplified version."""

    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states):
        return self.linear(self.ln(hidden_states)).to(torch.float32)


# Improving Language Understanding by Generative Pre-Training
# (https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
class CausalLMLoss(nn.Module):
    def __init__(self, shift_labels: bool = True) -> None:
        super().__init__()
        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.FloatTensor, labels: torch.LongTensor
    ) -> torch.FloatTensor:
        if self.shift_labels:
            logits, labels = logits[..., :-1, :], labels[..., 1:]
        return self.loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


class PhiPreTrainedModel(PreTrainedModel):
    config_class = PhiConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["ParallelBlock"]

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You have to specify either `input_ids` or `inputs_embeds`."
            )

        max_batch_size = (
            inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        )
        seqlen_offset = (
            inputs_embeds.shape[1] + input_ids.shape[1] - 2
            if inputs_embeds is not None
            else input_ids.shape[1] - 1
        )

        args = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None
            else {"input_ids": input_ids}
        )

        if not isinstance(past_key_values, InferenceParams):
            past_key_values = InferenceParams(
                max_seqlen=self.config.n_positions,
                max_batch_size=max_batch_size,
                seqlen_offset=0,
                batch_size_offset=0,
                key_value_memory_dict={},
                lengths_per_sample=None,
            )
        else:
            past_key_values.seqlen_offset = seqlen_offset
            args = {"input_ids": input_ids[:, -1].unsqueeze(-1)}

        return {
            **args,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }


class PhiModel(PhiPreTrainedModel):
    _keys_to_ignore_on_load_missing = [""]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]

    def __init__(self, config: PhiConfig) -> None:
        super().__init__(config)
        self.embd = Embedding(config)
        self.h = nn.ModuleList(
            [ParallelBlock(config, block_idx=i) for i in range(config.n_layer)]
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embd.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embd.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        hidden_states = self.embd(input_ids) if input_ids is not None else inputs_embeds

        for layer in self.h:
            func = layer.__call__ if self.gradient_checkpointing else layer
            args = (hidden_states, past_key_values, attention_mask)
            hidden_states = (
                torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=True)
                if self.gradient_checkpointing
                else func(*args)
            )

        return hidden_states


class PhiForCausalLM(PhiPreTrainedModel):
    _keys_to_ignore_on_load_missing, _keys_to_ignore_on_load_unexpected = (
        [""],
        [r"transformer\.h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"],
    )

    def __init__(self, config: PhiConfig) -> None:
        super().__init__(config)
        self.transformer = PhiModel(config)
        self.lm_head = CausalLMHead(config)
        self.loss = CausalLMLoss()
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head.linear

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head.linear = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = self.transformer(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        lm_logits = self.lm_head(hidden_states)
        loss = self.loss(lm_logits, labels) if labels is not None else None

        return CausalLMOutputWithPast(
            loss=loss, logits=lm_logits, past_key_values=past_key_values
        )
