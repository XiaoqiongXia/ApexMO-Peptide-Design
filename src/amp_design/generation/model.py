"""Length-conditioned bidirectional DDiT for antimicrobial peptides."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from amp_design.generation.tokenizer import INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE


@dataclass(frozen=True)
class AMPDDiTConfig:
    hidden_size: int = 256
    n_blocks: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    time_embedding_dim: int = 128
    length_embedding_dim: int = 128
    condition_dim: int = 256
    max_length: int = 64

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AMPDDiTConfig:
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    def __post_init__(self) -> None:
        if self.hidden_size % self.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if (self.hidden_size // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")


class TimestepEmbedding(nn.Module):
    def __init__(self, output_dim: int, frequency_dim: int = 128) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / half
        )
        arguments = time.float()[:, None] * frequencies[None]
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        if self.frequency_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inverse_frequency = 1.0 / (
            10_000
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        sequence_length = query.shape[-2]
        positions = torch.arange(sequence_length, device=query.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inverse_frequency)
        cosine = angles.cos()[None, None].repeat_interleave(2, dim=-1)
        sine = angles.sin()[None, None].repeat_interleave(2, dim=-1)

        def rotate_half(tensor: Tensor) -> Tensor:
            even = tensor[..., 0::2]
            odd = tensor[..., 1::2]
            return torch.stack((-odd, even), dim=-1).flatten(-2)

        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )


def _modulate(tensor: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return tensor * (1.0 + scale[:, None]) + shift[:, None]


class DDiTBlock(nn.Module):
    def __init__(self, config: AMPDDiTConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.n_heads = config.n_heads
        self.head_dim = hidden // config.n_heads
        self.dropout = config.dropout

        self.norm_attention = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.attention_output = nn.Linear(hidden, hidden, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

        self.norm_mlp = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, config.mlp_ratio * hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_ratio * hidden, hidden),
        )
        self.condition = nn.Linear(config.condition_dim, 6 * hidden)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, hidden: Tensor, condition: Tensor, attention_mask: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.condition(condition).chunk(6, dim=-1)
        )
        normalized = _modulate(self.norm_attention(hidden), shift_attn, scale_attn)
        batch_size, sequence_length, hidden_size = normalized.shape
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query, key, value = (
            tensor.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(
                1, 2
            )
            for tensor in (query, key, value)
        )
        query, key = self.rotary(query, key)
        key_mask = attention_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, hidden_size)
        attended = self.attention_output(attended)
        hidden = hidden + gate_attn[:, None] * F.dropout(
            attended, p=self.dropout, training=self.training
        )

        normalized = _modulate(self.norm_mlp(hidden), shift_mlp, scale_mlp)
        hidden = hidden + gate_mlp[:, None] * self.mlp(normalized)
        return hidden


class AMPDDiT(nn.Module):
    """A padding-aware, length-conditioned, bidirectional discrete denoiser."""

    def __init__(self, config: AMPDDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(INPUT_VOCAB_SIZE, config.hidden_size)
        self.time_embedding = TimestepEmbedding(config.time_embedding_dim)
        self.length_embedding = nn.Embedding(
            config.max_length + 1, config.length_embedding_dim
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(
                config.time_embedding_dim + config.length_embedding_dim,
                config.condition_dim,
            ),
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )
        self.blocks = nn.ModuleList(DDiTBlock(config) for _ in range(config.n_blocks))
        self.final_norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.final_condition = nn.Linear(config.condition_dim, 2 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, OUTPUT_VOCAB_SIZE)
        nn.init.zeros_(self.final_condition.weight)
        nn.init.zeros_(self.final_condition.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_t: Tensor,
        time: Tensor,
        lengths: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape [batch, sequence]")
        if time.shape != (x_t.shape[0],):
            raise ValueError("time must have shape [batch]")
        if lengths.shape != (x_t.shape[0],):
            raise ValueError("lengths must have shape [batch]")
        if attention_mask.shape != x_t.shape or attention_mask.dtype is not torch.bool:
            raise ValueError("attention_mask must be boolean with the same shape as x_t")
        if torch.any(lengths < 1) or torch.any(lengths > self.config.max_length):
            raise ValueError(f"lengths must be within [1, {self.config.max_length}]")

        time_condition = self.time_embedding(time)
        length_condition = self.length_embedding(lengths)
        condition = self.condition_projection(
            torch.cat((time_condition, length_condition), dim=-1)
        )
        hidden = self.token_embedding(x_t)
        for block in self.blocks:
            hidden = block(hidden, condition, attention_mask)
        shift, scale = self.final_condition(condition).chunk(2, dim=-1)
        hidden = _modulate(self.final_norm(hidden), shift, scale)
        return self.output(hidden)
