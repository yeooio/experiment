"""MS-AgentNet: Multi-Scale Agent Network for battery SOH estimation.

Architecture:
    Embedding
    -> [SLFA(DSConvS-K5 + ReLU^2 Agent Attention) -> DSConvL-K31 -> FFN] x depth
    -> LayerNorm -> Flatten -> Linear

This is the only canonical model entry in the final package.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn


FINAL_MODEL_NAME = "MS-AgentNet"
FINAL_ARCHITECTURE = "SLFA(DSConvS-K5+ReLU2-AgentAttention)+DSConvL-K31+FFN"

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "cs2": {
        "dataset": "CS2",
        "features": ("CCCT[3.8,4.0]", "IC_peak"),
        "input_dim": 2,
        "output_dim": 1,
        "learning_rate": 0.001,
        "depth": 4,
        "representation_dim": 16,
        "dense_hidden": 32,
        "seq_len": 5,
        "dropout": 0.01,
        "epochs": 500,
        "n_agents": 2,
        "n_heads": 4,
        "ffn_expand": 1,
        "batch_size": 128,
        "weight_decay": 0.0003,
        "seed": 42,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    },
    "cx2": {
        "dataset": "CX2",
        "features": ("Q_dch_3p8_3p4",),
        "input_dim": 1,
        "output_dim": 1,
        "learning_rate": 0.001,
        "depth": 1,
        "representation_dim": 16,
        "dense_hidden": 16,
        "seq_len": 5,
        "dropout": 0.01,
        "epochs": 500,
        "n_agents": 2,
        "n_heads": 4,
        "ffn_expand": 1,
        "batch_size": 128,
        "weight_decay": 0.0003,
        "seed": 42,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    },
    "mit": {
        "dataset": "MIT/Severson",
        "features": ("Q_dch_window", "energy_efficiency"),
        "input_dim": 2,
        "output_dim": 1,
        "learning_rate": 0.001,
        "depth": 1,
        "representation_dim": 32,
        "dense_hidden": 16,
        "seq_len": 5,
        "dropout": 0.01,
        "epochs": 500,
        "n_agents": 2,
        "n_heads": 4,
        "ffn_expand": 1,
        "batch_size": 128,
        "weight_decay": 0.0003,
        "seed": 42,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    },
    "oxford": {
        "dataset": "Oxford",
        "features": ("CCCT_V_main",),
        "input_dim": 1,
        "output_dim": 1,
        "learning_rate": 0.001,
        "depth": 1,
        "representation_dim": 128,
        "dense_hidden": 64,
        "seq_len": 5,
        "dropout": 0.01,
        "epochs": 500,
        "n_agents": 2,
        "n_heads": 4,
        "ffn_expand": 1,
        "batch_size": 128,
        "weight_decay": 0.0003,
        "seed": 42,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    },
}


def get_dataset_config(dataset: str) -> dict[str, Any]:
    key = dataset.lower()
    if key not in DATASET_CONFIGS:
        choices = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"unknown dataset {dataset!r}; expected one of: {choices}")
    return deepcopy(DATASET_CONFIGS[key])


class DSConvS(nn.Module):
    """K5 short-range depthwise separable convolution without output_linear."""

    def __init__(self, d_model: int, expand_ratio: int = 2):
        super().__init__()
        hidden = d_model * expand_ratio
        self.pw_up = nn.Conv1d(d_model, hidden, kernel_size=1)
        self.dw5 = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=5,
            padding=2,
            groups=hidden,
            bias=False,
        )
        self.act = nn.ReLU()
        self.pw_down = nn.Conv1d(hidden, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.pw_up(x.transpose(1, 2))
        h = self.act(self.dw5(h))
        h = self.pw_down(h).transpose(1, 2)
        return residual + h


class ReLUSquaredAgentAttention(nn.Module):
    """ReLU^2 Agent Attention with channel-wise Q/K/V/O gates and two learned agents."""

    def __init__(self, d_model: int, n_agents: int = 2, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_agents != 2:
            raise ValueError("the final model fixes n_agents=2")
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_scale = nn.Parameter(torch.ones(d_model))
        self.k_scale = nn.Parameter(torch.ones(d_model))
        self.v_scale = nn.Parameter(torch.ones(d_model))
        self.o_scale = nn.Parameter(torch.ones(d_model))
        self.agent = nn.Parameter(torch.randn(n_agents, d_model) * 0.02)

    @staticmethod
    def _relu2_norm(scores: torch.Tensor) -> torch.Tensor:
        weights = torch.relu(scores.float()).square()
        denominator = weights.sum(dim=-1, keepdim=True)
        safe_denominator = torch.where(
            denominator > 0,
            denominator,
            torch.ones_like(denominator),
        )
        normalized = weights / safe_denominator
        uniform = torch.full_like(weights, 1.0 / weights.shape[-1])
        return torch.where(denominator > 0, normalized, uniform).to(scores.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = (x * self.q_scale).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = (x * self.k_scale).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = (x * self.v_scale).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        agent = self.agent.unsqueeze(0).expand(batch_size, -1, -1)
        agent = agent.view(
            batch_size, self.n_agents, self.n_heads, self.head_dim
        ).transpose(1, 2)
        agent_weights = self._relu2_norm(
            torch.matmul(agent, k.transpose(-2, -1)) * self.scale
        )
        agent_values = torch.matmul(agent_weights, v)
        query_weights = self._relu2_norm(
            torch.matmul(q, agent.transpose(-2, -1)) * self.scale
        )
        out = torch.matmul(query_weights, agent_values)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return x + out * self.o_scale


class SLFA(nn.Module):
    def __init__(self, d_model: int, n_agents: int, n_heads: int, dropout: float):
        super().__init__()
        self.dsconv_s = DSConvS(d_model)
        self.agent_attn = ReLUSquaredAgentAttention(d_model, n_agents=n_agents, n_heads=n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.Wa = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dsconv_s(x)
        global_feature = self.dropout(self.agent_attn(local))
        return self.norm(local) + self.Wa * global_feature


class DSConvL(nn.Module):
    """K31 long-range grouped convolution."""

    def __init__(self, d_model: int, expand_ratio: int = 3):
        super().__init__()
        hidden = d_model * expand_ratio
        self.pw_up = nn.Conv1d(d_model, hidden, kernel_size=1)
        self.grouped31 = nn.Conv1d(
            hidden,
            d_model,
            kernel_size=31,
            padding=15,
            groups=d_model,
            bias=False,
        )
        self.act = nn.ReLU()
        self.pw_down = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw_up(x.transpose(1, 2))
        h = self.act(self.grouped31(h))
        return self.pw_down(h).transpose(1, 2)


class DenseHiddenFFN(nn.Module):
    """Affine-free FFN whose hidden width is controlled by dense_hidden."""

    def __init__(self, d_model: int, dense_hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.layers = nn.Sequential(
            nn.Linear(d_model, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.layers(self.norm(x)))


class MSAgentNetBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_agents: int,
        n_heads: int,
        dense_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.slf = SLFA(d_model, n_agents, n_heads, dropout)
        self.dsconv_l_norm = nn.LayerNorm(d_model)
        self.dsconv_l = DSConvL(d_model)
        self.Wl = nn.Parameter(torch.tensor(0.1))
        self.ffn = DenseHiddenFFN(d_model, dense_hidden=dense_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.slf(x)
        h = h + self.Wl * self.dsconv_l(self.dsconv_l_norm(h))
        return self.ffn(h)


class MSAgentNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        n_agents: int,
        n_heads: int,
        dense_hidden: int,
        dropout: float,
        num_blocks: int,
        window_size: int,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                MSAgentNetBlock(
                    d_model=d_model,
                    n_agents=n_agents,
                    n_heads=n_heads,
                    dense_hidden=dense_hidden,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.pre_readout_norm = nn.LayerNorm(
            d_model,
            elementwise_affine=False,
        )
        self.readout = nn.Linear(d_model * window_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.readout(self.pre_readout_norm(h).flatten(1))


def build_model(dataset: str) -> MSAgentNet:
    config = get_dataset_config(dataset)
    return MSAgentNet(
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        d_model=int(config["representation_dim"]),
        n_agents=int(config["n_agents"]),
        n_heads=int(config["n_heads"]),
        dense_hidden=int(config["dense_hidden"]),
        dropout=float(config["dropout"]),
        num_blocks=int(config["depth"]),
        window_size=int(config["seq_len"]),
    )
