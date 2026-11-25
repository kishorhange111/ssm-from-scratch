"""Mamba-2 block and a Transformer baseline sharing the same LM wrapper.

Mamba-2 block (Dao & Gu 2024, simplified to one B/C group):

    [z, x, B, C, dt] = in_proj(u)
    x, B, C = SiLU(causal_depthwise_conv(x, B, C))
    dt      = softplus(dt + dt_bias)               # per head
    y       = SSD(x · dt, a = dt · A, B, C) + D · x   with A = −exp(A_log) < 0
    out     = out_proj(RMSNorm(y · SiLU(z)))

Two execution modes with identical maths:
- ``forward``: whole sequence with the chunked SSD algorithm (training, prefill);
- ``step``: one token with an O(1) recurrent state (conv buffer plus (H, P, N) SSM state).
  This is why generation cost doesn't grow with context, unlike a KV cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .scan import ssd_chunked


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Mamba2(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, head_dim: int = 32,
                 d_conv: int = 4, chunk: int = 64) -> None:  # fmt: skip
        super().__init__()
        self.d_inner = expand * d_model
        self.h = self.d_inner // head_dim
        self.p, self.n, self.k, self.chunk = head_dim, d_state, d_conv, chunk
        self.conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.h, bias=False)
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv, groups=self.conv_dim,
                              padding=d_conv - 1)  # fmt: skip
        # dt initialised log-uniform in [1e-3, 1e-1], as in Mamba
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.h).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.h))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _split(self, zxbcdt):
        return torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.h], dim=-1)

    def _ssm_inputs(self, xbc, dt):
        x, B, C = torch.split(xbc, [self.d_inner, self.n, self.n], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # (..., h)
        A = -torch.exp(self.A_log)
        x = x.view(*x.shape[:-1], self.h, self.p)
        return x, B, C, dt, dt * A

    def _out(self, y, x, z):
        y = y + self.D[:, None] * x
        y = y.reshape(*y.shape[:-2], self.d_inner)
        return self.out_proj(self.norm(y * F.silu(z)))

    def forward(self, u):
        b, l, _ = u.shape
        z, xbc, dt = self._split(self.in_proj(u))
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C, dt, a = self._ssm_inputs(xbc, dt)
        Bh = B[:, :, None, :].expand(b, l, self.h, self.n)  # one group shared by all heads
        Ch = C[:, :, None, :].expand(b, l, self.h, self.n)
        y, _ = ssd_chunked(x * dt[..., None], a, Bh, Ch, self.chunk)
        return self._out(y, x, z)

    def init_state(self, b: int, device=None, dtype=None):
        return (torch.zeros(b, self.conv_dim, self.k - 1, device=device, dtype=dtype),
                torch.zeros(b, self.h, self.p, self.n, device=device, dtype=dtype))  # fmt: skip

    def step(self, u, state):
        """u: (b, d_model) for one position -> (b, d_model), new state."""
        conv_buf, h = state
        z, xbc, dt = self._split(self.in_proj(u))
        window = torch.cat([conv_buf, xbc[..., None]], -1)  # (b, conv_dim, k)
        xbc = F.silu((window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias)
        x, B, C, dt, a = self._ssm_inputs(xbc, dt)
        h = torch.exp(a)[..., None, None] * h + (x * dt[..., None])[..., None] * B[:, None, None, :]
        y = (h * C[:, None, None, :]).sum(-1)
        return self._out(y, x, z), (window[..., 1:], h)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, l, d))


@dataclass
class Config:
    vocab: int = 256
    d_model: int = 256
    n_layers: int = 4
    mixer: str = "mamba2"  # or "attention"
    d_state: int = 64
    n_heads: int = 4
    max_len: int = 1024
    mlp: bool = True  # Mamba-2 LMs often drop the MLP; kept for a like-for-like comparison


class LM(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model) if c.mixer == "attention" else None
        self.mix = nn.ModuleList(
            Mamba2(c.d_model, c.d_state) if c.mixer == "mamba2" else Attention(c.d_model, c.n_heads)
            for _ in range(c.n_layers)
        )
        self.n1 = nn.ModuleList(RMSNorm(c.d_model) for _ in range(c.n_layers))
        self.n2 = nn.ModuleList(RMSNorm(c.d_model) for _ in range(c.n_layers))
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(),
                          nn.Linear(4 * c.d_model, c.d_model)) if c.mlp else nn.Identity()
            for _ in range(c.n_layers)
        )  # fmt: skip
        self.norm = RMSNorm(c.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for mix, n1, n2, mlp in zip(self.mix, self.n1, self.n2, self.mlp, strict=True):
            x = x + mix(n1(x))
            if self.c.mlp:
                x = x + mlp(n2(x))
        return self.norm(x) @ self.emb.weight.t()

    @torch.no_grad()
    def step(self, tok, states):
        """Recurrent decoding for the Mamba-2 LM: one token in, logits out."""
        x = self.emb(tok)
        new = []
        for mix, n1, n2, mlp, s in zip(self.mix, self.n1, self.n2, self.mlp, states, strict=True):
            y, s = mix.step(n1(x), s)
            x = x + y
            if self.c.mlp:
                x = x + mlp(n2(x))
            new.append(s)
        return self.norm(x) @ self.emb.weight.t(), new


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
