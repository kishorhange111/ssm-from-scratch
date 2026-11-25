"""Three exact algorithms for the same selective state-space recurrence.

Mamba-2 form with a scalar decay per head (the "SSD" layer, Dao & Gu 2024):

    h_t = exp(a_t) · h_{t-1} + x_t ⊗ B_t        h: (heads, P, N)
    y_t = h_t · C_t                              y: (heads, P)

shapes: x (b, l, h, p), a (b, l, h) with a ≤ 0 (log-decay = Δ·A), B and C (b, l, h, n).

1. ``scan_sequential``: the recurrence, one step at a time. O(L) depth; the reference.
2. ``scan_parallel``: Hillis–Steele associative scan over the pairs (decay, input) with
   ``(a1, b1) ∘ (a2, b2) = (a1·a2, a2·b1 + b2)``. O(log L) depth, O(L log L) work.
3. ``ssd_chunked``: the state-space duality. Within a chunk of T steps, the output is
   attention-like, ``Y = (L ∘ C Bᵀ) X`` with the causal decay mask
   ``L_ij = exp(a_{j+1} + … + a_i)``: a quadratic, matmul-friendly form that tensor cores
   like. Across chunks only the (h, P, N) state is passed on, recurrently. Cost
   O(L·T·N + L·N·P): linear in L, and mostly matmuls.

All three agree to floating-point precision; the tests check this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def scan_sequential(x, a, B, C, h0=None):
    b, l, h, p = x.shape
    n = B.shape[-1]
    state = x.new_zeros(b, h, p, n) if h0 is None else h0
    ys = []
    for t in range(l):
        state = (
            torch.exp(a[:, t])[..., None, None] * state + x[:, t, :, :, None] * B[:, t, :, None, :]
        )
        ys.append((state * C[:, t, :, None, :]).sum(-1))
    return torch.stack(ys, 1), state


def scan_parallel(x, a, B, C):
    """Hillis–Steele inclusive scan over time. Materialises (b, l, h, p, n); for testing
    and small shapes, not for long sequences."""
    l = x.shape[1]
    decay = torch.exp(a)[..., None, None]  # (b, l, h, 1, 1)
    inp = x[..., None] * B[:, :, :, None, :]  # (b, l, h, p, n)
    off = 1
    while off < l:
        inp = torch.cat([inp[:, :off], decay[:, off:] * inp[:, :-off] + inp[:, off:]], 1)
        decay = torch.cat([decay[:, :off], decay[:, off:] * decay[:, :-off]], 1)
        off *= 2
    y = (inp * C[:, :, :, None, :]).sum(-1)
    return y, inp[:, -1]


def segsum(x: torch.Tensor) -> torch.Tensor:
    """out[..., i, j] = x[j+1] + ... + x[i] for i >= j, else -inf. Built with a masked
    cumsum, not a difference of cumsums, which would lose precision for long chunks."""
    T = x.shape[-1]
    x = x[..., None].expand(*x.shape, T)  # x[..., i, j] = x_i
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1)
    s = x.masked_fill(~strict, 0).cumsum(-2)
    return s.masked_fill(~torch.tril(torch.ones_like(strict), 0), float("-inf"))


def ssd_chunked(x, a, B, C, chunk: int = 64, h0=None):
    b, l, h, p = x.shape
    pad = (-l) % chunk
    if pad:  # zero inputs with zero log-decay leave the state untouched
        x, B, C = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (x, B, C))
        a = F.pad(a, (0, 0, 0, pad))
    c = x.shape[1] // chunk
    X = x.view(b, c, chunk, h, p)
    Bc = B.view(b, c, chunk, h, -1)
    Cc = C.view(b, c, chunk, h, -1)
    A = a.view(b, c, chunk, h).permute(0, 3, 1, 2)  # (b, h, c, T)
    A_cs = A.cumsum(-1)

    # 1. intra-chunk: the attention-like quadratic form
    Lmask = torch.exp(segsum(A))  # (b, h, c, T, T)
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc, Bc, Lmask, X)

    # 2. state at the end of each chunk, from that chunk's inputs alone
    decay_to_end = torch.exp(A_cs[..., -1:] - A_cs)  # (b, h, c, T)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_to_end, X)

    # 3. pass states across chunks (a short recurrence over c, written as a decay matrix)
    init = x.new_zeros(b, 1, h, p, Bc.shape[-1]) if h0 is None else h0[:, None]
    states = torch.cat([init, states], 1)  # (b, c+1, h, p, n)
    chunk_decay = torch.exp(segsum(F.pad(A_cs[..., -1], (1, 0))))  # (b, h, c+1, c+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
    carried, final = states[:, :-1], states[:, -1]

    # 4. contribution of the carried-in state to each position
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, carried, torch.exp(A_cs))
    y = (y_diag + y_off).reshape(b, c * chunk, h, p)[:, :l]
    return y, final
