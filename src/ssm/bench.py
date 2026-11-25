"""Decode cost vs context length: recurrent Mamba-2 state against an attention KV cache.

    python -m ssm.bench --out results     (GPU)

For each context length L, time one decode step (batch 8):
- Mamba-2: ``LM.step`` with an O(1) state. Cost should be flat in L;
- attention: a KV cache of length L per layer, one query token attending to all of it.
  Cost grows with L, since every step reads the whole cache.

It also reports bytes of decode state per sequence: the KV cache grows as 2·layers·L·d,
the SSM state is fixed at layers·(H·P·N + conv buffer).
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from .model import LM, Config


def _ms(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def attention_step(model: LM, tok, cache_k, cache_v):
    """One token through the attention LM with a static KV cache (no cache growth)."""
    x = model.emb(tok) + model.pos.weight[cache_k[0].shape[2] % model.c.max_len]
    for i, (mix, n1, n2, mlp) in enumerate(zip(model.mix, model.n1, model.n2, model.mlp,
                                               strict=True)):  # fmt: skip
        b, d = x.shape
        q, k, v = mix.qkv(n1(x)).view(b, 3, mix.h, d // mix.h).unbind(1)
        K = torch.cat([cache_k[i], k[:, :, None]], 2)
        V = torch.cat([cache_v[i], v[:, :, None]], 2)
        y = F.scaled_dot_product_attention(q[:, :, None], K, V)
        x = x + mix.proj(y.reshape(b, d))
        x = x + mlp(n2(x))
    return model.norm(x) @ model.emb.weight.t()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--lens", type=int, nargs="+", default=[256, 1024, 4096, 16384, 65536])
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)
    dev, dt = "cuda", torch.float16
    os.makedirs(a.out, exist_ok=True)
    mamba = LM(Config()).to(dev, dt).eval()
    attn = LM(Config(mixer="attention", max_len=max(a.lens) + 1)).to(dev, dt).eval()
    b, c = a.batch, attn.c
    tok = torch.randint(0, 256, (b,), device=dev)
    states = [m.init_state(b, dev, dt) for m in mamba.mix]
    ssm_ms = _ms(lambda: mamba.step(tok, states))
    m0 = mamba.mix[0]
    ssm_bytes = c.n_layers * (m0.h * m0.p * m0.n + m0.conv_dim * (m0.k - 1)) * 2
    rows = []
    for L in a.lens:
        hd = c.d_model // c.n_heads
        ks = [torch.randn(b, c.n_heads, L, hd, device=dev, dtype=dt) for _ in range(c.n_layers)]
        vs = [torch.randn_like(k) for k in ks]
        ms = _ms(lambda ks=ks, vs=vs: attention_step(attn, tok, ks, vs))
        rows.append({"context": L, "attention_ms": ms, "mamba2_ms": ssm_ms,
                     "kv_bytes_per_seq": 2 * c.n_layers * L * c.d_model * 2,
                     "ssm_state_bytes_per_seq": ssm_bytes})  # fmt: skip
        print(rows[-1], flush=True)
        del ks, vs
        torch.cuda.empty_cache()
    gpu = torch.cuda.get_device_name()
    lines = [f"GPU **{gpu}**, fp16, batch {b}, d_model {c.d_model}, {c.n_layers} layers.\n",
             "| context | attention step (KV cache) | Mamba-2 step | KV cache / seq "
             "| SSM state / seq |",
             "|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        lines.append(f"| {r['context']:,} | {r['attention_ms']:.2f} ms | {r['mamba2_ms']:.2f} ms | "
                     f"{r['kv_bytes_per_seq'] / 2**20:.1f} MiB | "
                     f"{r['ssm_state_bytes_per_seq'] / 2**10:.0f} KiB |")  # fmt: skip
    with open(os.path.join(a.out, f"decode_{gpu.replace(' ', '_')}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, f"decode_{gpu.replace(' ', '_')}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
