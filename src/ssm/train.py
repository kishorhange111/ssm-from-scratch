"""Experiments: Mamba-2 vs attention.

    python -m ssm.train --exp mqar --device cuda --out results
    python -m ssm.train --exp lm   --device cuda --out results

- ``mqar``: multi-query associative recall (Arora et al. 2023, "Zoology"). The sequence
  lists K key→value pairs, then queries the keys in random order; the model must emit each
  key's value. Attention can look the pair up directly. A recurrent model must have
  *compressed* all K pairs into its fixed state, so accuracy should fall as K outgrows the
  state. The sweep covers K and the state size N.
- ``lm``: byte-level language modelling on the Python standard library's source, at similar
  parameter counts, reporting val bits/byte and training throughput.

Each finished run appends a JSON line to ``<out>/runs.jsonl`` and is skipped on restart.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sysconfig
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from .model import LM, Config, n_params

KEYS, VALS = (1, 128), (128, 256)  # disjoint token ranges for keys and values


def mqar_batch(bs: int, n_pairs: int, gen: torch.Generator, device="cpu"):
    """Tokens (bs, 4K) and targets with -100 everywhere except after each query key."""
    keys = torch.stack([torch.randperm(KEYS[1] - KEYS[0], generator=gen)[:n_pairs]
                        for _ in range(bs)]) + KEYS[0]  # fmt: skip
    vals = torch.randint(VALS[0], VALS[1], (bs, n_pairs), generator=gen)
    order = torch.stack([torch.randperm(n_pairs, generator=gen) for _ in range(bs)])
    qk, qv = keys.gather(1, order), vals.gather(1, order)
    ctx = torch.stack([keys, vals], 2).flatten(1)
    qry = torch.stack([qk, qv], 2).flatten(1)
    x = torch.cat([ctx, qry], 1)
    y = torch.full_like(x, -100)
    y[:, 2 * n_pairs :: 2] = qv  # predict the value at each query-key position
    return x.to(device), y.to(device)


def train_mqar(cfg: Config, n_pairs: int, steps: int, device: str, bs: int = 64,
               lr: float = 1e-3, seed: int = 0) -> dict:  # fmt: skip
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = LM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = mqar_batch(bs, n_pairs, gen, device)
        loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(999)
        correct = total = 0
        for _ in range(8):
            x, y = mqar_batch(128, n_pairs, g, device)
            pred = model(x).argmax(-1)
            m = y != -100
            correct += int((pred[m] == y[m]).sum())
            total += int(m.sum())
    return {"accuracy": correct / total, "params": n_params(model), "final_loss": loss.item()}


def load_corpus():
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            (va if zlib.crc32(rel.encode()) % 1000 < 50 else tr).append(fh.read())
    return (np.frombuffer(b"\n".join(tr), np.uint8).copy(),
            np.frombuffer(b"\n".join(va), np.uint8).copy())  # fmt: skip


def lm_batch(data, bs, ctx, gen, device):
    ix = torch.randint(len(data) - ctx - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i : i + ctx + 1].astype(np.int64)) for i in ix])
    return x[:, :-1].to(device), x[:, 1:].to(device)


def train_lm(cfg: Config, steps: int, device: str, ctx: int = 512, bs: int = 16,
             lr: float = 1e-3, seed: int = 0, data=None) -> dict:  # fmt: skip
    train, val = data or load_corpus()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = LM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    amp = device == "cuda"
    t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = lm_batch(train, bs, ctx, gen, device)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    model.eval()
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
        vl = [float(F.cross_entropy(model(x).float().flatten(0, 1), y.flatten()))
              for x, y in (lm_batch(val, 16, ctx, g, device) for _ in range(20))]  # fmt: skip
    v = float(np.mean(vl))
    return {"val_bpb": v / math.log(2), "params": n_params(model),
            "train_tok_per_s": steps * bs * ctx / secs, "seconds": secs}  # fmt: skip


def runs(exp: str, steps: int):
    if exp == "mqar":
        for K in (8, 16, 32, 64):
            yield f"attention K={K}", ("mqar", Config(d_model=128, n_layers=2, mixer="attention",
                                                      n_heads=2, mlp=False), K)  # fmt: skip
            for N in (8, 32, 128):
                yield f"mamba2 N={N} K={K}", ("mqar", Config(d_model=128, n_layers=2, d_state=N,
                                                             mlp=False), K)  # fmt: skip
    if exp == "lm":
        yield "attention", ("lm", Config(mixer="attention"), None)
        yield "mamba2 N=64", ("lm", Config(d_state=64), None)
        yield "mamba2 N=64 no-MLP x2 layers", ("lm", Config(d_state=64, mlp=False, n_layers=8),
                                               None)  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["mqar", "lm"], required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["exp"], r["name"]) for r in map(json.loads, open(path)) if r}
    steps = a.steps or (4000 if a.exp == "mqar" else 3000)
    data = load_corpus() if a.exp == "lm" else None
    for name, (kind, cfg, K) in runs(a.exp, steps):
        if (a.exp, name) in done:
            continue
        t = time.time()
        if kind == "mqar":
            r = train_mqar(cfg, K, steps, a.device)
        else:
            r = train_lm(cfg, steps, a.device, data=data)
        rec = {"exp": a.exp, "name": name, "K": K, "d_state": cfg.d_state, "mixer": cfg.mixer,
               "steps": steps, "minutes": (time.time() - t) / 60, **r}  # fmt: skip
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
