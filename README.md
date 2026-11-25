# ssm-from-scratch

[![CI](https://github.com/kishorhange111/ssm-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/ssm-from-scratch/actions/workflows/ci.yml)

Mamba-2 selective state-space models from scratch, in plain PyTorch, and a controlled
comparison with attention.

## One recurrence, three algorithms ([`scan.py`](src/ssm/scan.py))

The Mamba-2 layer with a scalar decay per head:

```
h_t = exp(a_t) · h_{t−1} + x_t ⊗ B_t        y_t = h_t · C_t        (a_t = Δ_t·A ≤ 0)
```

| algorithm | depth | work | idea |
|---|---|---|---|
| `scan_sequential` | O(L) | O(L·P·N) | the recurrence: reference, and the decode path |
| `scan_parallel` | O(log L) | O(L log L·P·N) | Hillis–Steele scan over `(a₁,b₁)∘(a₂,b₂) = (a₁a₂, a₂b₁+b₂)` |
| `ssd_chunked` | O(L/T) | O(L·T·N + L·N·P) | **state-space duality**: inside a chunk, `Y = (L ∘ CBᵀ)X` with decay mask `L_ij = exp(a_{j+1}+…+a_i)`, which is masked linear attention and maps to matmuls. Only the (H, P, N) state crosses chunks |

The tests check all three agree to float64 precision, including lengths that aren't chunk
multiples and a carried-in initial state. Another test checks that a single chunk is
*literally* masked attention, the duality in one assertion.

## Mamba-2 block ([`model.py`](src/ssm/model.py))

The block is `in_proj → causal depthwise conv → SiLU → selective SSD (softplus Δ, A = −exp(A_log), D skip) → gate by SiLU(z) → RMSNorm → out_proj`.
It has two execution modes with identical maths:
- `forward`, which uses chunked SSD (training and prefill);
- `step`, which decodes one token at a time with an **O(1) state** (conv buffer plus SSM
  state).

The tests check that token-by-token decoding reproduces the parallel forward, for the block
and for the whole LM.

## Experiments (GPU, via Colab)

| `--exp` | question |
|---|---|
| `mqar` | **Multi-query associative recall** (Zoology, Arora et al. 2023). After K key→value pairs, answer queries for the keys. Attention looks pairs up; an SSM must compress them into a fixed state. Sweep: K ∈ {8, 16, 32, 64} × state N ∈ {8, 32, 128}, against attention |
| `lm` | Byte-level LM on the Python stdlib at similar parameter counts: val bits/byte and training throughput |
| `python -m ssm.bench` | **Decode cost against context length.** One decode step with a KV cache of length L (grows with L) against the Mamba-2 recurrent step (flat), plus the decode-state bytes per sequence |

```
pytest -q                                          # 12 tests
python -m ssm.train --exp mqar --device cuda --out results
python -m ssm.train --exp lm   --device cuda --out results
python -m ssm.bench --out results
```

## Results (Tesla T4)

### Decode cost against context length ([`results/decode_Tesla_T4.md`](results/decode_Tesla_T4.md))

fp16, batch 8, d_model 256, 4 layers.

| context | attention step (KV cache) | Mamba-2 step | KV cache / seq | SSM state / seq |
|---|---|---|---|---|
| 256 | 2.49 ms | 3.72 ms | 1.0 MiB | 271 KiB |
| 1,024 | 2.32 ms | 3.72 ms | 4.0 MiB | 271 KiB |
| 4,096 | 4.26 ms | 3.72 ms | 16.0 MiB | 271 KiB |
| 16,384 | 15.70 ms | 3.72 ms | 64.0 MiB | 271 KiB |
| 65,536 | 65.23 ms | 3.72 ms | 256.0 MiB | 271 KiB |

The Mamba-2 step is **flat** in context length, while the attention step grows linearly
once the KV cache stops fitting in cache. Attention is faster at short contexts; the crossover lies between 1k and 4k tokens, and attention is
17.5× slower at 65k. The per-sequence decode state is 271 KiB against a 256 MiB KV cache,
almost 1,000× smaller, which is what lets recurrent models batch long contexts.

### Associative recall (MQAR) ([`results/mqar_runs.jsonl`](results/mqar_runs.jsonl))

These are 2-layer, d = 128 models with 4,000 steps × batch 64 each, a **single seed**, and
33 GPU-minutes in total. Accuracy is on the query positions; chance is 1/128 = 0.8%.

| mixer | params | K = 8 | K = 16 | K = 32 | K = 64 |
|---|---|---|---|---|---|
| attention | 296k | 35.3% | **100.0%** | 15.5% | 10.5% |
| Mamba-2, N = 8 | 239k | 34.5% | 22.3% | 7.4% | 1.2% |
| Mamba-2, N = 32 | 252k | 33.1% | 18.8% | 9.8% | 4.1% |
| Mamba-2, N = 128 | 303k | 35.2% | 22.2% | 10.1% | 3.7% |

What this does and doesn't show:
- **The recall gap appears where attention trains successfully.** At K = 16 attention is
  perfect, and every Mamba-2 variant is at 22% or below.
- **Attention's results are not monotone in K.** It fails at K = 8 but succeeds at K = 16,
  so its failures here are optimisation, not capacity. A lookup circuit forms abruptly,
  and in three of the four runs it hadn't formed within 4,000 steps. More steps or seeds
  are needed before reading anything into attention at K ∈ {8, 32, 64}.
- **Mamba-2 degrades smoothly with K** (about 34% → 21% → 9% → 3%), as a fixed-size state
  would predict. But a 16× larger state (N = 8 → 128) buys at most 3 points. At this
  budget the nominal state size is not the binding limit; 2 layers and 4,000 steps are.

### Byte-level LM ([`results/lm_runs.jsonl`](results/lm_runs.jsonl))

The task is the Python standard library's source, with context 512, batch 16 and 3,000
steps, in fp32: the T4 has no bf16. There is a single seed.

| mixer | params | val bits/byte | training throughput | wall time |
|---|---|---|---|---|
| attention | 3.48M | **1.325** | 94.5k tok/s | 4.4 min |
| Mamba-2, N = 64 | 3.91M | 1.472 | 44.9k tok/s | 9.2 min |

- **At this scale attention wins on both quality and speed.** Its validation loss is 0.147
  bits/byte lower, despite 11% fewer parameters, and it trains 2.1× faster.
- The speed gap is mostly about implementation. The attention model uses PyTorch's fused
  SDPA kernel. Mamba-2 runs the chunked SSD algorithm as plain PyTorch einsums (chunk 64),
  with no fused scan kernel like the official Triton one.
- The quality gap fits the recall results above. Source code is full of exact copies
  (identifiers, repeated lines) that attention can look up and a fixed state must
  compress. But this is 3,000 steps at 3.5M parameters, not a scaling claim.
- A third config, Mamba-2 with no MLP and 8 layers, did not finish within the 15-minute
  budget and is not reported.
