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

**Results are pending** from the Colab runs.
