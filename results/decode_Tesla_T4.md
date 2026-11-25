GPU **Tesla T4**, fp16, batch 8, d_model 256, 4 layers.

| context | attention step (KV cache) | Mamba-2 step | KV cache / seq | SSM state / seq |
|---|---|---|---|---|
| 256 | 2.49 ms | 3.72 ms | 1.0 MiB | 271 KiB |
| 1,024 | 2.32 ms | 3.72 ms | 4.0 MiB | 271 KiB |
| 4,096 | 4.26 ms | 3.72 ms | 16.0 MiB | 271 KiB |
| 16,384 | 15.70 ms | 3.72 ms | 64.0 MiB | 271 KiB |
| 65,536 | 65.23 ms | 3.72 ms | 256.0 MiB | 271 KiB |
