### Optimizer write-back probe: `fp32`

- Probe dir: `/workspace/meta-RL/logs/gemma4_optim_probe/gemma4_e2b_rl05_16n64g_optprobe_fp32_1022_0829`; init records from 64/64 ranks (complete); optimizer ['AdamW'], torch ['2.13.0+cu130'].
- Optimizer-held parameter dtypes at construction: ['torch.float32']; total elements held over all ranks: 5,104,297,536.
- AdamW defaults: lr=1e-06, betas=[0.9, 0.999], eps=1e-08, weight_decay=0.01, foreach=None, fused=None.

| Call | Ranks | AdamW step t | lr | param / grad / exp_avg / exp_avg_sq dtype | elements | nonzero grad | changed | changed / nonzero grad | predicted to survive | ‖applied‖ / ‖ideal‖ | cos(applied, ideal) | ‖Δw‖ / ‖w‖ (applied) | ‖Δw‖ / ‖w‖ (ideal fp32) | post-clip grad norm |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64/64 | [1] | [1e-06] | torch.float32/torch.float32/torch.float32/torch.float32 | 4,634,769,472 | 50.50% | **42.400%** | 83.955% | 42.405% | **1.0000** | 1.0000 | 3.872e-06 | 3.872e-06 | 0.2197 |
| 1 | 64/64 | [2] | [1e-06] | torch.float32/torch.float32/torch.float32/torch.float32 | 4,634,769,472 | 50.49% | **42.757%** | 84.691% | 42.762% | **1.0000** | 1.0000 | 2.861e-06 | 2.861e-06 | 0.5460 |

- Call 0: measured changed set vs round-to-nearest prediction: 99.96% of changed elements were predicted; 99.95% of predicted elements changed. Mean |ideal| = 3.816e-07, mean |applied| = 3.813e-07.
- Call 1: measured changed set vs round-to-nearest prediction: 99.95% of changed elements were predicted; 99.94% of predicted elements changed. Mean |ideal| = 2.583e-07, mean |applied| = 2.581e-07.
