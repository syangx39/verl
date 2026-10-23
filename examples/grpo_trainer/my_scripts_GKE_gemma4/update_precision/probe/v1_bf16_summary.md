### Optimizer write-back probe: `bf16`

- Probe dir: `/workspace/meta-RL/logs/gemma4_optim_probe/gemma4_e2b_rl05_16n64g_optprobe_bf16_1022_0816`; init records from 64/64 ranks (complete); optimizer ['AdamW'], torch ['2.13.0+cu130'].
- Optimizer-held parameter dtypes at construction: ['torch.bfloat16']; total elements held over all ranks: 5,104,297,536.
- AdamW defaults: lr=1e-06, betas=[0.9, 0.999], eps=1e-08, weight_decay=0.01, foreach=None, fused=None.

| Call | Ranks | AdamW step t | lr | param / grad / exp_avg / exp_avg_sq dtype | elements | nonzero grad | changed | changed / nonzero grad | predicted to survive | ‖applied‖ / ‖ideal‖ | cos(applied, ideal) | ‖Δw‖ / ‖w‖ (applied) | ‖Δw‖ / ‖w‖ (ideal fp32) | post-clip grad norm |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64/64 | [1] | [1e-06] | torch.bfloat16/torch.bfloat16/torch.bfloat16/torch.bfloat16 | 4,634,769,472 | 50.50% | **0.504%** | 0.998% | 0.503% | **0.1671** | 0.1074 | 6.522e-07 | 3.903e-06 | 0.7420 |
| 1 | 64/64 | [2] | [1e-06] | torch.bfloat16/torch.bfloat16/torch.bfloat16/torch.bfloat16 | 4,634,769,472 | 50.38% | **0.272%** | 0.540% | 0.271% | **0.1237** | 0.0900 | 3.501e-07 | 2.831e-06 | 0.5013 |

- Call 0: measured changed set vs round-to-nearest prediction: 99.49% of changed elements were predicted; 99.64% of predicted elements changed. Mean |ideal| = 3.851e-07, mean |applied| = 6.728e-09.
- Call 1: measured changed set vs round-to-nearest prediction: 99.32% of changed elements were predicted; 99.62% of predicted elements changed. Mean |ideal| = 2.546e-07, mean |applied| = 2.565e-09.
