### Paired A/B: `fp32` (B) vs `bf16` (A)

- A: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05_16n64g_s200_seed1_1021_1955`  B: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05_16n64g_s60_fp32ab_seed1_1022_1744`; eval checkpoints in common: [0, 20, 40, 60]; training steps in common: 1–60; identical prompts at every common step: True.

| Eval step | A acc (%) | B acc (%) | B − A (pp) | paired SE | bootstrap 95% | +gained / −lost (B vs A) | A vs own step 0 (pp) | B vs own step 0 (pp, 95%) | identical outputs |
|---:|---:|---:|---:|---:|---|---|---:|---|---:|
| 0 | 36.26 | 35.84 | **-0.42** | 0.46 | [-1.32, +0.46] | +258 / −279 | — | — | 279 / 5000 |
| 20 | 36.34 | 37.24 | **+0.90** | 0.50 | [-0.06, +1.90] | +335 / −290 | +0.08 | +1.40 [+0.48, +2.44] | 128 / 5000 |
| 40 | 36.56 | 37.86 | **+1.30** | 0.51 | [+0.26, +2.30] | +353 / −288 | +0.30 | +2.02 [+1.04, +3.02] | 36 / 5000 |
| 60 | 37.32 | 38.04 | **+0.72** | 0.51 | [-0.24, +1.72] | +344 / −308 | +1.06 | +2.20 [+1.22, +3.20] | 29 / 5000 |

| Training steps (identical prompts) | n | A train acc (%) | B train acc (%) | B − A mean (pp) | SE (pp) | A format (%) | B format (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 10 | 59.56 | 59.80 | **+0.24** | 0.24 | 97.72 | 98.39 |
| 11–20 | 10 | 58.98 | 59.38 | **+0.41** | 0.28 | 98.19 | 99.31 |
| 21–40 | 20 | 59.92 | 61.40 | **+1.48** | 0.20 | 98.66 | 99.76 |
| 41–60 | 20 | 59.26 | 61.13 | **+1.87** | 0.23 | 98.98 | 99.78 |
| 1–60 | 60 | 59.48 | 60.71 | **+1.22** | 0.14 | 98.53 | 99.46 |

| Steps | metric | A median | B median |
|---|---|---:|---:|
| 1-20 | `rollout_corr/k3_kl` | 0.0006967 | 0.0006939 |
| 1-20 | `actor/grad_norm` | 0.2588 | 0.2413 |
| 1-20 | `actor/entropy` | 0.1964 | 0.197 |
| 1-20 | `response_length/mean` | 1937 | 1915 |
| 1-20 | `critic/score/mean` | 0.6907 | 0.6994 |
| 1-20 | `timing_s/step` | 69.19 | 64.92 |
| 1-20 | `response_length/clip_ratio` | 0.001953 | 0.0009766 |
| 21-40 | `rollout_corr/k3_kl` | 0.000688 | 0.0007055 |
| 21-40 | `actor/grad_norm` | 0.2427 | 0.3396 |
| 21-40 | `actor/entropy` | 0.1926 | 0.2014 |
| 21-40 | `response_length/mean` | 1887 | 1815 |
| 21-40 | `critic/score/mean` | 0.6997 | 0.7123 |
| 21-40 | `timing_s/step` | 66.27 | 67.1 |
| 21-40 | `response_length/clip_ratio` | 0.001953 | 0.0009766 |
| 41-60 | `rollout_corr/k3_kl` | 0.000686 | 0.0007319 |
| 41-60 | `actor/grad_norm` | 0.3115 | 0.2178 |
| 41-60 | `actor/entropy` | 0.1927 | 0.2163 |
| 41-60 | `response_length/mean` | 1917 | 1806 |
| 41-60 | `critic/score/mean` | 0.6992 | 0.7151 |
| 41-60 | `timing_s/step` | 68.82 | 67.81 |
| 41-60 | `response_length/clip_ratio` | 0.001465 | 0.001221 |
