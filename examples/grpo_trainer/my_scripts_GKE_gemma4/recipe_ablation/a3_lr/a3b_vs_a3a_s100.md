### Paired A/B: `a3b_lr4e6` (B) vs `a3a_lr2e6` (A)

- A: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753`  B: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_a3lr4e6_16n64g_s100_seed1_1023_1116`; eval checkpoints in common: [0, 20, 40, 60, 80, 100]; training steps in common: 1–100; training dumps validated at every step (2048 rows = 256 x 8, one input / ground truth per group, identical qid -> input / ground truth in A and B).

| Eval step | A acc (%) | B acc (%) | B − A (pp) | paired SE | bootstrap 95% | +gained / −lost (B vs A) | A vs own step 0 (pp) | B vs own step 0 (pp, 95%) | identical outputs |
|---:|---:|---:|---:|---:|---|---|---:|---|---:|
| 0 | 36.06 | 35.96 | **-0.10** | 0.48 | [-1.02, +0.80] | +282 / −287 | — | — | 263 / 5000 |
| 20 | 39.26 | 39.92 | **+0.66** | 0.51 | [-0.30, +1.66] | +340 / −307 | +3.20 | +3.96 [+2.76, +5.18] | 0 / 5000 |
| 40 | 39.24 | 40.72 | **+1.48** | 0.55 | [+0.44, +2.56] | +411 / −337 | +3.18 | +4.76 [+3.62, +5.90] | 0 / 5000 |
| 60 | 40.52 | 36.80 | **-3.72** | 0.58 | [-4.86, -2.58] | +333 / −519 | +4.46 | +0.84 [-0.38, +2.12] | 0 / 5000 |
| 80 | 40.92 | 11.32 | **-29.60** | 0.66 | [-30.96, -28.24] | +20 / −1500 | +4.86 | -24.64 [-25.92, -23.42] | 0 / 5000 |
| 100 | 40.96 | 0.00 | **-40.96** | 0.70 | [-42.30, -39.62] | +0 / −2048 | +4.90 | -35.96 [-37.22, -34.60] | 0 / 5000 |

| Training steps (identical prompts) | n | A train acc (%) | B train acc (%) | B − A mean (pp) | SD of per-step B − A (pp) | steps with B ahead | A format (%) | B format (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 10 | 59.93 | 60.87 | **+0.94** | 1.02 | 8 / 10 | 98.22 | 98.87 |
| 11–20 | 10 | 60.30 | 61.11 | **+0.82** | 1.43 | 8 / 10 | 99.82 | 99.96 |
| 21–40 | 20 | 62.32 | 62.27 | **-0.05** | 0.85 | 8 / 20 | 99.87 | 99.91 |
| 41–60 | 20 | 62.40 | 61.47 | **-0.92** | 2.14 | 7 / 20 | 99.92 | 99.53 |
| 61–80 | 20 | 61.66 | 26.05 | **-35.60** | 17.87 | 0 / 20 | 99.94 | 42.54 |
| 81–100 | 20 | 62.68 | 0.72 | **-61.96** | 2.87 | 0 / 20 | 99.93 | 0.72 |
| 1–100 | 100 | 61.83 | 42.30 | **-19.53** | 26.66 | 31 / 100 | 99.74 | 68.42 |

| Steps | metric | A median | B median |
|---|---|---:|---:|
| 1-20 | `rollout_corr/k3_kl` | 0.0007227 | 0.0007434 |
| 1-20 | `actor/grad_norm` | 0.2676 | 0.3277 |
| 1-20 | `actor/entropy` | 0.2069 | 0.2157 |
| 1-20 | `response_length/mean` | 1730 | 1597 |
| 1-20 | `critic/score/mean` | 0.7072 | 0.7165 |
| 1-20 | `timing_s/step` | 62.33 | 59.33 |
| 1-20 | `response_length/clip_ratio` | 0.0004883 | 0.0004883 |
| 21-40 | `rollout_corr/k3_kl` | 0.0007548 | 0.001005 |
| 21-40 | `actor/grad_norm` | 0.3604 | 0.2134 |
| 21-40 | `actor/entropy` | 0.2241 | 0.3374 |
| 21-40 | `response_length/mean` | 1484 | 1206 |
| 21-40 | `critic/score/mean` | 0.7202 | 0.7249 |
| 21-40 | `timing_s/step` | 61.86 | 52.09 |
| 21-40 | `response_length/clip_ratio` | 0.0004883 | 0 |
| 41-60 | `rollout_corr/k3_kl` | 0.0008001 | 0.001743 |
| 41-60 | `actor/grad_norm` | 0.2183 | 0.3441 |
| 41-60 | `actor/entropy` | 0.2504 | 0.5405 |
| 41-60 | `response_length/mean` | 1403 | 1211 |
| 41-60 | `critic/score/mean` | 0.7286 | 0.7211 |
| 41-60 | `timing_s/step` | 59.84 | 57.78 |
| 41-60 | `response_length/clip_ratio` | 0.0004883 | 0 |
| 61-80 | `rollout_corr/k3_kl` | 0.0008704 | 3.643 |
| 61-80 | `actor/grad_norm` | 0.2439 | 15.59 |
| 61-80 | `actor/entropy` | 0.2865 | 9.135 |
| 61-80 | `response_length/mean` | 1385 | 5055 |
| 61-80 | `critic/score/mean` | 0.7135 | 0.1926 |
| 61-80 | `timing_s/step` | 57.85 | 131.4 |
| 61-80 | `response_length/clip_ratio` | 0 | 0.3621 |
| 81-100 | `rollout_corr/k3_kl` | 0.0009621 | 0.7517 |
| 81-100 | `actor/grad_norm` | 0.2157 | 0.03344 |
| 81-100 | `actor/entropy` | 0.3284 | 12.45 |
| 81-100 | `response_length/mean` | 1313 | 2198 |
| 81-100 | `critic/score/mean` | 0.7257 | 0.0007324 |
| 81-100 | `timing_s/step` | 65.05 | 103.3 |
| 81-100 | `response_length/clip_ratio` | 0.0004883 | 0.2156 |
