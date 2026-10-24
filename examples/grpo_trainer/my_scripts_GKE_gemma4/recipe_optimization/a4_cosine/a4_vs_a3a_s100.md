### Paired A/B: `a4_cosine` (B) vs `a3a_lr2e6` (A)

- A: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753`  B: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_a4cos_16n64g_s100_seed1_1023_2105`; eval checkpoints in common: [0, 20, 40, 60, 80, 100]; training steps in common: 1–100; training dumps validated at every step (2048 rows = 256 x 8, one input / ground truth per group, identical qid -> input / ground truth in A and B).

| Eval step | A acc (%) | B acc (%) | B − A (pp) | paired SE | bootstrap 95% | +gained / −lost (B vs A) | A vs own step 0 (pp) | B vs own step 0 (pp, 95%) | identical outputs |
|---:|---:|---:|---:|---:|---|---|---:|---|---:|
| 0 | 36.06 | 35.82 | **-0.24** | 0.48 | [-1.14, +0.72] | +280 / −292 | — | — | 273 / 5000 |
| 20 | 39.26 | 38.52 | **-0.74** | 0.51 | [-1.74, +0.28] | +305 / −342 | +3.20 | +2.70 [+1.62, +3.76] | 0 / 5000 |
| 40 | 39.24 | 39.14 | **-0.10** | 0.52 | [-1.14, +0.92] | +334 / −339 | +3.18 | +3.32 [+2.18, +4.34] | 0 / 5000 |
| 60 | 40.52 | 39.30 | **-1.22** | 0.52 | [-2.20, -0.18] | +314 / −375 | +4.46 | +3.48 [+2.38, +4.62] | 0 / 5000 |
| 80 | 40.92 | 40.14 | **-0.78** | 0.53 | [-1.84, +0.26] | +333 / −372 | +4.86 | +4.32 [+3.26, +5.44] | 0 / 5000 |
| 100 | 40.96 | 40.38 | **-0.58** | 0.52 | [-1.60, +0.36] | +318 / −347 | +4.90 | +4.56 [+3.50, +5.66] | 0 / 5000 |

| Training steps (identical prompts) | n | A train acc (%) | B train acc (%) | B − A mean (pp) | SD of per-step B − A (pp) | steps with B ahead | A format (%) | B format (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 10 | 59.93 | 60.14 | **+0.21** | 0.82 | 6 / 10 | 98.22 | 99.19 |
| 11–20 | 10 | 60.30 | 60.34 | **+0.04** | 0.51 | 6 / 10 | 99.82 | 99.79 |
| 21–40 | 20 | 62.32 | 62.36 | **+0.03** | 1.10 | 10 / 20 | 99.87 | 99.86 |
| 41–60 | 20 | 62.40 | 61.93 | **-0.46** | 0.93 | 5 / 20 | 99.92 | 99.92 |
| 61–80 | 20 | 61.66 | 61.71 | **+0.06** | 0.68 | 10 / 20 | 99.94 | 99.96 |
| 81–100 | 20 | 62.68 | 62.08 | **-0.61** | 0.76 | 4 / 20 | 99.93 | 99.95 |
| 1–100 | 100 | 61.83 | 61.66 | **-0.17** | 0.88 | 41 / 100 | 99.74 | 99.84 |

| Steps | metric | A median | B median |
|---|---|---:|---:|
| 1-20 | `rollout_corr/k3_kl` | 0.0007227 | 0.0007069 |
| 1-20 | `actor/grad_norm` | 0.2676 | 0.2154 |
| 1-20 | `actor/entropy` | 0.2069 | 0.2039 |
| 1-20 | `response_length/mean` | 1730 | 1825 |
| 1-20 | `critic/score/mean` | 0.7072 | 0.7058 |
| 1-20 | `timing_s/step` | 62.33 | 60.84 |
| 1-20 | `response_length/clip_ratio` | 0.0004883 | 0.0009766 |
| 21-40 | `rollout_corr/k3_kl` | 0.0007548 | 0.0007499 |
| 21-40 | `actor/grad_norm` | 0.3604 | 0.2645 |
| 21-40 | `actor/entropy` | 0.2241 | 0.2217 |
| 21-40 | `response_length/mean` | 1484 | 1675 |
| 21-40 | `critic/score/mean` | 0.7202 | 0.7269 |
| 21-40 | `timing_s/step` | 61.86 | 63.22 |
| 21-40 | `response_length/clip_ratio` | 0.0004883 | 0.0004883 |
| 41-60 | `rollout_corr/k3_kl` | 0.0008001 | 0.0008204 |
| 41-60 | `actor/grad_norm` | 0.2183 | 0.2393 |
| 41-60 | `actor/entropy` | 0.2504 | 0.2555 |
| 41-60 | `response_length/mean` | 1403 | 1604 |
| 41-60 | `critic/score/mean` | 0.7286 | 0.7203 |
| 41-60 | `timing_s/step` | 59.84 | 66.36 |
| 41-60 | `response_length/clip_ratio` | 0.0004883 | 0.0004883 |
| 61-80 | `rollout_corr/k3_kl` | 0.0008704 | 0.0008813 |
| 61-80 | `actor/grad_norm` | 0.2439 | 0.2538 |
| 61-80 | `actor/entropy` | 0.2865 | 0.2809 |
| 61-80 | `response_length/mean` | 1385 | 1543 |
| 61-80 | `critic/score/mean` | 0.7135 | 0.7161 |
| 61-80 | `timing_s/step` | 57.85 | 64.37 |
| 61-80 | `response_length/clip_ratio` | 0 | 0.0004883 |
| 81-100 | `rollout_corr/k3_kl` | 0.0009621 | 0.000996 |
| 81-100 | `actor/grad_norm` | 0.2157 | 0.2226 |
| 81-100 | `actor/entropy` | 0.3284 | 0.3368 |
| 81-100 | `response_length/mean` | 1313 | 1548 |
| 81-100 | `critic/score/mean` | 0.7257 | 0.7189 |
| 81-100 | `timing_s/step` | 65.05 | 66.52 |
| 81-100 | `response_length/clip_ratio` | 0.0004883 | 0.001953 |
