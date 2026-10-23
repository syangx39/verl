### Paired A/B: `a3a_lr2e6` (B) vs `a0` (A)

- A: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_16n64g_s200_seed1_1022_2019`  B: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753`; eval checkpoints in common: [0, 20, 40, 60, 80, 100]; training steps in common: 1–100; training dumps validated at every step (2048 rows = 256 x 8, one input / ground truth per group, identical qid -> input / ground truth in A and B).

| Eval step | A acc (%) | B acc (%) | B − A (pp) | paired SE | bootstrap 95% | +gained / −lost (B vs A) | A vs own step 0 (pp) | B vs own step 0 (pp, 95%) | identical outputs |
|---:|---:|---:|---:|---:|---|---|---:|---|---:|
| 0 | 35.76 | 36.06 | **+0.30** | 0.46 | [-0.62, +1.16] | +272 / −257 | — | — | 268 / 5000 |
| 20 | 37.40 | 39.26 | **+1.86** | 0.51 | [+0.92, +2.84] | +378 / −285 | +1.64 | +3.20 [+2.16, +4.22] | 0 / 5000 |
| 40 | 38.52 | 39.24 | **+0.72** | 0.52 | [-0.34, +1.76] | +353 / −317 | +2.76 | +3.18 [+2.08, +4.26] | 0 / 5000 |
| 60 | 38.30 | 40.52 | **+2.22** | 0.54 | [+1.20, +3.30] | +417 / −306 | +2.54 | +4.46 [+3.32, +5.58] | 0 / 5000 |
| 80 | 38.76 | 40.92 | **+2.16** | 0.55 | [+1.08, +3.20] | +431 / −323 | +3.00 | +4.86 [+3.72, +6.00] | 0 / 5000 |
| 100 | 39.62 | 40.96 | **+1.34** | 0.52 | [+0.32, +2.44] | +377 / −310 | +3.86 | +4.90 [+3.76, +6.06] | 0 / 5000 |

| Training steps (identical prompts) | n | A train acc (%) | B train acc (%) | B − A mean (pp) | SD of per-step B − A (pp) | steps with B ahead | A format (%) | B format (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 10 | 59.64 | 59.93 | **+0.29** | 0.98 | 6 / 10 | 98.36 | 98.22 |
| 11–20 | 10 | 59.41 | 60.30 | **+0.88** | 0.51 | 9 / 10 | 99.37 | 99.82 |
| 21–40 | 20 | 61.52 | 62.32 | **+0.81** | 1.08 | 16 / 20 | 99.76 | 99.87 |
| 41–60 | 20 | 61.26 | 62.40 | **+1.14** | 0.93 | 16 / 20 | 99.78 | 99.92 |
| 61–80 | 20 | 61.05 | 61.66 | **+0.61** | 0.77 | 15 / 20 | 99.83 | 99.94 |
| 81–100 | 20 | 61.76 | 62.68 | **+0.92** | 1.13 | 15 / 20 | 99.85 | 99.93 |
| 1–100 | 100 | 61.02 | 61.83 | **+0.81** | 0.96 | 77 / 100 | 99.62 | 99.74 |

| Steps | metric | A median | B median |
|---|---|---:|---:|
| 1-20 | `rollout_corr/k3_kl` | 0.000703 | 0.0007227 |
| 1-20 | `actor/grad_norm` | 0.3703 | 0.2676 |
| 1-20 | `actor/entropy` | 0.1981 | 0.2069 |
| 1-20 | `response_length/mean` | 1857 | 1730 |
| 1-20 | `critic/score/mean` | 0.6996 | 0.7072 |
| 1-20 | `timing_s/step` | 63.69 | 62.33 |
| 1-20 | `response_length/clip_ratio` | 0.0009766 | 0.0004883 |
| 21-40 | `rollout_corr/k3_kl` | 0.0006939 | 0.0007548 |
| 21-40 | `actor/grad_norm` | 0.2394 | 0.3604 |
| 21-40 | `actor/entropy` | 0.1959 | 0.2241 |
| 21-40 | `response_length/mean` | 1728 | 1484 |
| 21-40 | `critic/score/mean` | 0.7146 | 0.7202 |
| 21-40 | `timing_s/step` | 63.41 | 61.86 |
| 21-40 | `response_length/clip_ratio` | 0.0004883 | 0.0004883 |
| 41-60 | `rollout_corr/k3_kl` | 0.000703 | 0.0008001 |
| 41-60 | `actor/grad_norm` | 0.2027 | 0.2183 |
| 41-60 | `actor/entropy` | 0.2019 | 0.2504 |
| 41-60 | `response_length/mean` | 1661 | 1403 |
| 41-60 | `critic/score/mean` | 0.7143 | 0.7286 |
| 41-60 | `timing_s/step` | 66.33 | 59.84 |
| 41-60 | `response_length/clip_ratio` | 0.0009766 | 0.0004883 |
| 61-80 | `rollout_corr/k3_kl` | 0.0007161 | 0.0008704 |
| 61-80 | `actor/grad_norm` | 0.2353 | 0.2439 |
| 61-80 | `actor/entropy` | 0.2079 | 0.2865 |
| 61-80 | `response_length/mean` | 1611 | 1385 |
| 61-80 | `critic/score/mean` | 0.7073 | 0.7135 |
| 61-80 | `timing_s/step` | 64.64 | 57.85 |
| 61-80 | `response_length/clip_ratio` | 0.0004883 | 0 |
| 81-100 | `rollout_corr/k3_kl` | 0.0007303 | 0.0009621 |
| 81-100 | `actor/grad_norm` | 0.2731 | 0.2157 |
| 81-100 | `actor/entropy` | 0.2155 | 0.3284 |
| 81-100 | `response_length/mean` | 1576 | 1313 |
| 81-100 | `critic/score/mean` | 0.7149 | 0.7257 |
| 81-100 | `timing_s/step` | 62.46 | 65.05 |
| 81-100 | `response_length/clip_ratio` | 0.0004883 | 0.0004883 |
