### Paired A/B: `fp32` (B) vs `fp32_abrun` (A)

- A: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05_16n64g_s60_fp32ab_seed1_1022_1744`  B: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_16n64g_s200_seed1_1022_2019`; eval checkpoints in common: [0, 20, 40, 60]; training steps in common: 1–60; training dumps validated at every step (2048 rows = 256 x 8, one input / ground truth per group, identical qid -> input / ground truth in A and B).

| Eval step | A acc (%) | B acc (%) | B − A (pp) | paired SE | bootstrap 95% | +gained / −lost (B vs A) | A vs own step 0 (pp) | B vs own step 0 (pp, 95%) | identical outputs |
|---:|---:|---:|---:|---:|---|---|---:|---|---:|
| 0 | 35.84 | 35.76 | **-0.08** | 0.46 | [-0.98, +0.80] | +268 / −272 | — | — | 291 / 5000 |
| 20 | 37.24 | 37.40 | **+0.16** | 0.50 | [-0.80, +1.10] | +312 / −304 | +1.40 | +1.64 [+0.66, +2.58] | 123 / 5000 |
| 40 | 37.86 | 38.52 | **+0.66** | 0.49 | [-0.36, +1.66] | +320 / −287 | +2.02 | +2.76 [+1.72, +3.80] | 83 / 5000 |
| 60 | 38.04 | 38.30 | **+0.26** | 0.50 | [-0.72, +1.24] | +315 / −302 | +2.20 | +2.54 [+1.46, +3.60] | 47 / 5000 |

| Training steps (identical prompts) | n | A train acc (%) | B train acc (%) | B − A mean (pp) | SD of per-step B − A (pp) | steps with B ahead | A format (%) | B format (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 10 | 59.80 | 59.64 | **-0.16** | 0.48 | 3 / 10 | 98.39 | 98.36 |
| 11–20 | 10 | 59.38 | 59.41 | **+0.03** | 0.63 | 6 / 10 | 99.31 | 99.37 |
| 21–40 | 20 | 61.40 | 61.52 | **+0.11** | 0.70 | 12 / 20 | 99.76 | 99.76 |
| 41–60 | 20 | 61.13 | 61.26 | **+0.13** | 0.96 | 11 / 20 | 99.78 | 99.78 |
| 1–60 | 60 | 60.71 | 60.77 | **+0.06** | 0.75 | 32 / 60 | 99.46 | 99.47 |

| Steps | metric | A median | B median |
|---|---|---:|---:|
| 1-20 | `rollout_corr/k3_kl` | 0.0006939 | 0.000703 |
| 1-20 | `actor/grad_norm` | 0.2413 | 0.3703 |
| 1-20 | `actor/entropy` | 0.197 | 0.1981 |
| 1-20 | `response_length/mean` | 1915 | 1857 |
| 1-20 | `critic/score/mean` | 0.6994 | 0.6996 |
| 1-20 | `timing_s/step` | 64.92 | 63.69 |
| 1-20 | `response_length/clip_ratio` | 0.0009766 | 0.0009766 |
| 21-40 | `rollout_corr/k3_kl` | 0.0007055 | 0.0006939 |
| 21-40 | `actor/grad_norm` | 0.3396 | 0.2394 |
| 21-40 | `actor/entropy` | 0.2014 | 0.1959 |
| 21-40 | `response_length/mean` | 1815 | 1728 |
| 21-40 | `critic/score/mean` | 0.7123 | 0.7146 |
| 21-40 | `timing_s/step` | 67.1 | 63.41 |
| 21-40 | `response_length/clip_ratio` | 0.0009766 | 0.0004883 |
| 41-60 | `rollout_corr/k3_kl` | 0.0007319 | 0.000703 |
| 41-60 | `actor/grad_norm` | 0.2178 | 0.2027 |
| 41-60 | `actor/entropy` | 0.2163 | 0.2019 |
| 41-60 | `response_length/mean` | 1806 | 1661 |
| 41-60 | `critic/score/mean` | 0.7151 | 0.7143 |
| 41-60 | `timing_s/step` | 67.81 | 66.33 |
| 41-60 | `response_length/clip_ratio` | 0.001221 | 0.0009766 |
