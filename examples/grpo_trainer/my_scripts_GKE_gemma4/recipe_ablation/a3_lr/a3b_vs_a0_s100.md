### Paired A/B: `a3b_lr4e6` (B) vs `a0` (A)

- A: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_16n64g_s200_seed1_1022_2019`  B: `/workspace/meta-RL/logs/gemma4_eval/gemma4_e2b_rl05fp32_a3lr4e6_16n64g_s100_seed1_1023_1116`; eval checkpoints in common: [0, 20, 40, 60, 80, 100]; training steps in common: 1–100; training dumps validated at every step (2048 rows = 256 x 8, one input / ground truth per group, identical qid -> input / ground truth in A and B).

| Eval step | A acc (%) | B acc (%) | B − A (pp) | paired SE | bootstrap 95% | +gained / −lost (B vs A) | A vs own step 0 (pp) | B vs own step 0 (pp, 95%) | identical outputs |
|---:|---:|---:|---:|---:|---|---|---:|---|---:|
| 0 | 35.76 | 35.96 | **+0.20** | 0.47 | [-0.76, +1.08] | +284 / −274 | — | — | 268 / 5000 |
| 20 | 37.40 | 39.92 | **+2.52** | 0.56 | [+1.54, +3.60] | +463 / −337 | +1.64 | +3.96 [+2.76, +5.18] | 0 / 5000 |
| 40 | 38.52 | 40.72 | **+2.20** | 0.58 | [+1.08, +3.28] | +470 / −360 | +2.76 | +4.76 [+3.62, +5.90] | 0 / 5000 |
| 60 | 38.30 | 36.80 | **-1.50** | 0.59 | [-2.58, -0.30] | +399 / −474 | +2.54 | +0.84 [-0.38, +2.12] | 0 / 5000 |
| 80 | 38.76 | 11.32 | **-27.44** | 0.65 | [-28.74, -26.20] | +24 / −1396 | +3.00 | -24.64 [-25.92, -23.42] | 0 / 5000 |
| 100 | 39.62 | 0.00 | **-39.62** | 0.69 | [-40.92, -38.28] | +0 / −1981 | +3.86 | -35.96 [-37.22, -34.60] | 0 / 5000 |

| Training steps (identical prompts) | n | A train acc (%) | B train acc (%) | B − A mean (pp) | SD of per-step B − A (pp) | steps with B ahead | A format (%) | B format (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 10 | 59.64 | 60.87 | **+1.23** | 1.06 | 9 / 10 | 98.36 | 98.87 |
| 11–20 | 10 | 59.41 | 61.11 | **+1.70** | 1.57 | 9 / 10 | 99.37 | 99.96 |
| 21–40 | 20 | 61.52 | 62.27 | **+0.75** | 1.40 | 17 / 20 | 99.76 | 99.91 |
| 41–60 | 20 | 61.26 | 61.47 | **+0.21** | 2.10 | 12 / 20 | 99.78 | 99.53 |
| 61–80 | 20 | 61.05 | 26.05 | **-35.00** | 17.92 | 0 / 20 | 99.83 | 42.54 |
| 81–100 | 20 | 61.76 | 0.72 | **-61.04** | 2.98 | 0 / 20 | 99.85 | 0.72 |
| 1–100 | 100 | 61.02 | 42.30 | **-18.72** | 26.67 | 47 / 100 | 99.62 | 68.42 |

| Steps | metric | A median | B median |
|---|---|---:|---:|
| 1-20 | `rollout_corr/k3_kl` | 0.000703 | 0.0007434 |
| 1-20 | `actor/grad_norm` | 0.3703 | 0.3277 |
| 1-20 | `actor/entropy` | 0.1981 | 0.2157 |
| 1-20 | `response_length/mean` | 1857 | 1597 |
| 1-20 | `critic/score/mean` | 0.6996 | 0.7165 |
| 1-20 | `timing_s/step` | 63.69 | 59.33 |
| 1-20 | `response_length/clip_ratio` | 0.0009766 | 0.0004883 |
| 21-40 | `rollout_corr/k3_kl` | 0.0006939 | 0.001005 |
| 21-40 | `actor/grad_norm` | 0.2394 | 0.2134 |
| 21-40 | `actor/entropy` | 0.1959 | 0.3374 |
| 21-40 | `response_length/mean` | 1728 | 1206 |
| 21-40 | `critic/score/mean` | 0.7146 | 0.7249 |
| 21-40 | `timing_s/step` | 63.41 | 52.09 |
| 21-40 | `response_length/clip_ratio` | 0.0004883 | 0 |
| 41-60 | `rollout_corr/k3_kl` | 0.000703 | 0.001743 |
| 41-60 | `actor/grad_norm` | 0.2027 | 0.3441 |
| 41-60 | `actor/entropy` | 0.2019 | 0.5405 |
| 41-60 | `response_length/mean` | 1661 | 1211 |
| 41-60 | `critic/score/mean` | 0.7143 | 0.7211 |
| 41-60 | `timing_s/step` | 66.33 | 57.78 |
| 41-60 | `response_length/clip_ratio` | 0.0009766 | 0 |
| 61-80 | `rollout_corr/k3_kl` | 0.0007161 | 3.643 |
| 61-80 | `actor/grad_norm` | 0.2353 | 15.59 |
| 61-80 | `actor/entropy` | 0.2079 | 9.135 |
| 61-80 | `response_length/mean` | 1611 | 5055 |
| 61-80 | `critic/score/mean` | 0.7073 | 0.1926 |
| 61-80 | `timing_s/step` | 64.64 | 131.4 |
| 61-80 | `response_length/clip_ratio` | 0.0004883 | 0.3621 |
| 81-100 | `rollout_corr/k3_kl` | 0.0007303 | 0.7517 |
| 81-100 | `actor/grad_norm` | 0.2731 | 0.03344 |
| 81-100 | `actor/entropy` | 0.2155 | 12.45 |
| 81-100 | `response_length/mean` | 1576 | 2198 |
| 81-100 | `critic/score/mean` | 0.7149 | 0.0007324 |
| 81-100 | `timing_s/step` | 62.46 | 103.3 |
| 81-100 | `response_length/clip_ratio` | 0.0004883 | 0.2156 |
