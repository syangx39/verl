### qwen: eval accuracy re-scored, v1 -> v2 (%), same outputs

| step | qwen_seed1 v1 | qwen_seed1 v2 | qwen_seed2 v1 | qwen_seed2 v2 | qwen_seed3 v1 | qwen_seed3 v2 | mean v1 | mean v2 | mean change (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.53 | 75.13 | 74.00 | 74.37 | 75.82 | 76.27 | 74.78 | 75.26 | +0.48 |
| 20 | 73.62 | 73.69 | 74.83 | 74.91 | 77.26 | 77.33 | 75.23 | 75.31 | +0.08 |
| 40 | 74.68 | 74.68 | 74.00 | 74.00 | 75.74 | 75.74 | 74.80 | 74.80 | +0.00 |
| 60 | 77.86 | 77.86 | 77.41 | 77.41 | 75.28 | 75.28 | 76.85 | 76.85 | +0.00 |
| 80 | 79.30 | 79.30 | 79.23 | 79.23 | 76.35 | 76.35 | 78.29 | 78.29 | +0.00 |
| 100 | 79.76 | 79.76 | 78.17 | 78.17 | 79.45 | 79.45 | 79.13 | 79.13 | +0.00 |
| 120 | 80.74 | 80.74 | 77.18 | 77.18 | 74.83 | 74.83 | 77.58 | 77.58 | +0.00 |
| 140 | 81.12 | 81.12 | 78.01 | 78.01 | 77.71 | 77.71 | 78.95 | 78.95 | +0.00 |
| 160 | 82.26 | 82.26 | 77.41 | 77.41 | 78.09 | 78.09 | 79.25 | 79.25 | +0.00 |
| 180 | 82.18 | 82.18 | 80.44 | 80.44 | 79.98 | 79.98 | 80.87 | 80.87 | +0.00 |
| 200 | 81.65 | 81.65 | 79.98 | 79.98 | 79.61 | 79.61 | 80.41 | 80.41 | +0.00 |
| 220 | 81.96 | 81.96 | 79.45 | 79.45 | 79.76 | 79.76 | 80.39 | 80.39 | +0.00 |
| 240 | 81.88 | 81.88 | 79.08 | 79.08 | 80.97 | 80.97 | 80.64 | 80.64 | +0.00 |
| 250 | 82.26 | 82.26 | 79.61 | 79.61 | 79.83 | 79.83 | 80.57 | 80.57 | +0.00 |

v1 gain step 0 -> 250: qwen_seed1 +7.73, qwen_seed2 +5.61, qwen_seed3 +4.02; mean +5.79 pp

v2 gain step 0 -> 250: qwen_seed1 +7.13, qwen_seed2 +5.23, qwen_seed3 +3.56; mean +5.31 pp

answers credited only by v2: 22 of 55398 scored; answers v2 credits less than v1: 0; stored-v1 vs recomputed-v1 mismatches: 0
by whitelist step: exact_value 20, currency 2

### gemma_smoke: eval accuracy re-scored, v1 -> v2 (%), same outputs

| step | gemma_smoke v1 | gemma_smoke v2 | mean v1 | mean v2 | mean change (pp) |
|---:|---:|---:|---:|---:|---:|
| 0 | 44.81 | 80.59 | 44.81 | 80.59 | +35.78 |

v1 gain step 0 -> 0: gemma_smoke +0.00; mean +0.00 pp

v2 gain step 0 -> 0: gemma_smoke +0.00; mean +0.00 pp

answers credited only by v2: 472 of 1319 scored; answers v2 credits less than v1: 0; stored-v1 vs recomputed-v1 mismatches: 0
by whitelist step: unit 166, currency 164, currency+exact_value 88, currency+grouping 34, percent 16, grouping+unit 2, degree 2

### gemma_smoke_train: training batches re-scored (same completions)

| step | v1 acc % | v2 acc % | v1 mixed / all-correct / all-wrong groups | v2 mixed / all-correct / all-wrong groups |
|---:|---:|---:|---|---|
| 1 | 50.10 | 73.68 | 88 / 11 / 29 | 100 / 22 / 6 |
| 2 | 41.99 | 72.95 | 77 / 9 / 42 | 102 / 23 / 3 |
| 3 | 44.97 | 72.36 | 86 / 10 / 32 | 106 / 19 / 3 |

