### Optimizer write-back probe: `bf16v2`

- Probe dir: `/workspace/meta-RL/logs/gemma4_optim_probe/gemma4_e2b_rl05_16n64g_optprobe_bf16v2_1022_1720` (probe v2); init records from 64/64 ranks (complete); optimizer ['AdamW'], torch ['2.13.0+cu130'].
- Optimizer-held parameter dtypes at construction: ['torch.bfloat16']; total elements held over all ranks: 5,104,297,536.
- AdamW defaults: lr=1e-06, betas=[0.9, 0.999], eps=1e-08, weight_decay=0.01, foreach=None, fused=None.

| Call | Ranks | AdamW step t | lr | param / grad / exp_avg / exp_avg_sq dtype | elements | nonzero grad | changed | changed / nonzero grad | predicted to survive | ‖applied‖ / ‖ideal‖ | cos(applied, ideal) | ‖Δw‖ / ‖w‖ (applied) | ‖Δw‖ / ‖w‖ (ideal fp32) | post-clip grad norm |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64/64 | [1] | [1e-06] | torch.bfloat16/torch.bfloat16/torch.bfloat16/torch.bfloat16 | 4,634,769,472 | 50.44% | **0.498%** | 0.987% | 0.497% | **0.1660** | 0.1069 | 6.455e-07 | 3.888e-06 | 0.8144 |
| 1 | 64/64 | [2] | [1e-06] | torch.bfloat16/torch.bfloat16/torch.bfloat16/torch.bfloat16 | 4,634,769,472 | 50.42% | **0.266%** | 0.528% | 0.265% | **0.1165** | 0.0878 | 3.250e-07 | 2.790e-06 | 1.0016 |

- Call 0: measured changed set vs round-to-nearest prediction: 99.48% of changed elements were predicted; 99.62% of predicted elements changed. Mean |ideal| = 3.833e-07, mean |applied| = 6.617e-09; |w| < 2^-11 for 1.030% of elements.
- Call 1: measured changed set vs round-to-nearest prediction: 99.36% of changed elements were predicted; 99.68% of predicted elements changed. Mean |ideal| = 2.536e-07, mean |applied| = 2.377e-09; |w| < 2^-11 for 1.031% of elements.

Optimizer-only counterfactual on the same gradients (moments x write-back):

| Call | ideal: fp32 m,v + fp32 write-back | fb: fp32 m,v + param-dtype write-back (‖·‖/‖ideal‖, cos, changed) | bf: optimizer m,v + fp32 write-back (‖·‖/‖ideal‖, cos) | applied: optimizer m,v + param-dtype write-back (‖·‖/‖ideal‖, cos, changed) | cos(applied, fb) | bf16-visible change: this step / cumulative |
|---:|---|---|---|---|---:|---|
| 0 | 1, 1, — | 0.1663, 0.1070, 0.499% | 1.0001, 1.0000 | 0.1660, 0.1069, 0.498% | 0.9947 | 0.498% / 0.498% |
| 1 | 1, 1, — | 0.1163, 0.0878, 0.266% | 0.9998, 1.0000 | 0.1165, 0.0878, 0.266% | 0.9876 | 0.266% / 0.493% |

Per original-parameter category (summed over ranks):

| Call | Category | params | elements | changed | predicted | |w| < 2^-11 | ‖applied‖/‖ideal‖ | cos | ‖Δw‖/‖w‖ applied | ‖Δw‖/‖w‖ ideal | bf16-visible (step / cum) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | attention | 4 | 278,396,928 | 1.269% | 1.267% | 1.507% | 0.1698 | 0.1091 | 4.908e-06 | 2.890e-05 | 1.269% / 1.269% |
| 0 | embedding | 3 | 406,192,128 | 0.034% | 0.034% | 1.343% | 0.1426 | 0.0945 | 7.568e-07 | 5.309e-06 | 0.034% / 0.034% |
| 0 | mlp | 3 | 1,557,135,360 | 1.173% | 1.172% | 1.414% | 0.1633 | 0.1051 | 4.876e-06 | 2.986e-05 | 1.173% / 1.173% |
| 0 | norm | 11 | 286,112 | 1.816% | 1.815% | 1.867% | 0.2421 | 0.1344 | 1.269e-08 | 5.241e-08 | 1.816% / 1.816% |
| 0 | other | 5 | 2,660,992 | 0.000% | 0.000% | 1.571% | 0.0000 | nan | 0.000e+00 | 1.000e-08 | 0.000% / 0.000% |
| 0 | per-layer embedding / projection | 4 | 2,390,097,920 | 0.048% | 0.048% | 0.671% | 0.2228 | 0.1455 | 4.745e-07 | 2.130e-06 | 0.048% / 0.048% |
| 1 | attention | 4 | 278,396,928 | 0.678% | 0.676% | 1.508% | 0.1208 | 0.0897 | 2.514e-06 | 2.080e-05 | 0.678% / 1.253% |
| 1 | embedding | 3 | 406,192,128 | 0.024% | 0.024% | 1.343% | 0.1025 | 0.0779 | 4.110e-07 | 4.009e-06 | 0.024% / 0.037% |
| 1 | mlp | 3 | 1,557,135,360 | 0.621% | 0.619% | 1.414% | 0.1135 | 0.0860 | 2.427e-06 | 2.138e-05 | 0.621% / 1.160% |
| 1 | norm | 11 | 286,112 | 0.576% | 0.574% | 1.868% | 0.1499 | 0.0898 | 5.774e-09 | 3.852e-08 | 0.576% / 1.786% |
| 1 | other | 5 | 2,660,992 | 0.000% | 0.000% | 1.571% | 0.0000 | nan | 0.000e+00 | 1.000e-08 | 0.000% / 0.000% |
| 1 | per-layer embedding / projection | 4 | 2,390,097,920 | 0.028% | 0.028% | 0.672% | 0.1676 | 0.1227 | 2.671e-07 | 1.594e-06 | 0.028% / 0.048% |
