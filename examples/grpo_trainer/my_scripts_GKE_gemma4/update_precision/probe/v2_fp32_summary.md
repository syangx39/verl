### Optimizer write-back probe: `fp32v2`

- Probe dir: `/workspace/meta-RL/logs/gemma4_optim_probe/gemma4_e2b_rl05_16n64g_optprobe_fp32v2_1022_1733` (probe v2); init records from 64/64 ranks (complete); optimizer ['AdamW'], torch ['2.13.0+cu130'].
- Optimizer-held parameter dtypes at construction: ['torch.float32']; total elements held over all ranks: 5,104,297,536.
- AdamW defaults: lr=1e-06, betas=[0.9, 0.999], eps=1e-08, weight_decay=0.01, foreach=None, fused=None.

| Call | Ranks | AdamW step t | lr | param / grad / exp_avg / exp_avg_sq dtype | elements | nonzero grad | changed | changed / nonzero grad | predicted to survive | ‖applied‖ / ‖ideal‖ | cos(applied, ideal) | ‖Δw‖ / ‖w‖ (applied) | ‖Δw‖ / ‖w‖ (ideal fp32) | post-clip grad norm |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64/64 | [1] | [1e-06] | torch.float32/torch.float32/torch.float32/torch.float32 | 4,634,769,472 | 50.48% | **42.382%** | 83.953% | 42.387% | **1.0000** | 1.0000 | 3.877e-06 | 3.877e-06 | 0.2527 |
| 1 | 64/64 | [2] | [1e-06] | torch.float32/torch.float32/torch.float32/torch.float32 | 4,634,769,472 | 50.48% | **42.759%** | 84.708% | 42.765% | **1.0000** | 1.0000 | 2.854e-06 | 2.854e-06 | 0.2794 |

- Call 0: measured changed set vs round-to-nearest prediction: 99.96% of changed elements were predicted; 99.95% of predicted elements changed. Mean |ideal| = 3.821e-07, mean |applied| = 3.818e-07; |w| < 2^-11 for 1.030% of elements.
- Call 1: measured changed set vs round-to-nearest prediction: 99.95% of changed elements were predicted; 99.94% of predicted elements changed. Mean |ideal| = 2.573e-07, mean |applied| = 2.570e-07; |w| < 2^-11 for 1.032% of elements.

Optimizer-only counterfactual on the same gradients (moments x write-back):

| Call | ideal: fp32 m,v + fp32 write-back | fb: fp32 m,v + param-dtype write-back (‖·‖/‖ideal‖, cos, changed) | bf: optimizer m,v + fp32 write-back (‖·‖/‖ideal‖, cos) | applied: optimizer m,v + param-dtype write-back (‖·‖/‖ideal‖, cos, changed) | cos(applied, fb) | bf16-visible change: this step / cumulative |
|---:|---|---|---|---|---:|---|
| 0 | 1, 1, — | 1.0000, 1.0000, 42.388% | 1.0000, 1.0000 | 1.0000, 1.0000, 42.382% | 1.0000 | 0.494% / 0.494% |
| 1 | 1, 1, — | 1.0000, 1.0000, 42.765% | 1.0000, 1.0000 | 1.0000, 1.0000, 42.759% | 1.0000 | 0.434% / 0.593% |

Per original-parameter category (summed over ranks):

| Call | Category | params | elements | changed | predicted | |w| < 2^-11 | ‖applied‖/‖ideal‖ | cos | ‖Δw‖/‖w‖ applied | ‖Δw‖/‖w‖ ideal | bf16-visible (step / cum) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | attention | 4 | 278,396,928 | 99.998% | 99.998% | 1.507% | 1.0000 | 1.0000 | 2.882e-05 | 2.882e-05 | 1.260% / 1.260% |
| 0 | embedding | 3 | 406,192,128 | 8.914% | 8.961% | 1.343% | 1.0000 | 1.0000 | 5.328e-06 | 5.328e-06 | 0.034% / 0.034% |
| 0 | mlp | 3 | 1,557,135,360 | 99.998% | 99.998% | 1.414% | 1.0000 | 1.0000 | 2.978e-05 | 2.978e-05 | 1.165% / 1.165% |
| 0 | norm | 11 | 286,112 | 95.005% | 95.550% | 1.867% | 0.9737 | 0.9630 | 5.090e-08 | 5.228e-08 | 1.806% / 1.806% |
| 0 | other | 5 | 2,660,992 | 0.000% | 0.000% | 1.571% | 0.0000 | nan | 0.000e+00 | 1.000e-08 | 0.000% / 0.000% |
| 0 | per-layer embedding / projection | 4 | 2,390,097,920 | 3.864% | 3.865% | 0.671% | 1.0000 | 1.0000 | 2.096e-06 | 2.096e-06 | 0.047% / 0.047% |
| 1 | attention | 4 | 278,396,928 | 99.934% | 99.934% | 1.512% | 1.0000 | 1.0000 | 2.109e-05 | 2.109e-05 | 1.107% / 1.508% |
| 1 | embedding | 3 | 406,192,128 | 9.933% | 9.981% | 1.343% | 1.0000 | 1.0000 | 4.261e-06 | 4.261e-06 | 0.038% / 0.049% |
| 1 | mlp | 3 | 1,557,135,360 | 99.935% | 99.935% | 1.418% | 1.0000 | 1.0000 | 2.190e-05 | 2.190e-05 | 1.021% / 1.397% |
| 1 | norm | 11 | 286,112 | 89.998% | 90.412% | 1.869% | 0.9597 | 0.9420 | 3.702e-08 | 3.858e-08 | 0.953% / 1.671% |
| 1 | other | 5 | 2,660,992 | 0.000% | 0.000% | 1.571% | 0.0000 | nan | 0.000e+00 | 1.000e-08 | 0.000% / 0.000% |
| 1 | per-layer embedding / projection | 4 | 2,390,097,920 | 4.471% | 4.473% | 0.672% | 1.0000 | 1.0000 | 1.621e-06 | 1.621e-06 | 0.041% / 0.056% |
