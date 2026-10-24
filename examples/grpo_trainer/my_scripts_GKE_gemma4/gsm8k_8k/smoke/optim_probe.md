### Optimizer write-back probe: `gsm8k_smoke`

- Probe dir: `/workspace/meta-RL/logs/gemma4_optim_probe/gemma4_e2b_gsm8k8k_smoke_16n64g_s250stop3_seed1_1024_0036` (probe v2); init records from 64/64 ranks (complete); optimizer ['AdamW'], torch ['2.13.0+cu130'].
- Optimizer-held parameter dtypes at construction: ['torch.float32']; total elements held over all ranks: 5,104,297,536.
- AdamW defaults: lr=2e-06, betas=[0.9, 0.999], eps=1e-08, weight_decay=0.0, foreach=None, fused=None.

| Call | Ranks | AdamW step t | lr | param / grad / exp_avg / exp_avg_sq dtype | elements | nonzero grad | changed | changed / nonzero grad | predicted to survive | ‖applied‖ / ‖ideal‖ | cos(applied, ideal) | ‖Δw‖ / ‖w‖ (applied) | ‖Δw‖ / ‖w‖ (ideal fp32) | post-clip grad norm |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64/64 | [1] | [0.0] | torch.float32/torch.float32/torch.float32/torch.float32 | 4,634,769,472 | 50.21% | **0.000%** | 0.000% | 0.000% | **nan** | nan | 0.000e+00 | 0.000e+00 | 1.0000 |
| 1 | 64/64 | [2] | [2e-07] | torch.float32/torch.float32/torch.float32/torch.float32 | 4,634,769,472 | 50.22% | **42.514%** | 84.663% | 42.513% | **1.0001** | 1.0000 | 6.748e-07 | 6.747e-07 | 0.9198 |
| 2 | 64/64 | [3] | [4e-07] | torch.float32/torch.float32/torch.float32/torch.float32 | 4,634,769,472 | 50.25% | **43.002%** | 85.568% | 43.001% | **1.0000** | 1.0000 | 1.231e-06 | 1.231e-06 | 1.0000 |

- Call 0: measured changed set vs round-to-nearest prediction: nan% of changed elements were predicted; nan% of predicted elements changed. Mean |ideal| = 0.000e+00, mean |applied| = 0.000e+00; |w| < 2^-11 for 1.030% of elements.
- Call 1: measured changed set vs round-to-nearest prediction: 100.00% of changed elements were predicted; 100.00% of predicted elements changed. Mean |ideal| = 6.320e-08, mean |applied| = 6.320e-08; |w| < 2^-11 for 1.030% of elements.
- Call 2: measured changed set vs round-to-nearest prediction: 100.00% of changed elements were predicted; 100.00% of predicted elements changed. Mean |ideal| = 1.127e-07, mean |applied| = 1.127e-07; |w| < 2^-11 for 1.032% of elements.

Optimizer-only counterfactual on the same gradients (moments x write-back):

| Call | ideal: fp32 m,v + fp32 write-back | fb: fp32 m,v + param-dtype write-back (‖·‖/‖ideal‖, cos, changed) | bf: optimizer m,v + fp32 write-back (‖·‖/‖ideal‖, cos) | applied: optimizer m,v + param-dtype write-back (‖·‖/‖ideal‖, cos, changed) | cos(applied, fb) | bf16-visible change: this step / cumulative |
|---:|---|---|---|---|---:|---|
| 0 | 1, 1, — | nan, nan, 0.000% | nan, nan | nan, nan, 0.000% | nan | 0.000% / 0.000% |
| 1 | 1, 1, — | 1.0001, 1.0000, 42.514% | 1.0000, 1.0000 | 1.0001, 1.0000, 42.514% | 1.0000 | 0.064% / 0.064% |
| 2 | 1, 1, — | 1.0000, 1.0000, 43.002% | 1.0000, 1.0000 | 1.0000, 1.0000, 43.002% | 1.0000 | 0.202% / 0.205% |

Per original-parameter category (summed over ranks):

| Call | Category | params | elements | changed | predicted | |w| < 2^-11 | ‖applied‖/‖ideal‖ | cos | ‖Δw‖/‖w‖ applied | ‖Δw‖/‖w‖ ideal | bf16-visible (step / cum) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | attention | 4 | 278,396,928 | 0.000% | 0.000% | 1.507% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 0 | embedding | 3 | 406,192,128 | 0.000% | 0.000% | 1.343% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 0 | mlp | 3 | 1,557,135,360 | 0.000% | 0.000% | 1.414% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 0 | norm | 11 | 286,112 | 0.000% | 0.000% | 1.867% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 0 | other | 5 | 2,660,992 | 0.000% | 0.000% | 1.571% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 0 | per-layer embedding / projection | 4 | 2,390,097,920 | 0.000% | 0.000% | 0.671% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 1 | attention | 4 | 278,396,928 | 99.816% | 99.816% | 1.507% | 1.0001 | 1.0000 | 4.931e-06 | 4.930e-06 | 0.158% / 0.158% |
| 1 | embedding | 3 | 406,192,128 | 10.569% | 10.563% | 1.343% | 1.0002 | 1.0000 | 8.677e-07 | 8.675e-07 | 0.006% / 0.006% |
| 1 | mlp | 3 | 1,557,135,360 | 99.826% | 99.825% | 1.414% | 1.0001 | 1.0000 | 5.190e-06 | 5.190e-06 | 0.151% / 0.151% |
| 1 | norm | 11 | 286,112 | 77.750% | 77.712% | 1.867% | 0.9923 | 0.9090 | 8.929e-09 | 8.999e-09 | 0.134% / 0.134% |
| 1 | other | 5 | 2,660,992 | 0.000% | 0.000% | 1.571% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 1 | per-layer embedding / projection | 4 | 2,390,097,920 | 3.973% | 3.972% | 0.671% | 1.0001 | 0.9999 | 3.909e-07 | 3.909e-07 | 0.007% / 0.007% |
| 2 | attention | 4 | 278,396,928 | 99.864% | 99.864% | 1.512% | 1.0000 | 1.0000 | 8.940e-06 | 8.940e-06 | 0.489% / 0.497% |
| 2 | embedding | 3 | 406,192,128 | 12.678% | 12.673% | 1.344% | 1.0000 | 1.0000 | 1.525e-06 | 1.525e-06 | 0.014% / 0.015% |
| 2 | mlp | 3 | 1,557,135,360 | 99.871% | 99.871% | 1.418% | 1.0000 | 1.0000 | 9.481e-06 | 9.481e-06 | 0.480% / 0.487% |
| 2 | norm | 11 | 286,112 | 84.706% | 84.692% | 1.870% | 0.9965 | 0.9490 | 1.663e-08 | 1.669e-08 | 0.210% / 0.214% |
| 2 | other | 5 | 2,660,992 | 0.000% | 0.000% | 1.571% | nan | nan | 0.000e+00 | 0.000e+00 | 0.000% / 0.000% |
| 2 | per-layer embedding / projection | 4 | 2,390,097,920 | 4.525% | 4.525% | 0.672% | 1.0000 | 1.0000 | 7.080e-07 | 7.080e-07 | 0.020% / 0.020% |
