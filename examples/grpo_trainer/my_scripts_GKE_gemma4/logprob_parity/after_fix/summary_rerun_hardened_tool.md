Fixture `/workspace/meta-RL/logprob_parity/fixture_fixed_1021_0813/fixture_step1.npz` (sha256 `284db69dddfb0cc864fb8c3c99b06e54c357e16d861f78765a1c5cde42f88b89`), config {"global_step": 1, "temperature": 1.0, "top_k": -1, "top_p": 1.0, "logprobs_mode": "processed_logprobs"}; all result files verified against this fixture and selection.

| stage | A | B | seqs | tokens | mean abs | p50 | p99 | max | mean signed (A-B) | prob MAE | k3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noise | `bf16_eager_unpacked` | `fp32_eager_unpacked` | 16 | 23564 | 0.0096 | 0.0000 | 0.130 | 0.70 | -0.0003 | 0.0035 | 0.0004 |
| backend | `bf16_sdpa_unpacked` | `bf16_eager_unpacked` | 16 | 23564 | 0.0092 | 0.0000 | 0.133 | 0.46 | +0.0001 | 0.0033 | 0.0005 |
| backend | `bf16_flex_attention_unpacked` | `bf16_eager_unpacked` | 16 | 23564 | 0.0090 | 0.0000 | 0.125 | 0.57 | +0.0001 | 0.0033 | 0.0004 |
| packing | `bf16_eager_packedA` | `bf16_eager_unpacked` | 16 | 23564 | 0.0094 | 0.0000 | 0.133 | 0.58 | -0.0001 | 0.0034 | 0.0005 |
| packing | `bf16_eager_packedB` | `bf16_eager_unpacked` | 16 | 23564 | 0.0094 | 0.0000 | 0.132 | 0.75 | +0.0003 | 0.0034 | 0.0005 |
| packing | `bf16_sdpa_packedA` | `bf16_sdpa_unpacked` | 16 | 23564 | 0.0086 | 0.0000 | 0.124 | 0.53 | -0.0001 | 0.0031 | 0.0004 |
| packing | `bf16_sdpa_packedB` | `bf16_sdpa_unpacked` | 16 | 23564 | 0.0084 | 0.0000 | 0.121 | 0.52 | -0.0000 | 0.0031 | 0.0004 |
| packing | `bf16_flex_attention_packedA` | `bf16_flex_attention_unpacked` | 16 | 23564 | 0.0084 | 0.0000 | 0.123 | 0.55 | +0.0000 | 0.0030 | 0.0004 |
| packing | `bf16_flex_attention_packedB` | `bf16_flex_attention_unpacked` | 16 | 23564 | 0.0086 | 0.0000 | 0.123 | 1.43 | -0.0002 | 0.0031 | 0.0004 |
| prefill | `prefill` | `fp32_eager_unpacked` | 16 | 23564 | 0.0083 | 0.0000 | 0.114 | 0.32 | +0.0001 | 0.0030 | 0.0003 |
| prefill | `prefill` | `bf16_eager_unpacked` | 16 | 23564 | 0.0097 | 0.0000 | 0.133 | 0.68 | +0.0004 | 0.0035 | 0.0005 |
| prefill | `prefill` | `bf16_flex_attention_packedA` | 16 | 23564 | 0.0093 | 0.0000 | 0.129 | 0.61 | +0.0002 | 0.0034 | 0.0005 |
| online | `online_rollout` | `prefill` | 16 | 23564 | 0.0077 | 0.0000 | 0.117 | 0.50 | +0.0003 | 0.0028 | 0.0003 |
| online | `online_rollout` | `fp32_eager_unpacked` | 16 | 23564 | 0.0084 | 0.0000 | 0.114 | 0.44 | +0.0004 | 0.0030 | 0.0003 |
| online | `online_trainer` | `bf16_flex_attention_packedA` | 16 | 23564 | 0.0085 | 0.0000 | 0.122 | 0.61 | +0.0000 | 0.0031 | 0.0004 |
| online | `online_trainer` | `fp32_eager_unpacked` | 16 | 23564 | 0.0093 | 0.0000 | 0.128 | 0.53 | -0.0001 | 0.0033 | 0.0004 |
| online | `online_trainer_repeat` | `online_trainer` | 16 | 23564 | 0.0000 | 0.0000 | 0.000 | 0.00 | +0.0000 | 0.0000 | 0.0000 |
| online | `online_trainer` | `online_rollout` | 16 | 23564 | 0.0095 | 0.0000 | 0.130 | 0.44 | -0.0005 | 0.0034 | 0.0005 |
| online | `FULL_BATCH online_trainer` | `online_rollout` | 2048 | 3639941 | 0.0093 | 0.0000 | 0.131 | 2.43 | -0.0005 | 0.0033 | 0.0005 |

Full batch (3639941 response tokens), online trainer vs online rollout: n_tok=3.64e+06, mean_abs=0.00926, p50_abs=8.348e-07, p99_abs=0.1308, max_abs=2.433, mean_signed=-0.0004788, prob_mae=0.003271, k3_mean=0.0004561
