| stage | A | B | seqs | tokens | mean abs | p50 | p99 | max | mean signed (A-B) | prob MAE | k3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noise | `bf16_eager_unpacked` | `fp32_eager_unpacked` | 16 | 26307 | 0.0071 | 0.0000 | 0.121 | 0.71 | -0.0002 | 0.0026 | 0.0003 |
| backend | `bf16_sdpa_unpacked` | `bf16_eager_unpacked` | 16 | 26307 | 0.0069 | 0.0000 | 0.122 | 0.86 | +0.0001 | 0.0025 | 0.0004 |
| backend | `bf16_flex_attention_unpacked` | `bf16_eager_unpacked` | 16 | 26307 | 0.0067 | 0.0000 | 0.121 | 0.74 | +0.0001 | 0.0025 | 0.0004 |
| packing | `bf16_eager_packedA` | `bf16_eager_unpacked` | 16 | 26307 | 0.0072 | 0.0000 | 0.125 | 0.74 | -0.0001 | 0.0026 | 0.0004 |
| packing | `bf16_eager_packedB` | `bf16_eager_unpacked` | 16 | 26307 | 0.0071 | 0.0000 | 0.125 | 0.70 | -0.0000 | 0.0027 | 0.0004 |
| packing | `bf16_sdpa_packedA` | `bf16_sdpa_unpacked` | 16 | 26307 | 0.0064 | 0.0000 | 0.119 | 0.56 | -0.0000 | 0.0024 | 0.0003 |
| packing | `bf16_sdpa_packedB` | `bf16_sdpa_unpacked` | 16 | 26307 | 0.0063 | 0.0000 | 0.117 | 0.83 | +0.0000 | 0.0023 | 0.0003 |
| packing | `bf16_flex_attention_packedA` | `bf16_flex_attention_unpacked` | 16 | 26307 | 0.0057 | 0.0000 | 0.115 | 0.83 | +0.0000 | 0.0021 | 0.0003 |
| packing | `bf16_flex_attention_packedB` | `bf16_flex_attention_unpacked` | 16 | 26307 | 0.0042 | 0.0000 | 0.102 | 0.48 | +0.0001 | 0.0016 | 0.0002 |
| prefill | `prefill` | `fp32_eager_unpacked` | 16 | 26307 | 0.0059 | 0.0000 | 0.103 | 0.41 | -0.0001 | 0.0022 | 0.0002 |
| prefill | `prefill` | `bf16_eager_unpacked` | 16 | 26307 | 0.0075 | 0.0000 | 0.128 | 0.62 | +0.0000 | 0.0028 | 0.0004 |
| prefill | `prefill` | `bf16_flex_attention_packedA` | 16 | 26307 | 0.0071 | 0.0000 | 0.124 | 1.01 | -0.0001 | 0.0026 | 0.0004 |
| online | `online_rollout` | `prefill` | 16 | 26307 | 0.0057 | 0.0000 | 0.110 | 0.66 | +0.0004 | 0.0021 | 0.0003 |
| online | `online_rollout` | `fp32_eager_unpacked` | 16 | 26307 | 0.0059 | 0.0000 | 0.100 | 0.47 | +0.0003 | 0.0022 | 0.0002 |
| online | `online_trainer` | `bf16_flex_attention_packedA` | 16 | 26307 | 1.2098 | 0.0005 | 12.906 | 24.07 | -1.1920 | 0.1936 | 1.0387 |
| online | `online_trainer` | `fp32_eager_unpacked` | 16 | 26307 | 1.2097 | 0.0009 | 12.894 | 24.07 | -1.1920 | 0.1936 | 1.0406 |
| online | `online_trainer_repeat` | `online_trainer` | 16 | 26307 | 0.0000 | 0.0000 | 0.000 | 0.00 | +0.0000 | 0.0000 | 0.0000 |
| online | `online_trainer` | `online_rollout` | 16 | 26307 | 1.2100 | 0.0006 | 12.906 | 24.07 | -1.1923 | 0.1937 | 1.0408 |
| online | `FULL_BATCH online_trainer` | `online_rollout` | 2048 | 3633450 | 1.1606 | 0.0038 | 11.834 | 37.75 | -1.1337 | 0.2010 | 0.9820 |

Full batch (3633450 response tokens), online trainer vs online rollout: n_tok=3.633e+06, mean_abs=1.161, p50_abs=0.003839, p99_abs=11.83, max_abs=37.75, mean_signed=-1.134, prob_mae=0.201, k3_mean=0.982
