# Qwen3 TPU/GPU RL parity rulebook — GSM8K, 8K response cap (recipe `gsm8k_8k_v1`)

This rulebook defines the TPU workload corresponding to the GB200 recipe validated by three 250-step runs on GSM8K. It replaces the OpenMathInstruct-2 `stab_kl0` round for the Meta Ads engagement: the customer asked for GSM8K, evaluation on the full 1,319-question test set, an 8K response budget matching their production workload, and a recipe that is efficient, stable and reproducible on both accelerators. The comparison is colocated, synchronous, on-policy GRPO on the post-trained Qwen3-0.6B checkpoint. It separates the training semantics both sides must preserve (Tier 1), the execution model (Tier 2), and the implementation choices each platform may optimize (Tier 3).

Where this recipe departs from the Meta GPU reference (`qwen3-0p6b-base-boxed-cliphigh`) and from the Meta TPU arm (`a26d`), the departure is deliberate and is listed in the section "Provenance".

## Rulebook anchor

Both sides implement the same versioned recipe and evaluation protocol. The anchor is the GB200 `gsm8k_8k_v1` recipe (three seeds, 250 steps, 64 GB200). A change to a frozen invariant creates a new recipe version and requires a matching reference run; a TPU-specific variant cannot be compared with the existing GPU band as though the workload were unchanged.

A secondary reference, `gsm8k_2k_v1` (response cap 2,048 with the overlong penalty, two seeds), is delivered for context only; it is not part of this round's acceptance and is described in the appendix.

## Tier 1 — Frozen invariants

Resolved run configuration and the executed code take precedence over framework defaults, comments, or preset names.

| Invariant | Frozen value or requirement |
|---|---|
| Initial model | `Qwen/Qwen3-0.6B` — the **post-trained** checkpoint, not `-Base`. Fingerprint: 311 tensors, untied `lm_head`, `sum(model.norm.weight) = 3932.626953125`; `generation_config.eos_token_id = [151645, 151643]`. Freeze the exact weight file, model config, tokenizer, chat template (`package/model/`, `MODEL_SHA256`). Every seed starts from that checkpoint. |
| Training data | GSM8K train, 7,473 rows, in Meta's boxed format (`data/gsm8k_boxed_train.parquet`, built from `data/train.jsonl`). Zero prompt-text overlap with the test set (`DATA_COUNTS.json`). |
| Data order | TPU seed *k* reads `data/train_order_seed{k}.parquet` (the 32,000 (step, question) rows GPU seed *k* consumed: 250 steps × 128 prompts, order within a step irrelevant) sequentially with shuffling OFF. Only if the TPU pipeline cannot honor a fixed order: shuffle natively, log per-step question IDs and state the deviation. Rollout sampling remains stochastic in either case. |
| Prompt | System + user messages exactly as in `data/gsm8k_boxed_train.parquet` (Meta's system prompt; user = question + boxed-answer instruction), rendered by the Qwen3 chat template with `add_generation_prompt=True` and **`enable_thinking=True` (the template default: no `<think>` tokens are injected; the policy decides whether to think)**. Rendered text ends with `<|im_start|>assistant\n`. Token IDs frozen by `fixtures/prompt_fixture.json`. Prompt limit 512 tokens (no training prompt exceeds it). |
| Evaluation data | `data/gsm8k_boxed_test.parquet`: all 1,319 GSM8K test questions (`extra_info.index` ≥ 10,000,000). The first-512 subset used by Meta's GPU team is not evaluated in this round. |
| Step accounting | 128 prompts × 16 generations = 2,048 completions per step. Exactly one optimizer update over the full batch per rollout (`mini = batch`, one PPO epoch, μ = 1). 250 steps ≈ 4.3 passes over the 7,473 training questions. |
| Optimizer | AdamW; peak LR **2e-6**; linear warmup over 10 steps (update 1 at LR 0, update 2 at 2e-7), then cosine decay to 0 at step 250; betas (0.9, 0.999); eps 1e-8; weight decay 0.0; global gradient-norm clipping at 1.0 after accumulation. |
| Advantage | GRPO: A = (R − mean_group) / (std_group + 1e-6), **sample std (correction = 1)**, groups of 16 by rollout UID; broadcast to every response token. Groups with equal rewards have zero advantage but stay in the batch and in the loss denominator. No dynamic sampling / group filtering. |
| Policy loss | **Single-forward REINFORCE with a detached truncated IS weight**: loss = token-mean over all valid response tokens of −A·w·log π_θ, with w = min(exp(clamp(log π_θ.detach() − log π_sampler, −20, 20)), **3.0**) computed under no-grad from the training pass. The PPO ratio is therefore exactly 1; clip (0.2 / 0.28) and dual-clip are inert. Gradient = −A·w·∇log π_θ. No lower truncation, no batch normalization, no rejection sampling. |
| Loss denominator | **Global token mean**: sum of per-token losses over the whole 2,048-completion batch divided by the total number of valid response tokens in the batch (terminating token included, prompt tokens excluded). Micro-batch accumulation must reproduce this global mean. |
| Reference KL / entropy | β = 0: no reference model, no KL loss, no KL reward shaping. Entropy coefficient 0. |
| Training sampling | Temperature 1.0; top_k disabled (−1); top_p 1.0; no repetition/presence penalties; stop tokens [151645, 151643]; `ignore_eos=False`. |
| Length limits | Prompt 512 tokens; **response 8,192 generated tokens**. A response that reaches 8,192 tokens is kept with all its tokens in the loss (no truncation masking, no resampling). |
| Training reward | `boxed_math` (shared `code/boxed_math_reward.py`): last `\boxed{…}`, brace-balanced, minimal normalization (strip spaces, trailing period, commas, `$`); exact string match → 1.0; parseable box but wrong → 0.1; no parseable box (including a truncated, unbalanced box) → 0.0. **No overlong / length penalty** (`REWARD_PENALTY_SOURCES` empty): training score = raw reward. |
| Evaluation | Greedy (`do_sample=False`, temperature 0, top_p 1, top_k −1), n = 1, response cap 8,192, at steps 0, 20, 40, …, 240, 250 (14 evaluations). Metric: `acc` = (raw reward == 1.0). Evaluate the initial checkpoint and all scheduled checkpoints; no best-checkpoint selection. |
| Precision | FP32 master weights and FP32 Adam moments; BF16 matrix compute; BF16 rollout weights and KV cache; no weight or KV quantization. Log-probabilities computed in FP32 from the logits. |
| Run | 250 optimizer steps per seed; seeds 1, 2, 3. |

### Loss and gradient accounting (what "the same update" means)

*Old-policy anchor.* The single-forward path does not compute a separate old-policy log-probability: with one update per batch the pre-update policy is the current policy, so the ratio exp(log π − log π.detach()) is exactly 1. The reviewed GPU implementation (`bypass_mode=True`, `loss_type=reinforce`) was validated against the two-pass implementation on an identical injected batch: step-1 gradient cosine 0.99995 (relative error 0.98 %), post-update Δθ cosine 0.99972 — i.e. equal up to BF16 pass-to-pass noise.

*Importance weight.* w uses the sampler's log-probability of the sampled token under the **same policy version** (weights synchronized after the previous update), the same temperature and probability normalization, and the training-pass log-probability of the same token, detached. On GB200 the weight is close to 1 (per-token |Δlogp| p50 ≈ 0.00, p99 ≈ 0.15, max ≈ 1.4 nats at step 1; effective sample fraction ≈ 0.998; fraction of tokens whose pre-truncation ratio exp(log π_θ − log π_sampler) exceeds 3 ≈ 0). It exists as a guard for sampler/trainer mismatch, which may be larger on TPU. **The TPU implementation must use the same TIS form; substituting the "sampler-as-denominator PPO" form used by Meta's TPU arm is not equivalent** (on an identical batch the two give gradient cosine 0.9897 / relative error 14.3 %, and over 250 steps the PPO-IS form learned slower and lower — see appendix).

*Denominator.* A global token mean is not equivalent to averaging per-sequence means (Meta TPU's `seq-mean-token-mean`) or to equally averaging unequal micro-batch means. Reduction order may cause floating-point differences; bitwise-identical gradients are not required.

*Numerical policy.* FP32 gradient accumulation and reduction (the FSDP default in the reviewed configuration), FP32 master weights and Adam state. Record the actual dtypes on the TPU side. The GB200 Adam update was verified to reproduce the saved weights from the saved moments to 1 FP32 ulp.

## Tier 2 — Execution model

Current track: colocated, synchronous, on-policy. The reference uses **16 nodes × 4 = 64 GB200** (data-parallel 64; 32 completions per GPU, micro-batch 8, 4 accumulation micro-steps; vLLM TP = 1, one instance per GPU); the corresponding TPU point uses 64 v7x chips. Rollout and training share the declared accelerator allocation. Other accelerator counts are separate, explicitly labeled scaling points.

For step *t*, generate the fixed 128 × 16 batch from policy version *t*, accumulate the full batch's contribution, perform one optimizer update, and synchronize rollout weights before admitting the next step's prompts. No cross-step rollout lookahead, policy lag, replay, or reuse of prior-step samples. Asynchronous requests, continuous batching and concurrent reward scoring within the step are allowed; batch membership and complete UID groups must be preserved.

The single-forward path is an implementation choice within this model (it removes one trainer forward per step), not a change of algorithm; the two-pass path is acceptable on TPU if the same loss is produced.

## Tier 3 — Free implementation variables

Each side may use its native best implementation within Tiers 1 and 2 and must record the actual settings and software versions with each result. Framework and runtime (verl 0.8-based fork / vLLM 0.20.2rc1 on GPU; MaxText/Tunix on TPU), attention kernels, fusion, graph capture, rematerialization; sharding and tensor parallelism on the declared chip count; micro-batching, padding removal and sequence packing (subject to the loss contract; packing must preserve positions and block cross-sequence attention); memory budgeting (GPU: vLLM `gpu_memory_utilization` 0.30, `max_num_batched_tokens` 8192, `max_num_seqs` 1024), prefix caching, scheduling, host concurrency. These describe the reference implementation, not requirements for TPU.

## Alignment sequence

Work through the gates in order; each is cheap relative to a 250-step run and localizes a mismatch to one layer.

1. **Prompt fixture.** Render `fixtures/prompt_fixture.json` on the TPU stack. For every row the full token-ID sequence must be identical to the GPU rendering, including the framing tokens (leading `151644, 8948, 198` for the system turn; trailing assistant prefix `151644, 77091, 198`) and **no `<think>` tokens**. Check complete IDs, not lengths.
2. **Scorer fixture.** Run `code/boxed_math_reward.py` (or a verbatim port) on `fixtures/meta_reward_fixtures.json` (Meta's 21 rules, validated at cap 2048 with the penalty on) and on `fixtures/scorer_fixture_8k.jsonl` (800 real completions from steps 0 and 250, expected `acc/fmt/score` under the 8K no-penalty rule). Row-by-row equality is required. The scorer is pure string logic (no math-verify, no subprocess): no timeout branch exists in this round.
3. **Step-0 evaluation.** Greedy evaluation of the untouched checkpoint on the 1,319 questions. GB200 (three runs, same weights): **0.7453 / 0.7400 / 0.7582, mean 0.7478**; format rate ≈ 0.92–0.93. Report the TPU value and its per-question agreement with the GPU outputs. Both are diagnostic references, not criteria (see "Evaluation variability"): a value inside [0.728, 0.768] (GPU mean ± 2.0 pp) is the suggested reading of "consistent"; a value outside it is a reason to check the template, cap, stop tokens and tokenizer before proceeding, not a failure.
4. **Trainer-vs-sampler numerics.** On `fixtures/logprob_fixture_8k.json` (96 sequences from step 1 of the fixture job, with `prompt_ids`, `response_ids`, `response_mask`, `position_ids_response`, `logp_sampler`, `logp_trainer`): TPU trainer log-probabilities on the same tokens compared with `logp_trainer`; and, on TPU's own rollouts, sampler-vs-trainer statistics computed the same way. GB200 values are recorded in the fixture's header and in `band/summary.json`. The three reference runs use the single-forward trainer, which logs the sampler-vs-trainer statistics as `actor/rollout_corr/k3_kl` (non-negative estimator), `actor/rollout_corr/kl` (signed) and `actor/rollout_corr/log_ppl_abs_diff`; their 250-step means are `rollout_corr_k3_kl`, `rollout_corr_kl` and `rollout_corr_log_ppl_abs_diff` in `band/summary.json`: k3_kl 0.00066 / 0.00069 / 0.00067 (seeds 1–3; flat across the 250 steps, first-5 and last-5 means both ≈ 0.0007), signed kl equal to k3_kl at this precision, log_ppl_abs_diff 0.00091 / 0.00098 / 0.00091. The probability-space MAE `training/rollout_probs_diff_mean` comes from the old-log-prob pass and is therefore not recorded for these runs (it was ≈ 0.005 on the two-pass runs of the 2K recipe). On the 8K fixture the independent PyTorch/HF reference (sdpa attention, bf16 autocast) vs the FSDP trainer's pre-update log-probabilities gave mean |Δlogp| 0.0157 nats, p99 0.136, max 1.25 over 2,830,118 valid tokens (`fixtures/replay_step1_reference_8k.log`). Investigation trigger: a non-negative error metric (probability MAE, mean |Δlogp|, tail percentiles) an order of magnitude above the GB200 value. Sampler and trainer must use the same checkpoint, trainer log-probabilities before any update.
5. **Single-step replay (recommended).** From `fixtures/fixture_step1.npz` (the full step-1 batch: token IDs, masks, sampler log-probs, rewards, advantages): (a) recompute advantages on TPU and compare (max |Δ| on GB200 vs an independent PyTorch reference: 4e-7); (b) inject the identical batch and compare per-token log-probabilities, gradient direction and norm (the reference replay uses the ratio form −A·w·exp(logπ − logπ.detach()), whose gradient equals the REINFORCE form's; loss scalars differ by construction and are not compared) with `fixtures/replay_step1_reference_8k.json`; (c) optimizer: replay both fixture steps in order from θ₀ — update 1 on `fixture_step1.npz` at LR 0 (weights unchanged, Adam moments initialised), then update 2 on `fixture_step2.npz` at LR 2e-7 — and compare θ₂ with `checkpoints/fixture_seed1_step2/` (GB200 Adam application vs the saved moments: 1 FP32 ulp). GB200-vs-reference on the shipped 8K fixture (`fixtures/grad_compare_8k.log`, `fixtures/replay_delta_8k.log`): (b) pre-clip gradient norm 0.1306 (trainer) vs 0.1315 (reference), 0.7 % apart; gradient direction cosine **0.9825**, relative error **18.8 %**, per layer type embed 0.992 / mlp 0.976 (mean) / attention 0.962 (mean, min 0.66); (c) two-step Δθ cosine **0.948**, relative error **32 %**. These are measured references for one GPU-internal comparison (FSDP trainer vs an independent PyTorch/HF implementation on the same batch); they are not acceptance thresholds for a TPU gradient or Δθ, and the gate-4 investigation trigger is not applied here. The per-tensor values are retained in `grad_compare_8k.json` for diagnosis: the small norm weights (`q_norm` / `k_norm`, head_dim 128; `input_layernorm`, hidden 1024) show the largest deviations (one cosine is negative on GB200 itself); BF16 numerics, kernel differences and the micro-batch split (reference 1 vs trainer 8) may each contribute, and the cause has not been isolated. For comparison, the same model on the 2K-cap fixture gave 0.9905 / 13.7 % and Δθ 0.969; the 8K step-1 batch has 1.39× the valid response tokens (2,830,118 vs 2,032,090) and a smaller gradient (norm 0.13 vs 0.19). The per-parameter reference gradient is shipped as `fixtures/replay_grad_step1/grad_step1.safetensors` (compare with `code/compare_grads.py`).
6. **Training comparison.** Three TPU seeds through 250 steps with the frozen evaluation schedule, reported against the GB200 band below.

## Quality and performance reporting

Overlay all GPU and TPU seed curves through 250 steps; report step 0, the final value and the gain separately. Because TPU seed *k* uses GPU seed *k*'s data order, also report each TPU seed against its paired GPU seed; the acceptance rule is still applied against the band.

### Predeclared reference (three GB200 seeds, `gsm8k_8k_v1`)

GSM8K test accuracy (1,319 questions, greedy):

| step | seed 1 | seed 2 | seed 3 | min–max width |
|---:|---:|---:|---:|---:|
| 0 | 0.7453 | 0.7400 | 0.7582 | 1.8 pp |
| 20 | 0.7362 | 0.7483 | 0.7726 | 3.6 |
| 40 | 0.7468 | 0.7400 | 0.7574 | 1.7 |
| 60 | 0.7786 | 0.7741 | 0.7528 | 2.6 |
| 80 | 0.7930 | 0.7923 | 0.7635 | 3.0 |
| 100 | 0.7976 | 0.7817 | 0.7945 | 1.6 |
| 120 | 0.8074 | 0.7718 | 0.7483 | 5.9 |
| 140 | 0.8112 | 0.7801 | 0.7771 | 3.4 |
| 160 | 0.8226 | 0.7741 | 0.7809 | 4.9 |
| 180 | 0.8218 | 0.8044 | 0.7998 | 2.2 |
| 200 | 0.8165 | 0.7998 | 0.7961 | 2.0 |
| 220 | 0.8196 | 0.7945 | 0.7976 | 2.5 |
| 240 | 0.8188 | 0.7908 | 0.8097 | 2.8 |
| **250** | **0.8226** | **0.7961** | **0.7983** | 2.7 |

Band width across the 14 checkpoints: median 2.6 pp, p90 4.5 pp, max 5.9 pp. Step 250: **mean 0.8057, range [0.7961, 0.8226]**. Gain 0 → 250 per seed: +7.7 / +5.6 / +4.0 pp. Seed 1 is the highest of three ordinary draws; it is retained with equal weight.

Training diagnostics (means over the last 10 steps; full per-step values in `band/summary.json` and the TensorBoard files): trainer entropy 0.50 / 0.52 / 0.54 nats per token; mean response length 2,364 / 2,312 / 2,612 tokens (from ≈ 1,350 at step 1, rising to ≈ 2,300 by step 50 and drifting slowly upward; no length pressure exists in this recipe; 250-step mean 2,303, seed-to-seed spread 427); cap-hit fraction ≈ 0.01–0.02 throughout; pre-clip gradient norm 250-step mean 0.090 (spread 0.013); training score (raw reward, T = 1) 0.90 / 0.91 / 0.90; steady-state step time on 64 GB200 (rulebook window: steps 20–250 excluding evaluation and checkpoint steps, 216 steps per seed): median **39.5 / 39.4 / 40.7 s** (`band/summary.json`).

Time-to-quality (end-to-end from launch, including startup and evaluations; 64 GPUs):

| threshold (two consecutive evaluations) | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| ≥ 0.78 | step 100, 75.6 min, 80.6 GPU-h | step 100, 73.7 min, 78.6 GPU-h | step 180, 133.8 min, 142.7 GPU-h |
| ≥ 0.80 | step 140, 102.9 min, 109.7 GPU-h | never (max 0.804) | never (max 0.810) |

**0.80 is not an acceptance threshold for this round**: one of three GB200 seeds sustains it. Report time-to-0.78 as the efficiency headline and time-to-0.80 as secondary.

### Evaluation variability (measured, not assumed)

Greedy evaluation of the *same* checkpoint on the same 1,319 questions is not reproducible at the output level on the GPU stack (vLLM 0.20.2rc1, 64 instances): across the three step-0 evaluations only 215–241 of 1,319 outputs were identical per pair (80 across all three), the first divergence occurs at a median of ≈ 500 characters, and 119–149 questions per pair flip between correct and incorrect (204 questions unstable across the three). Inputs, gold answers, weight file and the resolved generation configuration were verified identical across the three runs (`band/step0_greedy_variability.json`, `env/CONFIG_DIFF.txt`); the cause of the remaining divergence has not been established (vLLM does not guarantee reproducible outputs by default; batch-dependent numerics are a candidate, not a finding). The net effect on the mean is ±1–2 pp per evaluation (σ ≈ 0.9 pp). Consequences: (i) per-question output agreement between GPU and TPU is reported but never a criterion; (ii) single checkpoints are not compared point-wise; (iii) the 2K secondary band (0.5 pp median width) is at this noise floor, the 8K band (2.6 pp) is training variance on top of it.

### Comparison rule for this round

The goal of this round is to reproduce the GB200 training behaviour and its spread, not to clear a threshold. The three-seed min–max range is a descriptive empirical range, not a confidence interval and not an automatic acceptance envelope. The TPU result is reported as three complete 250-step curves overlaid on the three GB200 curves, with the per-checkpoint values, the step-250 values and the gains tabulated side by side (mean and range for each side). A TPU result is regarded as compatible when, under the same configuration and evaluation protocol, its curves and its final level fall within the range GB200 has shown; reproducing seed 1's high endpoint is not required, and an occasional checkpoint outside the GB200 range is not by itself a failure. Step 0 is reported as in gate 3. A run stopped by the collapse guard, or with fewer than 14 evaluations, is incomplete.

*Suggested numerical reading (proposal, not an agreed criterion):* per checkpoint, [GB200 min − 2.5 pp, GB200 max + 2.5 pp] with at most two checkpoints outside; step 250 within [0.781, 0.838] (the GB200 endpoint range widened by 1.5 pp on both sides; two-sided, because a result well above the band is as much a sign of a different workload as one below). The 2.5 pp figure is roughly half of the GB200 p90 band width; the 1.5 pp figure is the measured single-evaluation variability. Both sides must agree to these numbers before they are used to judge a result.

Training diagnostics outside the GB200 spread do not fail a run but must be explained: entropy near 0.5 and length near 2.3–2.6k tokens are what GB200 showed; entropy collapsing below 0.35 or length falling below 1,000 tokens is what the 2K secondary recipe shows, and would point to a different loss or cap.

### Performance

Compare steady-state end-to-end step times in the same window (steps 20–250, excluding evaluation and checkpoint steps) and the same recipe: generation, reward computation, training, weight synchronization and coordination on the critical path. Report the time-to-0.78 table above in the same form for TPU, from launch and from step 1, with accelerator-hours = chips × elapsed. Record generated-token counts and length distributions per step, since cost changes as the policy learns. Instrumentation must be matched (the GB200 runs had no fixture dumps; per-step rollout dumps and TensorBoard logging are on).

## Provenance and departures from the Meta arms

| item | Meta GPU reference | Meta TPU arm a26d | this recipe | reason |
|---|---|---|---|---|
| model | Qwen3-0.6B-Base | Qwen3-0.6B | Qwen3-0.6B | Base saturates at 0.76 on GSM8K-only RL; the customer's 0.80 expectation and the TPU arm both use the post-trained checkpoint |
| LR | 2e-5 declared | 2.5e-6 as run (their doc §10) | 2e-6 | at the declared 2e-5 the recipe is unstable on a verified implementation; at 2e-6 GB200 reproduces the Meta GPU curve within evaluation noise |
| cap / penalty | 2048, penalty from 1536 | 2048, none | **8192, none** | customer production budget; at 8K the penalty (from 7,680) is rarely reached (cap-hit ≈ 1–2 %), and an 8K + penalty run fell inside the 8K no-penalty band |
| loss form | ratio ≡ 1 + TIS 3.0 | sampler-denominator PPO clip | ratio ≡ 1 + TIS 3.0, single forward | Meta GPU semantics; PPO-IS shown non-equivalent |
| advantage | GRPO (std) | Dr.GRPO (no std) | GRPO (std, ddof 1) | verified to 4e-7 against reference |
| aggregation | token-mean | seq-mean-token-mean | token-mean | verified by replay |
| eval | first 512 | random 512 | all 1,319 | customer requirement |

## GB200 reference and handoff package

Reference experiment IDs (64 GB200, 16 nodes):

```
qwen3_0p6b_base_sf_tis_16n_cap8k_nopen_seed1_16n64g_20260922_0254
qwen3_0p6b_base_sf_tis_16n_cap8k_nopen_seed2_16n64g_20260922_1735
qwen3_0p6b_base_sf_tis_16n_cap8k_nopen_seed3_16n64g_20260922_2105
```

Package root: `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_8k/` (`PACKAGE_MANIFEST.sha256` covers every file).

- `model/`: weights, configs, tokenizer, `MODEL_SHA256`, `model_identity.json` (fingerprint).
- `data/`: `gsm8k_boxed_train.parquet`, `gsm8k_boxed_test.parquet`, Meta's `train.jsonl` / `test.jsonl`, `meta_reference/` (Meta's CSVs and `prompt_example.json`), `train_order_seed{1,2,3}.parquet` + `step_manifest_seed{1,2,3}.json`, `DATA_COUNTS.json`.
- `fixtures/`: `prompt_fixture.json`, `meta_reward_fixtures.json`, `scorer_fixture_8k.jsonl`, `logprob_fixture_8k.json`, `fixture_step{1,2}.npz/.json` (full step batches), `replay_step1_reference_8k.json/.log`, `replay_grad_step1/grad_step1.safetensors` (reference per-parameter gradient), `grad_compare_8k.json/.log`, `replay_delta_8k.json/.log`, reward self-test logs.
- `code/`: launcher `run_qwen3_0p6b_base_gsm8k_boxed.sh`, `boxed_math_reward.py`, `build_gsm8k_boxed_data.py`, the four verl patches, `replay_single_step.py`, `verl_grad_from_optim.py`, `adam_apply_check.py`, `compare_two_runs.py`, `band_plot.py`, `plot_phase0.py`, `collapse_guard.py`, `paired_eval_bootstrap.py`, `make_logprob_fixture.py`.
- `env/`: verl commit, executed `ray_trainer.py` with patch markers, `versions.txt`, `pip_freeze.txt`, GPU/driver, `resolved_config_seed{1,2,3}.yaml`, `CONFIG_DIFF.txt` (only `data.seed` and derived names differ).
- `runs/`: `launch_seed{1,2,3}.log`; `seed{1,2,3}/`: TensorBoard events, `val_dump/<step>.jsonl` (14 files), `rollout_dump/<step>.jsonl` (250 files), guard log, start/end epoch anchors, `EXPERIMENT_NAME`.
- `band/`: `gb200_band_gsm8k_8k.png/.json`, `gb200_curves_gsm8k_8k.png`, `diagnostics_seed{1,2,3}.png`, `summary.json` (all numbers in this rulebook), `step0_greedy_variability.json`.
- `checkpoints/`: `seed{1,2,3}_step250/` (HF safetensors), `fixture_seed1_step2/` (θ₂ of the fixture job, with README).

### TPU result formats

The GPU tooling reads two formats; deliver TPU results in them or with a converter (which then becomes part of the package).

1. Validation dumps: one JSONL per evaluation checkpoint named `<step>.jsonl`, one row per question, fields `input` (rendered prompt text, special tokens stripped), `output`, `gts`, `score`, `reward_raw`, `acc`, `fmt`, `length_penalty`, `qid`, plus a sidecar recording dataset, seed and step.
2. Scalar curves: TensorBoard event files with exactly these tags — `val-core/gsm8k_boxed_test/acc/mean@1`, `val-aux/gsm8k_boxed_test/fmt/mean@1`, `critic/score/mean`, `actor/entropy_loss` (entropy from the update pass; `actor/entropy` is accepted as an alias), `actor/grad_norm`, `actor/lr`, `response_length/mean`, `response_length/clip_ratio`, `timing_s/step`, `training/rollout_probs_diff_mean`, `rollout_corr/kl` — together with the per-step rollout dump (one JSONL per step: `uid`, `qid`, `acc`, `score`, `fmt`) used for train accuracy and zero-advantage groups.

Recipe source reference: `https://github.com/syangx39/verl/tree/<commit>/examples/grpo_trainer/my_scripts_GKE_gsm8k` — fill in the commit recorded in `env/verl_commit.txt`; the manifest identifies the code actually executed by the reference runs.

## Appendix A — Secondary reference `gsm8k_2k_v1` (context only)

Identical to `gsm8k_8k_v1` except: response cap 2,048 and the training-only overlong penalty (0 through 1,536 tokens, linear to −1 at 2,048, applied to `gsm8k_boxed_train` only; evaluation reports the raw reward). Two GB200 seeds, 64 GPUs:

| | seed 1 | seed 2 |
|---|---|---|
| step 0 | 0.7036 | 0.7096 |
| step 250 | 0.8226 | 0.8180 |
| sustained ≥ 0.80 | step 80, 26.6 min, 28.4 GPU-h | step 100, 31.3 min, 33.4 GPU-h |
| steady step time | 14.1 s | 14.0 s |
| final length / entropy | 586 / 0.33 | 572 / 0.32 |

Band width median 0.5 pp, max 3.6 pp; final mean 0.820 [0.818, 0.823]. Compared with the 8K recipe: +1.5 pp mean accuracy, one fifth of the seed variance, 2.8× shorter steps and 3–5× fewer accelerator-hours to 0.78. With the 2K cap the policy sharpens to entropy ≈ 0.33 and 570-token answers; with the 8K cap it does not, with or without the penalty (an 8K + penalty run, seed 1, ended at 0.797 with length 2,146 and entropy 0.50, inside the 8K no-penalty range). No causal claim is made about which component produces the difference. Experiment IDs: `qwen3_0p6b_base_sf_tis_16n_seed1_16n64g_20260921_1823`, `qwen3_0p6b_base_sf_tis_16n_seed2_16n64g_20260922_0040`.

## Appendix B — Validation history behind the frozen choices

- Meta's GPU reference (declared LR 2e-5) did not reproduce on a verified implementation: two seeds diverged at steps 15–20; the full update chain (advantages 4e-7, loss 1e-4, gradient direction 0.998, Adam to 1 ulp) was checked against an independent PyTorch reference. At LR 2e-6 the GB200 curve matches Meta's reference within evaluation noise (eval −1.6 / −2.9 / −1.8 pp at steps 20/40/60; train accuracy point-wise from step 30). Meta's TPU documentation independently records 2.5e-6 as the LR actually run.
- Single forward vs two-pass on an identical injected batch: gradient cosine 0.99995 (rel. err 0.98 %), Δθ cosine 0.99972.
- TIS vs sampler-denominator PPO on an identical batch: gradient cosine 0.9897 (rel. err 14.3 %), Δθ cosine 0.974; over 250 steps (2K recipe) PPO-IS sustained 0.80 at step 140 vs 80–100, final 0.811 vs 0.820, entropy 0.45 vs 0.33, length 750 vs 570.
- Cap sensitivity: see Appendix A and `band/` of the `gsm8k_2k` runs.