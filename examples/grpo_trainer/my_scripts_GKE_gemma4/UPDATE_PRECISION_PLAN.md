# Gemma-4: investigate weak learning and verify optimizer update precision

## Gist

Investigate why the numerically stable Gemma-4 `rl-05` run in [PR #9](https://github.com/syangx39/verl/pull/9) shows only a small held-out accuracy gain, and use the evidence to decide whether the training recipe needs an optimizer-precision change.

The first hypotheses to test are ineffective small updates with BF16 persistent weights / optimizer state, and weak or misleading training reward signal. Neither is an established root cause. This work should produce measured update evidence and a reproducible recipe decision, including a negative result if precision is not responsible.

**Status: items 1–3 complete; item 4 in progress (FP32 frozen, 200-step seed 1 running)** ([UPDATE_PRECISION_RESULTS.md](UPDATE_PRECISION_RESULTS.md)). The initial commit was plan only.

## Plan

### 1. Preserve the reference and inspect existing evidence

- [x] Keep the original PR #9 BF16 seed 1 as a separate reference. Record source/image/patch hashes, model identity, resolved config, and data/prompt/eval fingerprints. — done: [`update_precision/reference_manifest.json`](update_precision/reference_manifest.json).
- [x] Pair step-0 and trained eval outputs by stable question identity; assert unique IDs and matching input/ground truth before computing gained/lost questions, paired uncertainty, or a paired bootstrap interval. Do not rely on row count or positional zip alone. — done: unique `qid`s, input/ground-truth hashes checked across all checkpoints and the replicate; bootstrap intervals ([results](UPDATE_PRECISION_RESULTS.md#12-eval-pairing-by-question-identity-and-the-eval-noise-floor) §1.2).
- [x] Validate the existing full training dumps: 256 prompt groups with 8 completions each, matching inputs/answers within each group. Distinguish repeated question IDs from distinct group occurrences if necessary. — done: 200 steps × 256 groups × 8, no mixed groups ([results](UPDATE_PRECISION_RESULTS.md#13-training-groups-and-reward-signal) §1.3).
- [x] Report actual reward variance, constant-reward / zero-advantage groups, all-correct and all-wrong groups, and groups whose reward differences come only from format credit. Separate accuracy from format reward. — done ([results](UPDATE_PRECISION_RESULTS.md#13-training-groups-and-reward-signal) §1.3).
- [x] Inspect a small sample of incorrect and correct-to-incorrect eval answers for extraction/scoring errors. — done, incl. correct→wrong eval transitions; re-scoring reproduces all labels (0/600 changed) ([results](UPDATE_PRECISION_RESULTS.md#14-grading-audit) §1.4).
- [x] Correct report presentation where needed: show or annotate timing outliers, avoid describing format improvement as monotonic, and avoid attributing correctness flips to decision boundaries without supporting logits. — done in EVAL_PROTOCOL.md §5 / README §4 ([results](UPDATE_PRECISION_RESULTS.md#15-presentation-corrections-to-the-pr-9-docs) §1.5).

Defer new replication launches until these checks guide the next experiment. Any already running or completed replication remains evidence for the original BF16 recipe; this plan does not stop or discard it.

### 2. Measure effective updates in a 1–2-step probe

Use the same model, objective and a captured frozen batch. Preserve token IDs, masks, advantages, recorded log-probs and the global loss denominator.

- [x] Inspect actual optimizer-owned parameters, gradients, `exp_avg` and `exp_avg_sq` dtypes at runtime, including whether an FP32 master copy exists. Config names alone are insufficient. — done: bf16 param / grad / `exp_avg` / `exp_avg_sq`, no FP32 master ([results](UPDATE_PRECISION_RESULTS.md#22-runtime-dtypes-and-settings-all-64-ranks) §2.2).
- [x] Record the LR actually used, optimizer settings, gradient norm before/after clipping, and finite status. — done: lr 1e-6 read from the param group at the step; grads finite (no non-finite skip in any log) ([results](UPDATE_PRECISION_RESULTS.md#22-runtime-dtypes-and-settings-all-64-ranks) §2.2).
- [x] Snapshot persistent parameters before and after the update; subtract in FP32. Report global and per-tensor update norm, relative update norm, and fraction of changed elements, including small/norm tensors. — done: global, per flat tensor and per original-parameter category ([results](UPDATE_PRECISION_RESULTS.md#23-v1-results-applied-vs-ideal-update) §2.3–2.4).
- [x] Separately measure changes in persistent trainer weights and the BF16 weights exposed to rollout. Small FP32 updates may accumulate before becoming visible after casting. — done: BF16-cast visibility per step and cumulative, both precisions ([results](UPDATE_PRECISION_RESULTS.md#24-v2-moments-vs-write-back-per-parameter-categories-rollout-visibility) §2.4).
- [x] Run an optimizer-only counterfactual with identical initial numerical parameter values, captured gradients, and initial Adam states. Compare the FP32 update with the update retained in BF16. — done in-run on the captured post-clip gradients of each step (shadow FP32 AdamW from the same BF16-quantized values cast to FP32 and fresh moments) ([results](UPDATE_PRECISION_RESULTS.md#21-method) §2.1).
- [x] Distinguish final weight write-back rounding from BF16 Adam-state arithmetic. An FP32-master-plus-FP32-moments variant changes both; it does not by itself isolate either effect. — done for steps 1–2 with a 2×2 (moments × write-back) ([results](UPDATE_PRECISION_RESULTS.md#24-v2-moments-vs-write-back-per-parameter-categories-rollout-visibility) §2.4); long-run moment effects noted as open in §2.5.

For initial-value matching, cast the same BF16-quantized reference weights to FP32 for the counterfactual. Do not silently compare them with a differently initialized or higher-precision checkpoint. A low changed-element fraction on one step is evidence to investigate, not proof that learning is impossible.

### 3. Run a controlled precision comparison only if warranted

- [x] If the probe supports the precision hypothesis, add a configuration with FP32 persistent optimizer weights/master and FP32 moments while retaining BF16 model computation and rollout where supported; verify the actual runtime dtypes. — done: `++actor_rollout_ref.actor.fsdp_config.model_dtype=fp32`; runtime dtypes verified fp32 for params, grads and moments; compute stays bf16 (FSDP mixed precision default) ([results](UPDATE_PRECISION_RESULTS.md#22-runtime-dtypes-and-settings-all-64-ranks) §2.2).
- [x] Keep the initial numerical weights, data order, prompt/reward, batch size, objective, LR schedule, clipping, and eval protocol fixed for the comparison. — done: single override `model_dtype=fp32`; identical prompt `qid`s verified at all 60 steps ([results](UPDATE_PRECISION_RESULTS.md#3-controlled-precision-comparison-plan-item-3--complete-60-steps-seed-1) §3).
- [x] Preserve the Gemma-4 recipe from PR #9: GRPO advantages, PPO clipping, and global token-mean aggregation. Do not substitute the earlier Qwen REINFORCE/TIS recipe. — done: same frozen launcher and wrapper ([results](UPDATE_PRECISION_RESULTS.md#3-controlled-precision-comparison-plan-item-3--complete-60-steps-seed-1) §3).
- [x] Re-run the RoPE guard and trainer–sampler log-prob diagnostics, then a short operational check before a full run. — done in the 2-step FP32 probe runs: RoPE guard PASS 64/64, `k3_kl` 0.00071–0.00072, memory and step time unchanged.
- [x] For end-to-end comparisons, measure gradient differences rather than requiring identical gradients after precision paths diverge. Only the optimizer-only probe reuses exactly the same captured gradients. — done at the level of per-window gradient-norm medians (BF16 0.26 / 0.24 / 0.31 vs FP32 0.24 / 0.34 / 0.22); per-parameter gradient differences were not measured ([results](UPDATE_PRECISION_RESULTS.md#3-controlled-precision-comparison-plan-item-3--complete-60-steps-seed-1) §3.3).
- [x] If precision is not supported as the cause, record that finding and select the next single-factor investigation from the reward, update, or eval evidence. Avoid changing precision, LR, prompt and reward together. — not triggered: precision is supported (training accuracy +1.87 pp on identical prompts in steps 41–60; [results](UPDATE_PRECISION_RESULTS.md#3-controlled-precision-comparison-plan-item-3--complete-60-steps-seed-1) §3.4).

Short runs establish numerical and operational behavior; they do not establish a convergence gain.

### 4. Establish the revised learning baseline, then replicate

> Revised order after the `ab70122` review: (1) FP32 seed 1 for the full 200 steps; (2) single-variable optimization against that curve, compared under the fixed eval on accuracy vs steps and vs wall time, screened with seed 1; (3) select and freeze the recipe that balances accuracy and training efficiency, then validate it with multiple seeds. FP32 fixes update precision but is not yet the optimized recipe.

- [x] Freeze the chosen candidate recipe before running a controlled 200-step seed 1 with the existing 5,000-question greedy eval protocol. — done: FP32 frozen in `run_gemma4_e2b_rl05_parity.sh` (sha256 `7d511735…`, commit `203b40f`); 200-step seed 1 `gemma4_e2b_rl05fp32_16n64g_s200_seed1_1022_2019` running.
- [ ] Report the full trajectory and the predeclared step-200 endpoint, with question-paired uncertainty after identity validation. Do not replace the endpoint with the best checkpoint.
- [ ] Compare learning behavior, trainer–sampler consistency, response lengths and cost against the preserved BF16 reference.
- [ ] Decide whether the evidence justifies freezing the candidate and launching seeds 2/3. If gains remain unclear, document that outcome and the next diagnostic decision rather than claiming convergence.
- [ ] Keep different precision/recipe cohorts separate. A revised seed 1 must not be pooled with original BF16 seeds for the three-seed reference.

The shared eval protocol remains fixed. A comparable GPU/TPU gate reference still requires the agreed final recipe and its three-seed endpoint results; this investigation alone does not establish the customer accuracy gate.

## Starting evidence from PR #9

Reference head: `4c23cccf536bbe1901953a0b1e001f26e95c824d`.

| Item | Observed reference |
|---|---|
| Run | `gemma4_e2b_rl05_16n64g_s200_seed1_1021_1955` |
| Hardware / duration | 64 GB200, 200 training steps |
| Eval | 5,000 questions, greedy, step 0 and every 20 steps through 200 |
| Accuracy | 36.26% at step 0 → 36.80% at step 200; +0.54 percentage points |
| Trainer–sampler k3 | Approximately 0.00064–0.00074 |
| Actor configuration | `model_dtype=bfloat16`, `torch.optim.AdamW`, constant LR `1e-6` |
| Runtime optimizer precision | Still needs direct parameter/state inspection |

The existing paired calculation suggests that the endpoint gain is not yet clearly distinguishable from zero; validate question pairing before treating its uncertainty estimate as final. Stable log-prob diagnostics and nonzero gradient norms do not establish that optimizer updates are effectively retained.

Sources:

- [PR #9 seed-1 report](https://github.com/syangx39/verl/blob/4c23cccf536bbe1901953a0b1e001f26e95c824d/examples/grpo_trainer/my_scripts_GKE_gemma4/eval_5k/seed1_16n64g/seed1_report.md)
- [Frozen eval protocol](https://github.com/syangx39/verl/blob/4c23cccf536bbe1901953a0b1e001f26e95c824d/examples/grpo_trainer/my_scripts_GKE_gemma4/EVAL_PROTOCOL.md)
- [Pinned FSDP engine](https://github.com/verl-project/verl/blob/ace775e87d8765bcdd114aac734ab71da5367a0f/verl/workers/engine/fsdp/transformer_impl.py)
- [Pinned optimizer construction](https://github.com/verl-project/verl/blob/ace775e87d8765bcdd114aac734ab71da5367a0f/verl/workers/config/optimizer.py)

## Completion and evidence

Deliver the offline audit, effective-update measurements, the recipe decision, and any resulting controlled run evidence with exact commands and manifests. Report both positive and negative findings. Update the PR description as each stage finishes.

This follows PR #9's patch persistence, frozen eval and first-seed baseline; it does not duplicate those changes or claim that the three-seed learning baseline is already complete.

## Validation of this initial commit

Documentation only; no runtime code or training configuration changed. Reviewed the PR #9 reports/configuration and pinned optimizer construction. No GPU experiments or runtime tests were executed for this plan.

AI assistance: Codex helped draft this investigation plan. Human review and experimental validation are required before accepting any implementation or recipe conclusion.
