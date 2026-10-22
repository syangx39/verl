# Gemma-4: investigate weak learning and verify optimizer update precision

## Gist

Investigate why the numerically stable Gemma-4 `rl-05` run in [PR #9](https://github.com/syangx39/verl/pull/9) shows only a small held-out accuracy gain, and use the evidence to decide whether the training recipe needs an optimizer-precision change.

The first hypotheses to test are ineffective small updates with BF16 persistent weights / optimizer state, and weak or misleading training reward signal. Neither is an established root cause. This work should produce measured update evidence and a reproducible recipe decision, including a negative result if precision is not responsible.

**Status: plan only.** No new training run, optimizer change, or precision fix is included in this initial commit.

## Plan

### 1. Preserve the reference and inspect existing evidence

- [ ] Keep the original PR #9 BF16 seed 1 as a separate reference. Record source/image/patch hashes, model identity, resolved config, and data/prompt/eval fingerprints.
- [ ] Pair step-0 and trained eval outputs by stable question identity; assert unique IDs and matching input/ground truth before computing gained/lost questions, paired uncertainty, or a paired bootstrap interval. Do not rely on row count or positional zip alone.
- [ ] Validate the existing full training dumps: 256 prompt groups with 8 completions each, matching inputs/answers within each group. Distinguish repeated question IDs from distinct group occurrences if necessary.
- [ ] Report actual reward variance, constant-reward / zero-advantage groups, all-correct and all-wrong groups, and groups whose reward differences come only from format credit. Separate accuracy from format reward.
- [ ] Inspect a small sample of incorrect and correct-to-incorrect eval answers for extraction/scoring errors.
- [ ] Correct report presentation where needed: show or annotate timing outliers, avoid describing format improvement as monotonic, and avoid attributing correctness flips to decision boundaries without supporting logits.

Defer new replication launches until these checks guide the next experiment. Any already running or completed replication remains evidence for the original BF16 recipe; this plan does not stop or discard it.

### 2. Measure effective updates in a 1–2-step probe

Use the same model, objective and a captured frozen batch. Preserve token IDs, masks, advantages, recorded log-probs and the global loss denominator.

- [ ] Inspect actual optimizer-owned parameters, gradients, `exp_avg` and `exp_avg_sq` dtypes at runtime, including whether an FP32 master copy exists. Config names alone are insufficient.
- [ ] Record the LR actually used, optimizer settings, gradient norm before/after clipping, and finite status.
- [ ] Snapshot persistent parameters before and after the update; subtract in FP32. Report global and per-tensor update norm, relative update norm, and fraction of changed elements, including small/norm tensors.
- [ ] Separately measure changes in persistent trainer weights and the BF16 weights exposed to rollout. Small FP32 updates may accumulate before becoming visible after casting.
- [ ] Run an optimizer-only counterfactual with identical initial numerical parameter values, captured gradients, and initial Adam states. Compare the FP32 update with the update retained in BF16.
- [ ] Distinguish final weight write-back rounding from BF16 Adam-state arithmetic. An FP32-master-plus-FP32-moments variant changes both; it does not by itself isolate either effect.

For initial-value matching, cast the same BF16-quantized reference weights to FP32 for the counterfactual. Do not silently compare them with a differently initialized or higher-precision checkpoint. A low changed-element fraction on one step is evidence to investigate, not proof that learning is impossible.

### 3. Run a controlled precision comparison only if warranted

- [ ] If the probe supports the precision hypothesis, add a configuration with FP32 persistent optimizer weights/master and FP32 moments while retaining BF16 model computation and rollout where supported; verify the actual runtime dtypes.
- [ ] Keep the initial numerical weights, data order, prompt/reward, batch size, objective, LR schedule, clipping, and eval protocol fixed for the comparison.
- [ ] Preserve the Gemma-4 recipe from PR #9: GRPO advantages, PPO clipping, and global token-mean aggregation. Do not substitute the earlier Qwen REINFORCE/TIS recipe.
- [ ] Re-run the RoPE guard and trainer–sampler log-prob diagnostics, then a short operational check before a full run.
- [ ] For end-to-end comparisons, measure gradient differences rather than requiring identical gradients after precision paths diverge. Only the optimizer-only probe reuses exactly the same captured gradients.
- [ ] If precision is not supported as the cause, record that finding and select the next single-factor investigation from the reward, update, or eval evidence. Avoid changing precision, LR, prompt and reward together.

Short runs establish numerical and operational behavior; they do not establish a convergence gain.

### 4. Establish the revised learning baseline, then replicate

- [ ] Freeze the chosen candidate recipe before running a controlled 200-step seed 1 with the existing 5,000-question greedy eval protocol.
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
