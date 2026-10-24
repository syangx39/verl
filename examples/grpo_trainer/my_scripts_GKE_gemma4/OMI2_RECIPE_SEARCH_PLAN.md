# Gemma-4 E2B / OMI2: recipe search after the GSM8K baseline (PR #15)

## Gist

Find a stronger, efficient **Gemma-4 E2B OMI2 recipe** using bounded, interpretable 100-step seed-1 screens. Start from PR #11's **A3a: 36.06% → 40.96% at step 100**, preserve the frozen OMI2 evaluation, and test exploration, the policy-loss path, and group size before spending on additional seeds. The deliverable is the **best tested candidate and its evidence**, not a claim of a globally optimal recipe.

OMI2 already learns: A3a gains **+4.90 pp**, with a question-paired 95% interval of **[+3.76, +6.06] pp**. GSM8K's larger gain motivates hypotheses; it does not establish an OMI2 accuracy target or identify which change caused the difference.

**Status: plan only. No new training runs are launched by this PR.** Implementation, launch commands, and results will be added as the plan is executed.

## 1. Evidence and relationship to existing work

| Evidence | Observation | Decision for this round |
|---|---|---|
| A0, OMI2 FP32 control | Step 100: 39.62% | Retain as historical context |
| A1, remove format reward | 38.96%; no detectable accuracy improvement; reasoning-format compliance falls | Do not repeat without a new scorer finding |
| A2, remove duplicate training prompt wrapping | 39.20%; no detectable improvement under the frozen eval prompt | Preserve A3a's prompt for the new screens |
| **A3a, constant LR 2e-6** | **40.96%**; +1.34 pp vs A0, paired CI [+0.32, +2.44] | Primary historical control |
| A3b, constant LR 4e-6 | Collapses; final accuracy 0% | Do not retry or increase LR above A3a in this round |
| A4, half-cosine | 40.38%; −0.58 pp vs A3a, CI [−1.60, +0.36]; no desired stability improvement | Not a candidate; do not rerun this schedule |
| A5 / A6 | Temperature / group-size learning runs paused in PR #12; A6's two-step smoke already passed | Reuse their definitions and tools; do not create duplicate experiments |
| GSM8K, PR #14, seed 1 | Normalized-scorer accuracy 80.59% → 91.51% at 100 → 92.80% at 250 | Evidence that the repaired model/trainer can learn; different data, scorer and recipe |

Sources: [PR #11 result ledger](RECIPE_ABLATION_PLAN.md), [PR #12 plan and A4 evidence](https://github.com/syangx39/verl/blob/220d30ceff39e4c06355ae6a8d436ad7a6e02824/examples/grpo_trainer/my_scripts_GKE_gemma4/RECIPE_OPTIMIZATION_PLAN.md), and [PR #14](https://github.com/syangx39/verl/pull/14) (reviewed at `c97e43bc366e58aa6dfa923f1c249523720053cf`).

The GSM8K gain is strongly associated with improved boxed-answer compliance; format versus mathematical improvement has not been separated. Do not assume its +12.21 pp is entirely new mathematical capability, or that copying its loss will reproduce that gain on OMI2.

**Why a new PR:** PR #12 retains A4's result and the prepared A5/A6 tooling. PR #15 adds the OMI2 return strategy, a controlled full-vocabulary sampling arm, a REINFORCE + TIS comparison, and common selection/cost reporting. A5/A6 keep their original IDs and have one result ledger; link the PR #15 results from PR #12 when they run. Do not duplicate their launchers or silently change their definitions.

Qwen's successful OMI2 `stab_kl0` run is also relevant: it used PPO-style clipping and n=8. Therefore neither REINFORCE nor n=16 is already proven necessary for OMI2 learning. Cross-model gains and the old Qwen 1,000-question eval are context, not Gemma acceptance thresholds.

## 2. Frozen control and evaluation

**Control run:** `gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753`.

Use the actual A3a runtime config, not launcher defaults: [922-key config](https://github.com/syangx39/verl/blob/220d30ceff39e4c06355ae6a8d436ad7a6e02824/examples/grpo_trainer/my_scripts_GKE_gemma4/recipe_optimization/a3a_control/resolved_config.json) and [manifest](https://github.com/syangx39/verl/blob/220d30ceff39e4c06355ae6a8d436ad7a6e02824/examples/grpo_trainer/my_scripts_GKE_gemma4/recipe_optimization/a3a_control/manifest.json). Freeze the image digest, runtime commit, model hashes and managed patch hashes in each new run's manifest. If an environment change is unavoidable, record and validate it before attributing a difference to the recipe.

| Item | Unchanged unless named as an arm's factor |
|---|---|
| Model | `google/gemma-4-E2B-it`, original checkpoint, fresh optimizer; no continuation from A3a or GSM8K |
| Topology / precision | 64 GB200, 16 × 4, colocated, rollout TP=1; FP32 persistent parameters and Adam state, BF16 compute/rollout; validated Flex/RoPE patches and all-rank guards |
| Batch / updates | 256 prompts × 8 completions; `ppo_mini_batch_size=256`, `ppo_epochs=1`; one full-batch optimizer update, global valid-response-token mean |
| Advantages | GRPO group-normalized reward advantages; unchanged epsilon and masking |
| Control loss | `loss_mode=vanilla`, PPO clipping low/high 0.2/0.28, dual-clip c=3; `bypass_mode=false`, **`rollout_is=null`** |
| Control correction note | The configured IS threshold 2 is **inactive** when `rollout_is=null`; A3a is not a TIS-2 recipe |
| Optimizer | **LR=2e-6 constant, no warmup**, weight decay **0.01**; preserve A3a's betas, epsilon and gradient clipping |
| Regularization | No reference-KL loss and no entropy bonus |
| Training decoding | T=0.8, top-k=50, top-p=0.95; response cap 8192, prompt cap 8192; no length penalty |
| Prompt / reward | Existing double-wrapped OMI2 training prompt; correctness plus **0.1 × format**; unchanged math scorer and answer extraction |
| Data order | Seed 1, same ordered prompt instances as A3a, including repeats; preserve row/question/answer identities |
| Screen | 100 optimizer updates; evaluate at **0, 20, 40, 60, 80, 100** |

The base launcher still defaults to LR=1e-6. Every A3a-derived arm must explicitly override it to **2e-6** and assert the resolved schedule. Clear A1-specific reward variables and A2-specific train-file overrides before launch. Do not copy GSM8K's weight decay 0, LR schedule, 128-prompt batch or numeric-only scorer into an OMI2 arm unnoticed.

Keep [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) frozen: greedy n=1, 8K response cap, the same rendered prompts and unchanged scorer on **5,000 unique, train-disjoint questions**. Do not use the raw OMI2 validation partition, which has training overlap.

| Artifact | SHA256 |
|---|---|
| `data/openmathinstruct2/train.parquet` | `5404981a07c78a5c02d4f493293fc30bd1ddb42908376566fa9a1020abbdf01b` |
| `data/openmathinstruct2/gemma4_eval_5k/eval_5000.parquet` | `d092a9c5e80aa2b3cf8e4e12d8367769e770f69ee6627ff1285fd5693efc662d` |
| `data/openmathinstruct2/gemma4_eval_5k/eval_5000_prompt_tokens.parquet` | `a54fbb327d9c60545c49edfc26107b21a1e55f3ef9e673d80024c79e821b5ca4` |

Paths above are relative to `/workspace/meta-RL`. A temperature or training-filter override must not change greedy eval decoding through an interpolation.

## 3. Before spending on new 100-step runs

### 3.1 Audit reward and useful learning signal offline

Use A3a's existing train/eval dumps and the actual OMI2 scorer. The GSM8K currency/unit false negatives justify an audit; they do **not** prove the same bug exists in OMI2.

- Recompute the frozen scores on the archived responses and assert agreement with the logged fields. Inspect a deterministic, documented sample of failed extractions, wrong answers with valid answer tags, missing reasoning tags, cap hits and scored-correct answers. Distinguish extraction errors from genuinely wrong mathematics; manually check any proposed scorer false positives/negatives.
- Report sampled training accuracy, format rate and total reward separately. A formatting gain is not automatically a mathematical gain.
- At windows 1–20, 41–60 and 81–100, report **zero actual reward variance / zero advantage**, all-correct, all-wrong, correctness-mixed and format-only reward-varying groups. Also report how much absolute advantage mass belongs to correctness-mixed versus format-only groups. Correctness-mixed share alone is not the fraction of groups producing a gradient.
- For the same windows, report length mean/p90/p99, cap-hit rate, grad norm/clipping, entropy and trainer–sampler mismatch. A3a's cap-hit maximum was only about 0.54%; widespread truncation is not the current leading explanation.

If this finds a material scorer implementation bug, fix and version it **before** recipe screening, rescore both control and candidates under common rules, and establish a fresh control if training rewards change. Do not silently normalize only new runs, or import GSM8K numeric normalization into OMI2's symbolic-answer scorer. If no bug is found, keep reward and prompt unchanged.

### 3.2 Remove the logging bottleneck before comparing performance

Carry over the PR #14 logging fix only after its validation. Repeated TensorBoard scalar writes through GCS-Fuse added substantial time in that run; an internal timer excluding those writes is not end-to-end throughput.

- Use the validated console + per-step JSONL file path, with a unique output path for every run; disable synchronous per-scalar TensorBoard writes to Fuse. At the current pin the intended backend is `trainer.logger=[console,file]` plus `VERL_FILE_LOGGER_PATH`. Verify the actual resolved runtime environment and report reader support, not just the shell variable.
- Keep the live stop monitor supplied with promptly flushed metrics. Preserve dumps, provenance and durable log delivery; export TensorBoard offline if needed for plots.
- Run a **20-update A3a-equivalent system check** with the intended 100-step recipe/horizon, terminating at update 20. This is for log completeness, monitoring and timing, not a new accuracy candidate. Record one monotonic coordinator timestamp per completed training step, after its update, sync, scheduled eval/checkpoint work and mandatory logging.
- Record the step interval mean, median/p90 and component timers, including log-write duration, to demonstrate where the old delay went. Do not promise that the new actual interval equals the old internal timer.

Historical A3a cost and new-arm cost may differ because of logging, hardware placement or caches. **Do not credit that improvement to a new loss or sampling recipe.** A formal cost win over A3a needs a matched current-logger control over the same training horizon; the short system check establishes only early-run system performance. A7 and A8 can supply a matched full-length loss-path cost comparison.

## 4. Experiment order and bounded budget

Every arm starts from the initial model with fresh optimizer state. Do not resume another arm's checkpoint or carry forward its overrides. Initial budget: **A5, A7 and A8, each at most 100 steps**. Review those results before the higher-cost A6 or one combination. No LR grid, new model, TP sweep or response-cap change in this round.

| Order / ID | Change | Primary control | Hypothesis / decision |
|---|---|---|---|
| 1. **A5: temperature** | T=0.8 → **1.0** only; retain top-k=50/top-p=0.95, n=8, PPO | A3a | More diverse samples may make more useful within-group correctness comparisons. This is the original PR #12 A5, not a full-vocabulary arm |
| 2. **A7: sampling support** | **top-k=-1, top-p=1.0**; retain T=0.8, n=8, PPO | A3a | Test removing truncation; establish a full-vocabulary PPO control for A8 |
| 3. **A8: loss/correction path** | A7 sampling plus **single-forward REINFORCE + detached token TIS, cap 3** | **A7**; A3a secondary | Test the GSM-style policy-loss/correction path at matched sampling; measure quality and saved old-logprob work |
| 4. **A6: group size, conditional** | n=8 → **16**, retaining 256 prompts, T=0.8 and A3a's PPO/filter settings | A3a | More comparisons per question may improve signal; judge against the extra token/time cost |
| 5. **Combination, conditional** | At most one explicitly frozen combination of supported changes | Best relevant individual arm and A3a | Test interaction; improvements are not assumed additive |

**Why A7 before A8:** the installed sampler reports processed log-probabilities after temperature and top-k/top-p truncation, while trainer log-probabilities are full-vocabulary at the training temperature. Enabling IS on that truncated sampler introduces a support/distribution issue, not merely a numerical correction. A7 removes the filters explicitly, so A8 can be compared at identical full-vocabulary sampling. This is a controlled pair, not a claim that A8 differs from A3a in only one factor.

If A7 hits a genuine training-collapse stop, do not automatically launch A8 on its unstable sampling setting. First classify the failure and update the plan; do not quietly change A8's temperature or LR. If A7 completes but is weaker, A8 may still run to test whether the loss/correction path changes that outcome, with both comparisons reported.

Run A6 after reviewing the initial screens if learning signal remains a plausible bottleneck or no satisfactory candidate emerges. Reuse the passed PR #12 memory/group-shape smoke; recheck only what changed in code, nodes or logging. Do not run A6 on a winning A5/A8 recipe and still label it the original single-factor A6.

### 4.1 A8's exact objective and configuration contract

Retain GRPO advantages. Let \(m_{it}\) be the response mask, \(A_i\) the detached normalized group advantage, \(\mu\) the rollout policy, and \(\pi_\theta\) the current trainer policy, both at the **same training temperature**:

\[
w_{it}=\operatorname{stopgrad}\!\left[\min\!\left(\exp(\operatorname{clip}(\log\pi_{\theta,it}-\log\mu_{it},-20,20)),3\right)\right],
\qquad
L=-\frac{\sum_{i,t}m_{it}\,w_{it}\,A_i\log\pi_{\theta,it}}{\sum_{i,t}m_{it}}.
\]

The denominator is over all valid response tokens in the global batch. Gradients flow through the final trainer log-probability, not the IS weights or advantages. No additional PPO/dual clipping, reference-KL loss or entropy bonus is active. The number 3 is the TIS upper cap; it is not A3a's dual-clip parameter.

Required overrides on top of A7 (to be incorporated into the validated arm runner):

```yaml
algorithm.rollout_correction.bypass_mode: true
algorithm.rollout_correction.loss_type: reinforce
algorithm.rollout_correction.rollout_is: token
algorithm.rollout_correction.rollout_is_threshold: 3.0
algorithm.rollout_correction.rollout_is_batch_normalize: false
algorithm.rollout_correction.rollout_rs: null
algorithm.rollout_correction.rollout_rs_threshold: null
actor_rollout_ref.actor.policy_loss.loss_mode: bypass_mode
actor_rollout_ref.actor.policy_loss.rollout_correction: ${algorithm.rollout_correction}
actor_rollout_ref.rollout.calculate_log_probs: true
actor_rollout_ref.actor.loss_agg_mode: token-mean
actor_rollout_ref.actor.ppo_mini_batch_size: 256
actor_rollout_ref.actor.ppo_epochs: 1
```

This changes the **objective/correction and execution path together**. It is not an equivalence claim obtained by toggling `bypass_mode` alone. With one optimizer update, the PPO ratio initially equals one when current and old trainer log-probs agree; PPO clipping may not bind. A3a's logged clipping fraction was zero, so do not claim clipping caused its limited gain. A8 may primarily save forward computation; accuracy improvement is an experiment, not a guarantee.

Before A8's 100-step run, use a few frozen rollout groups to check token alignment, temperature, masks, detached weights/advantages and global normalization against the explicit objective. Check that unit IS weights recover the corresponding unweighted policy-gradient reference; do not require A8's gradient to equal A7 when actual IS weights differ. Then run three updates on the production topology, verifying a nonzero FP32 parameter update, finite TIS/ESS, and that the separate old-logprob pass is absent. A smoke validates implementation, not convergence.

### 4.2 A6's batch and cost contract

Keep `ppo_mini_batch_size=256`: at the pinned trainer it is multiplied by rollout n, giving the whole **256 × 16 = 4096** response batch and still **one optimizer update**. Do not change it to 128. Preserve the global token denominator and dynamic microbatch token budget; microbatches only split accumulation.

At 100 steps, n=8 uses **204,800 completions**, n=16 uses **409,600**, with the same 25,600 prompt instances. A6 step 40 and A3a step 80 each use 163,840 completions. This is an **equal-completion comparison, not equal compute**: response lengths, training tokens and GPU-hours must also be reported. Raw correctness-mixed group share rises mechanically with n; include PR #12's n=8-equivalent statistic and actual reward variance.

### 4.3 Conditional combination

Do not schedule a combination merely to imitate the entire GSM8K config. Examples are T=1 plus A7's full-vocabulary support, or adding n=16 to a promising A8. Freeze the exact combination and its primary comparator before launch; keep reward, prompt, LR and evaluation fixed. Start fresh, run at most 100 steps, and label it an interaction experiment. If no individual result justifies it, stop this round rather than spending on a blind bundle.

## 5. Preflight, monitoring and stopping

Reuse and extend the [PR #12 tools](https://github.com/syangx39/verl/tree/220d30ceff39e4c06355ae6a8d436ad7a6e02824/examples/grpo_trainer/my_scripts_GKE_gemma4): `gemma4_config_diff.py`, `gemma4_stop_monitor.py`, `gemma4_ab_compare.py`, `gemma4_signal_density.py`, and `recipe_optimization/launch/g4run_pr12.sh`. They are not all on `main` yet. Import the required reviewed files/commits or work from the reviewed dependency; record their hashes. Extend a single arm registry instead of maintaining divergent copies.

At each launch validate the resolved runtime config against A3a with only the declared arm, output identity, and common logging changes allowed. For A8 also compare directly to A7. Check model/data/prompt/scorer identity, full batch shape, one optimizer/scheduler update, LR=2e-6, FP32 updates, all-rank RoPE guards and metric freshness. Pin the sampler's actual logprob mode; missing fields are not zeros.

Reuse PR #12's tested monitor machinery and existing rules; extend and test the offset and probability-convention handling for A7/A8 before launch. The operator/monitor may terminate **only the affected run** on a predeclared trigger without waiting for another approval. Archive the trigger and actual final completed update. A stopped screen is incomplete, not a step-100 result.

| Signal | Action |
|---|---|
| Non-finite loss, gradients or log-probs; failed model/RoPE guard; corrupted batch or undeclared config change | Stop immediately; classify an invalid launch separately from learning failure |
| Trainer–sampler `k3_kl > 0.003` | Stop after 3 consecutive complete updates, subject to the probability-convention check below |
| Missing `</answer>` in >10% of responses | Stop after 3 consecutive complete updates |
| Sampled accuracy deterioration | For unchanged sampling (A6), >10 pp below matched A3a prompts for 3 updates. For A5/A7/A8, use the offset-adjusted rule below |
| Frozen held-out degradation | Stop if at least 2 pp below the run's own step 0 **and** the paired 95% interval's upper bound is below zero |
| Missing/delayed metrics or partial dumps | Treat as an observability failure; pause/stop unattended execution and repair. Do not score partial data or fill missing metrics with zero |

**Sampling-offset rule:** on matched prompts define \(d_k=\mathrm{acc}_{arm,k}-\mathrm{acc}_{A3a,k}\) in pp, freeze \(b=\operatorname{median}(d_1,\ldots,d_5)\), and from step 6 stop when both \(d_k<-10\) and \(d_k-b<-10\) hold for three consecutive updates. This avoids treating a fixed initial sampling difference as training collapse; frozen eval still exposes a static quality deficit.

**Probability-convention check:** A5 retains truncation, so its processed-sampler/full-vocabulary-trainer mismatch includes a truncation term. Verify this on the initial frozen batch before relying on the numerical guard. A7/A8 use untruncated support. If the convention alone invalidates the threshold, stop at preflight and document a corrected common-convention diagnostic/threshold before training; do not retrospectively loosen a rule to rescue a collapsing arm. `k3_kl` is trainer–sampler mismatch, not KL to the initial model.

Do not use one absolute entropy threshold across temperatures, or stop on one isolated pre-clip gradient spike. Report relative entropy drift and gradient clipping. These are conservative budget-protection rules, not sequentially calibrated hypothesis tests.

## 6. What every completed arm must deliver

1. **Accuracy:** the full six-point frozen eval curve; fixed step-100 accuracy and gain from own step 0; paired gained/lost counts and bootstrap 95% CI against A3a, and against A7 for A8. Do not substitute a best checkpoint. Save per-question predictions and identity checks.
2. **Learning signal:** sampled correctness, total reward and format separately; zero-advantage/correctness-mixed/format-only groups and advantage-mass shares. Use complete 256 × n dumps and retain actual prompt membership.
3. **Stability:** length distribution/cap hits, entropy, gradient norm/clipping, trainer–sampler metrics, loss-specific active clipping/IS diagnostics, monitor latency and stop events. For A8 report IS mean/quantiles, fraction capped and token ESS; inactive PPO metrics are N/A, not evidence of parity.
4. **Efficiency:** actual launch-to-step-100-eval elapsed time and GPU-hours; cumulative completions, response tokens and trainer tokens; component timing. Plot eval accuracy against step, cumulative tokens and actual elapsed time.
5. **Provenance:** exact command, resolved config + allowed diff, code/image/model/data/scorer/patch hashes, per-step metric JSONL, checkpoints and exit status. Save step 100 and run the agreed fixed eval; retain existing intermediate checkpoint policy consistently unless its change is explicitly shared and excluded from performance claims.

For steady throughput use the same declared window **steps 21–99 excluding eval/checkpoint-bearing intervals**, and report the number of retained intervals. Record completion timestamps on one coordinator after the same work boundary. For a contiguous block, average step cost is \((t_b-t_a)/(b-a)\); after exclusions, average the retained intervals themselves. Include mean plus median/p90. Also show end-to-end time including startup, eval, checkpoints and logging. Do not infer wall time from copied file modification times or label `timing_s/step` as end-to-end without checking its boundary.

## 7. Selection and follow-through

- **Accuracy candidate:** completes step 100 without a stop, has a positive paired 95% interval versus its declared primary control, and no unresolved correctness/numerical defect. Report A3a comparison too: an A8 win over a weak A7 alone does not make it better than the best existing OMI2 recipe. A consistent curve is supporting evidence, not six independent significance tests.
- **Efficiency candidate:** predeclare **−0.5 pp** as a provisional within-GPU noninferiority margin for this screen. Require the paired lower bound versus the relevant matched control to exceed −0.5 pp and at least **10% less measured full-run GPU-hours** under the same logging/eval/checkpoint setup. These are engineering screening choices, **not the customer's 1 pp TPU gate**. If the interval is wider, call quality unresolved; lack of a detected difference is not equivalence. A8 vs A7 is the clean loss-path cost comparison; a claim versus A3a requires a current-logger full-length A3a control.
- **No winner:** retain A3a. Summarize which hypotheses failed, and use the reward/signal audit to propose the next experiment. Do not automatically retry 4e-6, remove format reward again, shorten the response cap, or launch more seeds hoping for a better result.
- **After a promising screen:** freeze the exact candidate and run **200 steps from the initial checkpoint with fresh optimizer state and the same constant LR**, then decide on seeds 2/3. Do not restart the optimizer at the screen's step-100 weights or silently add a new schedule. The existing customer protocol remains fixed step 200 and a three-seed GPU reference before testing **TPU mean accuracy ≥ GPU mean accuracy − 1 pp**. A 100-step screen or GSM8K result does not satisfy it.

Question-paired intervals quantify uncertainty for the evaluated fixed models, not training-seed variance. The 5k set is repeatedly used for recipe selection and therefore serves as a development benchmark here. Fresh-seed confirmation improves training-repeatability evidence but does not remove validation-set selection bias; do not present the winning screen's CI as independent final generalization proof. Any later independent audit set must be declared separately without silently replacing the frozen customer evaluation.

## 8. Execution checklist and result ledger

- [ ] Finish PR #14 log-backend validation; carry over only the reviewed, relevant changes.
- [ ] Complete the OMI2 reward/signal audit and archive its reproducible report.
- [ ] Reuse PR #12's arm runner and monitoring/comparison tools; add A7/A8, logging allowlist, probability-convention checks and exact launch commands. Test only the changed contracts, including negative config/group-shape cases.
- [ ] Run the current-logger A3a-equivalent 20-update system check; verify complete, timely metrics and actual timing boundaries.
- [ ] Run A5 from the initial checkpoint; report fixed step 100 before proceeding.
- [ ] Run A7 from the initial checkpoint; report and classify its outcome.
- [ ] Validate A8's objective and three-update smoke, then run A8 if A7's sampling setting is stable.
- [ ] Review the initial screens; run A6 only if justified, without stacking changes.
- [ ] Run at most one justified, predeclared combination.
- [ ] Freeze the best tested candidate and write the 200-step confirmation decision; defer seeds 2/3 until that result.

| Arm | State | Step-0 / step-100 accuracy | Paired delta vs primary control | Actual cost / evidence |
|---|---|---|---|---|
| A3a | Historical control | 36.06% / 40.96% | — | PR #11; old logging, not a matched new-arm cost control |
| A5 | Planned; reuse PR #12 definition | Pending | Pending vs A3a | Pending |
| A7 | Planned | Pending | Pending vs A3a | Pending |
| A8 | Planned; depends on A7 and objective smoke | Pending | Pending vs A7; also report vs A3a | Pending |
| A6 | Conditional; prior memory smoke passed | Pending | Pending vs A3a | Pending |
| Combination | Not selected | Pending | Comparator must be frozen before launch | Pending |

This PR initially changes documentation only. It does not claim that A7/A8 launch support, the logging port, or new GPU tests are already complete.
