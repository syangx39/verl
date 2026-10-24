# Gemma-4 recipe optimization, round 2 (PR #12 draft)

## Gist

Find a better **100-step Gemma-4 E2B FP32 recipe** by testing learning-rate decay and sampling/signal-density hypotheses against PR #11's strongest completed candidate, **A3a: LR 2e-6 constant, step-100 accuracy 40.96%**. Continue seed-1 screening before spending on seeds 2/3 or declaring the final GPU reference.

PR #11 is merged at `39ca638358c121b83b0f8a1c7646932ca993250a`. Its [result ledger](RECIPE_ABLATION_PLAN.md#result-ledger-update-as-experiments-finish) is the frozen source for A0–A3. A3a improved from **36.06% to 40.96% (+4.90 pp)**; vs A0 at step 100 it gained **+1.34 pp, question-paired 95% CI [+0.32, +2.44]**. LR 4e-6 collapsed, and removing format reward or the inner training wrapper did not produce a detectable accuracy improvement under frozen eval.

**Status: A4 complete, not a candidate (§A4 result). A5 and A6 are paused at the user's request (2026-10-24); PR #12 is set aside while the GSM8K baseline (PR #14) proceeds.** GPU preflight passed (§4.3), and the A6 2-step smoke passed. The RayCluster runs on node pool `a4x-highgpu-4g-a4x-pool-4`.

## Plan

### 1. Use Qwen's results to choose hypotheses, not a target score

The successful Qwen runs below are **Qwen/Qwen3-0.6B post-trained**, not Qwen3-0.6B-Base. Older experiment names containing `base` are not model-identity evidence.

| Reference | Workload and evaluation | Step 0 → step 100 | Relevant recipe |
|---|---|---|---|
| Qwen GSM8K 2K disaggregated | 128 prompts × 16 completions; 250 steps × 3 seeds; full 1,319-question test every 20 steps plus 250 | Three-seed mean **70.53% → 81.50%**; final step-250 mean 82.74% | REINFORCE-style loss + detached token TIS capped at 3; LR 2e-6 with 10-step warmup then cosine decay; T=1, top-p=1, no top-k; training length penalty |
| Qwen OMI2 `stab_kl0` (closer task comparator) | 256 × 8; 300 steps × 3 seeds; 1,000-question OMI2 held-out + 1,319 GSM8K, evaluated every 10 steps | OMI2 three-seed mean **36.9% → 49.0% (+12.1 pp)** | **PPO clipping 0.2/0.28, constant LR 1e-6, no warmup**, T=1/top-p=1/no top-k; format coefficient 0 + overlong penalty; FP32 persistent weights/moments |
| Gemma A3a | 256 × 8; 100 steps × 1 seed; frozen 5,000-question held-out every 20 steps | **36.06% → 40.96% (+4.90 pp)** | PPO clipping 0.2/0.28; constant LR 2e-6; T=0.8/top-p=0.95/top-k=50; format coefficient 0.1, no length penalty; FP32 persistent weights/moments |

Sources: [GSM8K rulebook](../my_scripts_GKE_gsm8k_disagg/TPU_GPU_RL_Parity_Rulebook_gsm8k_2k_async1.md), [Qwen OMI2 guide §8](../my_scripts_GKE/Tianyu_TPU_Parity_Guide.md), [Qwen launcher and `stab` preset](../my_scripts_GKE/run_qwen3_0p6b_rl05_parity.sh), [PR #10 comparison and signal-density evidence §4.5](UPDATE_PRECISION_RESULTS.md#45-reading).

What this supports:

- **The closest Qwen OMI2 recipe already learned with eight completions, PPO clipping and a constant LR.** Its success does not establish that Gemma needs REINFORCE, 16 completions or a scheduler.
- **Sampling and correctness-signal density are concrete differences to investigate.** The recorded mixed-correctness-group share was about **50–59% for Qwen**, versus **25.5–27.1% for Gemma A0** and about **26–27% for A3a**. These are observed distributions, not proof that sampling caused the gap; model capabilities and prompt difficulty also differ.
- **A3a motivates a schedule screen independently of Qwen.** Its entropy rose 0.204 → 0.366 and trainer–sampler `k3_kl` about 0.00071 → 0.00105. They are warning indicators to monitor, not evidence that LR decay will improve accuracy or prevent collapse.
- **Prompt and format reward are lower-priority now.** The successful Qwen run also used the double wrapper. A1/A2 found no detectable improvement under Gemma's frozen eval; do not repeat those changes without a new hypothesis.
- The models, held-out sets, seed counts and training recipes differ. These accuracy gains are context, **not a causal ranking, equal-compute learning-speed comparison or Gemma acceptance threshold**.

### 2. Freeze the new control and comparison contract

Control: `gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753`, using its completed steps 0–100 and immutable artifacts from PR #11. Keep A0 (39.62%) as secondary historical context.

Every new arm starts from the same **initial model checkpoint with fresh optimizer state, seed 1**. Do not resume A3a at step 100, stack changes from a preceding arm, or replace the fixed endpoint with a best checkpoint.

Except for the declared factor, preserve:

- Model/config/tokenizer identity; 64 GB200 colocated topology; FP32 persistent weights and Adam moments, BF16 compute/rollout; fixed attention/RoPE patches and guards.
- 256 prompt instances per update, original training parquet and the same per-step sampled rows/order; original double-wrapped prompt and format coefficient 0.1; unchanged correctness scorer, no length penalty.
- PPO/GRPO objective and 0.2/0.28 clipping, normalized group-relative advantages, global valid-response-token mean, no KL and no entropy bonus. Do not silently switch bypass mode or the policy objective.
- One optimizer update per rollout batch; unchanged Adam settings and clipping. Adjust accumulation only as needed to preserve that one full-batch update when the number of completions changes.
- Prompt/response caps 8192/8192; T=0.8, top-p=0.95, top-k=50, n=8 unless that arm explicitly changes one of them.
- The [frozen eval protocol](EVAL_PROTOCOL.md): the same 5,000 unique questions, rendered inputs, ground truths, greedy decoding and scoring at **0, 20, 40, 60, 80, 100**.

### 3. Proposed launch order: independent 100-step screens

| Arm | Only recipe change vs A3a | Hypothesis | Extra evidence required |
|---|---|---|---|
| **A4 — native cosine** | First 100 updates of a 200-step cosine from peak 2e-6 to 0; no warmup (schedule value at count 100 is 1e-6) | Reduce update size with a schedule available on GPU and TPU | Exact update-used LR sequence, resolved schedule horizon and stability trajectory |
| **A5 — temperature** | Training T **0.8 → 1.0**; keep top-p=0.95, top-k=50 and n=8 | More sampling diversity may increase mixed-correctness groups and useful learning signal | Actual sampler/trainer temperature and log-prob semantics; correctness/zero-advantage group shares |
| **A6 — completions** | n **8 → 16**; keep 256 prompts and A3a's T=0.8, constant LR 2e-6 | More answers per question may expose correctness differences within a group | 4,096 completions/update, 256 groups × 16, one update; measured token and time cost |

A4 first addresses the observed late drift using a native cosine schedule; it no longer uses a custom hold-then-linear schedule. A5 then tests the most direct sampling difference suggested by the closer Qwen OMI2 comparator. A6 is a separate, higher-cost signal-density screen. **Each compares against A3a; A5 does not inherit A4's decay, and A6 does not inherit either preceding change.** Record an arm's exact command/config before launch. After each result, record whether the next planned arm still addresses an unresolved question; no automatic large sweep.

**A4 decision: native half-cosine, no custom scheduler.** Use zero-based optimizer count `s`:

```text
lr(s) = 1e-6 * (1 + cos(pi * s / 200))
1-based update k uses s = k - 1, for k = 1..100.
```

| Setting | GPU / verl FSDP | TPU / MaxText RL |
|---|---|---|
| Peak LR | `actor.optim.lr=2e-6` | `learning_rate=2e-6` |
| Training horizon | `actor.optim.total_training_steps=100` and trainer runs 100 updates | Actual training horizon 100 updates |
| Schedule | `lr_scheduler_type=cosine`, `num_cycles=0.25`, `min_lr_ratio=0` | `learning_rate_schedule_steps=200`, RL cosine end value 0 |
| Warmup | Resolved warmup steps 0 (`lr_warmup_steps=0`, `lr_warmup_steps_ratio=0`) | `warmup_steps_fraction=0` |
| Optimizer count | `zero_indexed_step=true`; update 1 uses count 0 | Fresh optimizer count 0; verify the update-used values |

GPU optimizer fields above are under `actor_rollout_ref.actor.optim`. Confirm the fields and scheduler implementation in the actual pinned runtime before launch.

**Indexing matters:** `lr(0)=2e-6`, `lr(40)=1.809016994e-6`, `lr(99)=1.015707317e-6`, `lr(100)=1e-6`. Thus **update 100 uses about 1.0157e-6**; the GPU LR logged after its scheduler advances can be 1e-6. Do not call a post-step value the LR used for that update. The mean over the 100 update-used values `s=0..99` is **1.641606682e-6**. This replaces the former hold-40-then-linear schedule and its average LR.

MaxText's `get_optimizer` may treat `learning_rate_schedule_steps == steps` as unset and fall back to its computed training horizon. Require the **effective schedule horizon to be 200**, the actual update count to be 100 and zero warmup; do not infer the effective horizon from one YAML field. Compare all update-used values, rather than assuming that identically named step counters have the same meaning.

Sources checked: [verl schedule at runtime pin ace775e8](https://github.com/verl-project/verl/blob/ace775e87d8765bcdd114aac734ab71da5367a0f/verl/utils/torch_functional.py) and [MaxText RL optimizer at c28166ca7](https://github.com/AI-Hypercomputer/maxtext/blob/c28166ca7/src/maxtext/trainers/post_train/rl/utils_rl.py). This checks the referenced implementations; it does not verify the TPU team's current Gemma image.

Before eventual TPU parity, obtain its actual optimizer implementation, resolved LR/warmup/schedule fields and used-LR sequence. A very long cosine with **zero warmup** can approximate constant LR, but that is not exact constant LR; retaining 10% warmup with a huge horizon instead creates a huge warmup. Defaults alone do not establish how TPU's current constant-LR baseline was implemented. This verification need not delay the GPU-only screening.

**A5 is deliberately not the entire Qwen sampling bundle.** Top-p/top-k stay fixed so temperature is the only recipe factor. Removing either filter would be a separately declared follow-up, not a silent addition. At the new temperature verify that teacher-forced trainer log-probs and saved sampler log-probs use the intended matching probability convention. Trainer entropy at different T is not an apples-to-apples numerical comparison. **T=1 plus LR 2e-6 is an unvalidated combination**; Qwen's closer OMI2 reference used LR 1e-6. If A5 fails, report failure of this combination/run, not that T=1 is generally ineffective or that its failure shares A3b's mechanism. Use growth relative to A5's own early entropy baseline, not an absolute entropy=0.6 stop line.

**A6 is not a performance-only change.** It keeps question coverage at 25,600 prompt slots over 100 steps but increases completions from 204,800 to 409,600. It changes the statistical training batch. Report actual generated/trained tokens and GPU-hours; wall time need not double exactly. A larger mixed-group fraction alone is not success: it can rise mechanically with n and must translate to accuracy or measured cost efficiency. Do not halve prompts to keep 2,048 completions and call it the same experiment.

Keep **eval every 20 steps** for this screen. A6 step 40 and A3a step 80 each represent **163,840 completions**, but differ in prompt exposure, optimizer updates, token lengths and compute. Label this an **equal-completion-count comparison**, not equal compute. Use measured cumulative tokens and GPU-hours for resource comparisons; do not add five extra evals to the primary arm at this stage.

### 4. Implement and validate before the next GPU runs

- [x] Archive the A3a control manifest, exact launch command/resolved config, model/data/code hashes and result paths. — [`a3a_control/manifest.json`](recipe_optimization/a3a_control/manifest.json), the 922-key runtime config [`resolved_config.json`](recipe_optimization/a3a_control/resolved_config.json) extracted from A3a's driver log, and the model file hashes [`model_manifest.sha256`](recipe_optimization/a3a_control/model_manifest.sha256).
- [x] Configure A4's native cosine, export all 100 update-used learning rates and verify the zero-based update index/horizon against the formula below. Compare the intended MaxText schedule using its actual resolved config before claiming TPU parity; then verify the GPU trainer path on a short smoke. — Configured in [`g4run_pr12.sh`](recipe_optimization/launch/g4run_pr12.sh). The pure export and verl's own scheduler class give identical values ([`used_lr_pure.json`](recipe_optimization/a4_cosine/used_lr_pure.json), [`used_lr_verl.json`](recipe_optimization/a4_cosine/used_lr_verl.json)).
  - GPU trainer path, checked on A4's own first updates (§4.2): rank 0 printed `Total steps: 100, num_warmup_steps: 0`. The logged post-step `actor/lr` is 1.9998766e-6 at step 1 and 1.9995066e-6 at step 2, equal to the export. The monitor checks every step.
  - **MaxText is not verified.** That needs the TPU team's resolved config and update-used LR sequence before any parity claim.
- [x] Make each arm's override explicit in the launcher and resolved config; compare against A3a and reject unintended recipe changes. Keep training-only changes out of eval settings. — Overrides are explicit in [`g4run_pr12.sh`](recipe_optimization/launch/g4run_pr12.sh). [`gemma4_config_diff.py`](gemma4_config_diff.py) compares runtime configs and fails on any undeclared key. At launch, the A6 smoke and A4 passed with only their declared keys plus run identity ([A4 diff](recipe_optimization/a4_cosine/config_diff.json), [smoke diff](recipe_optimization/a6_smoke/config_diff.json)). A5 and A6 are checked at their own launches.
- [ ] Verify A5's probability convention at T=1, including raw/processed sampler log-probs and retained top-k/top-p filters. A training log-prob mismatch must not be mistaken for a useful exploration change. — Code path checked at the pin and in the installed vLLM 0.29.0 (§4.2): sampler log-probs are processed (temperature + top-k/top-p, renormalized); trainer log-probs are full-vocabulary at temperature T. So T=1 can raise `k3_kl` mechanically. Remaining: A5's step-1 `k3_kl`.
- [x] Generalize comparison/signal tools that currently assume 2,048 rows and groups of eight. For A6 require **4,096 rows, exactly 256 groups × 16**, consistent prompt/answer per group and unchanged ordered prompt membership. Add a valid n=16 case and negative incomplete/incorrect-group cases; do not disable identity checks. — Each run is now validated against its own declared shape, with the same identity checks:
  - [`gemma4_ab_compare.py`](gemma4_ab_compare.py): `--a/--b-rows-per-step`, `--a/--b-group-size`. Tests: 1 valid 8-vs-16 case plus 5 negative cases. The real smoke dumps pass against A3a.
  - [`gemma4_signal_density.py`](gemma4_signal_density.py): `--shape` validation and an n=8-equivalent mixed share.
  - The stop monitor accepts only complete 256 × n dumps.
- [x] Confirm global token-mean loss normalization and one optimizer update across A6's full batch. Record gradient-accumulation/microbatch settings; more completions must not silently become multiple optimizer steps. — Code at the pin (§4.2): mini-batch = 256 × n = 4,096, which is 64 per GPU, so 1 iteration and one optimizer/scheduler step per training step. Micro-batches (`use_dynamic_bsz`, 16,384 tokens per GPU) only split accumulation. In the smoke both updates completed, `actor/lr` was logged once per step, and peak memory was unchanged.
- [x] Run an **A6 two-step smoke** before its 100-step run to check memory, the 256 × 16 grouping contract and completion of both updates. Record memory and component timing; two startup-heavy steps do not establish steady-state throughput. The proposed 1.4–1.9× step-time / 4.5–6 h budget is an estimate to replace with measurements, not a performance result. — Passed; see §4.3. At step 2 the smoke's step time was 1.28× A3a's, with 2.0× the tokens and unchanged peak memory. Steady-state cost will come from the A6 run itself.
- [x] Implement and test the predeclared stop rules in §4.1, including consecutive-step counting, complete-dump validation, metric freshness and experiment-specific termination. Do not run unattended until the monitor can observe each completed step promptly. — [`gemma4_stop_monitor.py`](gemma4_stop_monitor.py), with 38 checks. Replays on the real PR #11 dumps: A3b triggers first at step 57; A0, A1 and A3a never trigger. In the smoke, metric lines arrived about 31 s after their dumps (§4.3).
- [x] Run the existing reward-scope/template-identity tests and the changed schedule/batch-contract checks. Keep code and manifests identical across nodes; record probe results. — All pass on the pod after node prep (§4.3). Node prep makes every node identical to the A3a patch hashes.
- [ ] Launch each authorized arm from the initial checkpoint only after its preflight passes. — A4 launched at 2026-10-23T21:05Z after its preflight. A5 and A6 are pending.

No production launch commands are promised in this initial draft: exact overrides will be added with the implementation, rather than presenting unsupported scheduler flags as runnable.

### 4.1 Predeclared monitoring and stop authorization

**Decision:** once the rules below are implemented and checked, the experiment operator/monitor may terminate the affected PR #12 run as soon as any applicable rule triggers, **without waiting for another confirmation**. Terminate only that run's driver/workers; do not stop other jobs or tear down the cluster. These are conservative experiment-budget rules, not calibrated GPU/TPU parity thresholds or proof of a particular failure mechanism.

| Signal | Stop condition |
|---|---|
| Numerical or correctness fault | Any non-finite loss/gradient/log-prob metric, or failed node/RoPE guard: stop immediately |
| Trainer–sampler mismatch | `k3_kl > 0.003` for **3 consecutive complete training steps** |
| Missing final answer delimiter | Fraction of responses without `</answer>` **>10% for 3 consecutive complete training steps** |
| Sampled training accuracy, A4/A6 | Accuracy **more than 10 pp below A3a on the same step's prompt instances for 3 consecutive steps**, with prompt identity validated |
| Sampled training accuracy, A5 | Apply the temperature-offset rule below, starting at step 6 |
| Held-out degradation | At a complete frozen eval, accuracy **at least 2 pp below this run's own step 0**, **and** the question-paired 95% CI upper bound for that difference is below 0 |

A single small held-out decrease or one unusual training batch does not trigger these statistical rules. The held-out rule is a conservative abort heuristic under repeated checks, not a sequentially calibrated significance test.

**A5 temperature-offset rule:** compute `d_k = train_acc_A5(k) - train_acc_A3a(k)` in percentage points on matched prompt instances. Freeze `b = median(d_1,...,d_5)` once the first five complete dumps are available. Starting at step 6, stop only if **both** `d_k < -10 pp` and `d_k - b < -10 pp` hold for 3 consecutive steps. This avoids treating a persistent initial sampling-accuracy shift as training collapse; it can also miss a static quality deficit, which still appears in frozen eval. The other stop rules apply from the beginning.

**Entropy:** report A5's early baseline and relative growth (e.g. last-20 / first-20 median), but do not use entropy=0.6 as a cross-temperature hard cutoff. A4/A6 retain A3a's temperature; report their same-temperature trajectories too.

**Fresh evidence, not delayed Ray log delivery:**

- Read only complete step dumps: 2,048 rows / 256 × 8 for A4/A5, 4,096 / 256 × 16 for A6, with complete group identity and required fields. An atomic completion marker or validated manifest is preferable to guessing from file existence. A partially written dump is neither a low-accuracy batch nor a missing-answer event.
- Derive training accuracy from the explicit correctness field (or unchanged scorer), not total reward; compute missing-delimiter rate from response text.
- Text/reward dumps alone may not contain `k3_kl`, loss or gradient metrics. Require a promptly flushed, per-step trainer metric stream/JSONL or TensorBoard for those signals, keyed by the same update number. A Ray driver log arriving 10–20 steps late is insufficient for this monitor.
- Test monitor latency and metric completeness during smoke. Missing/unreadable metrics are not zeros; resolve an observability failure before unattended training rather than claiming the run stayed below a threshold.
- Save the trigger step, measured values, threshold, timestamp, stop request and actual last completed update; preserve existing logs/dumps/artifacts. Mark the arm `early-stopped/incomplete` with reason, and do not substitute an earlier best checkpoint for step 100 or promise a checkpoint that was never saved.

### 4.2 Implementation and prelaunch decisions

**One launcher for every PR #12 run:** [`recipe_optimization/launch/g4run_pr12.sh`](recipe_optimization/launch/g4run_pr12.sh) `A4 | A5 | A6 | A6SMOKE`.

- Each arm is the frozen wrapper plus A3a's `actor_rollout_ref.actor.optim.lr=2e-6` plus its own declared overrides. The overrides are appended through the wrapper's `"$@"`, so they win over the launcher's `lr=1e-6`.
- It unsets the A1 variables and `GEMMA_TRAIN_FILE`.
- It writes a provenance line (file hashes of data, reward, launcher, wrapper, monitor and itself) and appends `<time, arm, experiment>` to `logs/gemma4_pr12_launches.tsv`.
- It starts the stop monitor before the run.

| Arm | Overrides on top of A3a (`actor_rollout_ref.` prefix omitted) | Monitor |
|---|---|---|
| A4 | `actor.optim.lr_scheduler_type=cosine`, `num_cycles=0.25`, `min_lr_ratio=0.0`, `lr_warmup_steps=0`, `lr_warmup_steps_ratio=0.0`, `zero_indexed_step=true` | acts; logged LR checked against `used_lr_pure.json` |
| A5 | `rollout.temperature=1.0` | acts; LR 2e-6 |
| A6 | `rollout.n=16` | acts; 256 × 16 dumps; LR 2e-6 |
| A6SMOKE | A6's override, `TOTAL_STEPS=2`, no eval | observes only |

**Driver stdout unbuffered (logging only).**

- *Observed in PR #11:* per-step metric lines reached the driver log 13–18 steps late. When A1's progress bar showed step 26, its metrics had reached only step 13; when A3b was at about step 78, they had reached step 60.
- *Likely cause:* the TaskRunner prints each line with `flush=True`, and the progress bar (stderr) arrived promptly. The most likely culprit is therefore the driver's block-buffered stdout, which is redirected to a file on GCS Fuse.
- *A second symptom:* the same buffering lets stderr lines land inside stdout lines. A1's log copy has garbled configuration keys for this reason.
- *Change:* `g4run_pr12.sh` sets `PYTHONUNBUFFERED=1` for the driver process only. It is not a Hydra key and does not reach the Ray workers, so the recipe and resolved config are unchanged.
- *Verification before any unattended run:* the monitor records when every dump and metric line first becomes visible. The A6 smoke must show step 1's metric line before step 2's dump. A4's first steps must also arrive promptly.
- *Safeguard:* a line without its final key (`perf/throughput`) raises an alert, and its missing values count as unevaluable.

**Stop monitor:** [`gemma4_stop_monitor.py`](gemma4_stop_monitor.py) implements §4.1 as written and acts only on its own run's driver.

- *Process match:* the driver must have both `verl.trainer.main_ppo` and `trainer.experiment_name=<exp>` as exact argv elements. The monitor sends SIGTERM, then SIGKILL after 180 s, and records `ray status`.
- *Outputs:* `status.json` (per-step values, streaks, latency, alerts), `monitor.jsonl`, and `stop_trigger.json` when a rule triggers. All are written under `logs/gemma4_stop_monitor/<exp>/`.
- *Validation, 38 checks:*
  - every rule on synthetic runs;
  - partial and invalid dumps;
  - prompt-identity mismatch;
  - the A5 offset baseline;
  - 256 × 16 against an 8-completion reference;
  - a held-out interval that spans 0;
  - guard records;
  - a live `--act` run that terminates only the matching dummy driver;
  - a finished run that exits cleanly.
- *Replay of the committed PR #11 driver logs:*
  - The metric rules trigger on A3b at step 57: `k3_kl` was 0.00349, 0.00485 and 0.00640 at steps 55–57.
  - They never trigger on A0, A1, A2 or A3a, whose `k3_kl` max is 0.00107.
  - No watched metric is non-finite in any committed log.
- *Still to replay:* the dump-based rules, against the A3b, A3a and A0 dumps on the pod, before A4 starts.

**Configuration check at every launch:** [`gemma4_config_diff.py`](gemma4_config_diff.py).

- It compares the configuration the TaskRunner prints before any training step with A3a's.
- `--expect` marks a declared change; `--assert` marks a restated value. Interpolated copies of a factor are declared too.
- An undeclared difference stops the run as an *invalid launch*, not an experiment result.

| Arm | Declarations |
|---|---|
| A4 | `--expect` `lr_scheduler_type=cosine`, `num_cycles=0.25`, `lr_warmup_steps=0`; `--assert` `lr=2e-6`, `min_lr_ratio=0.0`, `lr_warmup_steps_ratio=0.0`, `zero_indexed_step=true` |
| A5 | `rollout.temperature=1.0` and its unused interpolation `distillation.teacher_models.teacher_model.inference.temperature` |
| A6 | `rollout.n=16` and its copies `actor.rollout_n`, `ref.rollout_n`, `critic.rollout_n` |
| A6SMOKE | A6, plus `trainer.total_training_steps=2`, `test_freq=-1`, `val_before_train=False` |

**A4's GPU trainer path is checked on A4's own first updates.**

- *Why not a 2-step smoke:* the trainer sets `actor.optim.total_training_steps` to the run length, so a 2-step smoke would build a 2-step cosine and test a different schedule.
- *Checks:*
  - rank 0 must print `Total steps: 100, num_warmup_steps: 0`;
  - the logged post-step `actor/lr` must be 1.99988e-6 at step 1 and 1.99951e-6 at step 2. The monitor checks every later step too.
- *On failure:* the same as a failed smoke. The run is stopped, recorded as an invalid launch, and relaunched from the initial checkpoint after the fix.
- *Before launch:* `gemma4_lr_schedule_export.py --use-verl` runs verl's own scheduler class on the pod.

**A6 performs one optimizer update per step (code at runtime pin `ace775e8`).**

- `RayPPOTrainer._update_actor` sets `mini_batch_size = global_batch_size = ppo_mini_batch_size × rollout.n` = 256 × 16 = 4,096.
- `train_mini_batch` then takes 4,096 / 64 = 64 completions per GPU, which is all of each GPU's share. So `total_num_iterations = 1`: one `train_batch` call, one optimizer step and one scheduler step.
- Each mini-batch's per-sequence token counts are all-gathered across the 64 ranks. Micro-batching (`use_dynamic_bsz`, 16,384 tokens per GPU) only splits gradient accumulation inside that single update.
- The smoke records memory, component timing and `agent_loop/num_preempted`.

**A5 log-prob convention (code at the runtime pin; confirm the installed vLLM on the pod).**

- *Sampler log-probs:* verl's rollout config defaults to `logprobs_mode=processed_logprobs`. vLLM then returns `log_softmax` of the logits after temperature and top-k/top-p masking, renormalized over the kept tokens.
- *Trainer log-probs:* `old_log_prob` and the policy log-probs are `log_softmax(logits / T)` over the full vocabulary, with no top-k/top-p.
- *Effect on the diagnostic only:* `rollout_corr/k3_kl` therefore contains a truncation term, `(Z − 1) − log Z` per token, where `Z` is the kept probability mass at temperature T. The term is at most about 0.0013 while only top-p = 0.95 binds. It is larger where top-k = 50 binds. Raising T increases it mechanically, without any change in training.
- *Effect on the update: none.* The PPO ratio uses trainer log-probs only, and this recipe has no rollout importance correction. The gap changes the diagnostic and the `k3_kl` stop rule, not the update.
- *Existing measurements:*

  | Run | Truncation | `k3_kl` |
  |---|---|---|
  | T=0.8 runs | top-k / top-p on | about 0.0007 |
  | PR #8 post-fix T=1 run, 4n16g | none (`top_k=-1`, `top_p=1.0`) | 0.00046 at step 1 |

  The numeric part of a KL scales roughly as 1/T². That is consistent with almost all of the 0.0007 at T=0.8 being numeric, and the truncation term being small there.
- *Expectation for A5:* step-1 `k3_kl` of about 0.0005–0.0015, below the 0.003 rule with margin.
- *If A5's first steps are much higher:* that is reported as a convention effect before the rule is relied on. A trigger caused only by the convention would not be evidence of a training failure.

**Analysis tools for unequal batch shapes.**

- [`gemma4_ab_compare.py`](gemma4_ab_compare.py) takes per-run shapes (`--b-rows-per-step 4096 --b-group-size 16`) and per-run training temperatures (labels only).
- [`gemma4_signal_density.py`](gemma4_signal_density.py) validates `--shape label=256x16`. It also reports `mixed_n8eq`, the expected correctness-mixed share had 8 of each group's completions been drawn, because the raw n=16 share rises mechanically with n.
- [`gemma4_eval_report.py`](gemma4_eval_report.py) labels completions per step from the dumps, and the training temperature via `--train-temp`.

### 4.3 GPU preflight results (2026-10-23)

**Cluster change.**

- PR #11's node pool (`a4x-highgpu-4g-a4x-pool-7`) has had no nodes since about 19:36Z.
- At the user's request, the RayCluster's `nodeSelector` was patched to **`a4x-highgpu-4g-a4x-pool-4`** at 20:27Z. The previous spec is backed up.
- Unchanged: the machine type (a4x-highgpu-4g, GB200), 16 nodes × 4 GPUs, the container image digest (`verl-disagg-cu130-arm64@sha256:70e1f1ed…`) and the code. Node prep rebuilds the pin and its managed patches to the same hashes A3a recorded: `monkey_patch` `9be46b09…`, `transformer_impl` `9a43adb8…`.
- The physical nodes differ from the A3a control's.
- The ComputeDomain is Ready, with all 16 nodes in one NVLink clique. It still lists one stale node from pool-7, in a different clique.
- Runtime versions: Python 3.12.3, torch 2.13.0+cu130, vLLM 0.29.0, Ray 2.55.1, transformers 5.12.1 ([`versions.txt`](recipe_optimization/preflight/versions.txt)).

**Preflight checks.** Evidence is in [`recipe_optimization/preflight/`](recipe_optimization/preflight/).

| Check | Result |
|---|---|
| Files on the pod vs the branch | 22 files, including frozen launcher, wrapper, reward and patches: all sha256 identical |
| Node prep, pool-4 | PASS on 16 nodes: managed patch hashes as A3a, pinned `math-verify` installed |
| Pod tests | All PASS after node prep: reward scope; RoPE guard 14/14; template 7/7; group shapes 7/7; signal-density shapes 8/8; config diff 10/10; stop monitor 38/38 |
| A4 learning rates from verl's own `get_cosine_schedule_with_warmup` | 6/6 checks; identical to `used_lr_pure.json` (maximum difference 0) — [`used_lr_verl.json`](recipe_optimization/a4_cosine/used_lr_verl.json) |
| Sampler log-prob convention, installed vLLM | `processed_logprobs` is applied in vLLM's top-k/top-p sampler path; verl default `processed_logprobs` |
| Model files | [`model_manifest.sha256`](recipe_optimization/a3a_control/model_manifest.sha256): sha256 of all 21 files in the model directory, e.g. `model.safetensors` `2db5482b…` |

Before node prep ran on the fresh pods, the reward-scope and RoPE-guard tests failed. `math-verify` was not yet installed and the patches were not yet applied. Both passed after node prep.

**Stop-monitor replay on the PR #11 dumps** (never acting):

| Run vs reference | Rule set | Triggers (rule, step) | Alerts |
|---|---|---|---|
| A3b vs A3a | matched (A4/A6) | **`k3_kl` 57**, `train_acc` 65, `missing_answer` 66, held-out 80 (−24.64 pp) | 0 |
| A3b vs A3a | A5 offset | identical | 0 |
| A3a vs A0 | matched | none | 0 |
| A0 vs A3a | matched | none | 0 |
| A1 vs A3a | matched | none | 0 |

The held-out values reproduce the recorded curves; for example, A3a vs its own step 0 is +4.90 pp at step 100.

**A6 2-step smoke:** `gemma4_e2b_rl05fp32_a6n16smoke_16n64g_s2_seed1_1023_2039`, launched at 20:39Z. Evidence is in [`recipe_optimization/a6_smoke/`](recipe_optimization/a6_smoke/).

- **Valid run:**
  - Driver `exit_code=0`, and both updates completed.
  - Runtime config vs A3a passes, with only the declared keys.
  - The guard check passes for all 64 ranks.
  - The only traceback is the known exit-phase `DataLoader worker … killed`.
- **Batch contract:** both dumps have 4,096 rows = 256 groups × 16. They pass the per-run-shape comparison against A3a's 256 × 8 dumps: identical qid → input / ground truth.
- **Metric latency:** each step's metric line became visible about **31 s** after the monitor saw the step's dump (30.7 s and 31.0 s). Step 1's line arrived before step 2's dump, versus 13–18 steps late in PR #11. The monitor raised no alert.
- **Step 1 slowdown on one node:** the 4 ranks of one node (44–47, pod `q74mq`) reached their first collective about 7 minutes after the other 60. They were CPU-busy in `compute_log_prob`, with GPUs idle, apparently compiling on cold caches. NCCL communicator initialization itself took 2.1 s on every rank.
  - That delay is included in step 1's 585 s.
  - Caches are now warm on all nodes.
  - The job's NCCL timeout is 600 s, so a recurrence would be close to it; A4's first step is watched.
- **Measured step-2 values:**

| Measure (step 2) | A6 smoke | A3a | Ratio |
|---|---:|---:|---:|
| `timing_s/step` | 86.4 s | 67.4 s | 1.28× |
| `timing_s/gen` | 45.3 s | 39.8 s | 1.14× |
| `timing_s/old_log_prob` | 10.1 s | 6.1 s | 1.67× |
| `timing_s/update_actor` | 16.0 s | 9.9 s | 1.62× |
| `perf/total_num_tokens` | 7.86M | 3.90M | 2.01× |
| actor peak allocated / reserved memory | 32.5 / 60.0 GB | 32.5 / 60.2 GB | unchanged; the per-micro-batch token budget is fixed |
| Training accuracy, same prompts | 61.26% | 61.23% | — |
| Correctness-mixed groups, steps 1–2 | 34.6% (n=8-equivalent **27.5%**) | 28.1% | — |

  Two startup-heavy steps do not establish steady-state throughput.
- **Gradient norm:** `actor/grad_norm` at step 1 was 1.95. Across the PR #11 launches, which have identical step-1 weights and prompts, it ranged 0.14–0.80. At step 2 it was 0.41 vs A3a's 0.30. With n=16, a group with a single correct answer gets advantage 3.75, against 2.475 at n=8, so the larger tail is expected. Gradient clipping at 1.0 bounds the update. A6's grad-norm trajectory will be reported against A3a's.

### 5. Judge accuracy, signal quality, stability and cost together

**Primary endpoint:** greedy held-out accuracy at **fixed step 100 vs A3a's 40.96%**. Report question-paired gained/lost counts and bootstrap 95% CI, with exact eval identity checks. Also show each arm's gain from its own step 0 and the complete six-point curve. Do not tune eval prompts, question membership or scorers after seeing results.

**Learning diagnostics:** sampled training accuracy and format separately from total reward; all-correct/all-wrong/mixed-correctness groups; zero-advantage and format-only groups; advantage-mass shares; response length/cap-hit rate; entropy, gradient norm, PPO clipping and trainer–sampler mismatch. Track whether the chosen arm makes correctness-bearing updates more useful, not only more numerous. Trainer–sampler `k3_kl` is not KL to the initial policy.

**Cost:** report time from the same launch boundary through completion of step-100 eval, GPU-hours, generated response tokens and trainer tokens. For steady throughput give full-window mean plus median/p90 with a declared warmup/eval exclusion policy. Plot accuracy against cumulative tokens and time as well as step. A6's twofold completion budget must be visible; fixed-step improvement alone does not establish better compute efficiency.

**A4 stability-candidate label:** accuracy and stability are separate axes. If the step-100 paired CI vs A3a includes zero, A4 can still be recorded as a **provisional stability candidate, quality unresolved** when both entropy and `k3_kl` have lower last-20-step medians and at least **25% less early-to-late drift** than A3a, with no stop rule triggered. Define drift for each metric as `median(steps 81..100) - median(steps 1..20)`, computed from full-resolution recorded values at the same T. The 25% reduction is a predeclared descriptive screen, not a significance claim. Report any observed accuracy loss alongside it; if accuracy clearly degrades, label a quality/stability tradeoff rather than a winning recipe. No detectable difference does not prove accuracy equivalence or long-term stability.

**Selection:** a positive paired interval, a consistent full curve and acceptable diagnostics nominate a candidate. If the interval includes zero, describe the result as inconclusive; a cheaper arm with comparable observed accuracy can still be recorded as a cost candidate, without claiming statistical equivalence. These intervals describe fixed trained models, not seed variance. Repeated recipe screening can overfit the validation set; freeze the candidate before fresh-seed confirmation and do not advertise a seed-1 screen as final generalization proof.

### 6. Keep larger algorithm changes and final validation separate

Do not escalate constant LR beyond 2e-6 after the A3b failure. REINFORCE-style loss + token TIS is a valid future algorithm hypothesis, but Qwen OMI2 did not require it. Such an experiment must specify the full objective, importance-weight detachment/truncation, log-prob sources and bypass path; it is not an equivalent single-flag change.

If A4–A6 leave a clear unresolved question, write the next single-factor hypothesis before running it (for example one sampling filter or a separately specified objective). Combining winners requires its own labeled run because benefits need not add.

After choosing a recipe, freeze it and evaluate the longer horizon and additional seeds. The existing customer protocol remains **step 200, three-seed mean, TPU ≥ GPU − 1 percentage point**. PR #12's 100-step seed-1 screens do not complete or redefine that gate.

## A4 result: native half-cosine

**Run.** `gemma4_e2b_rl05fp32_a4cos_16n64g_s100_seed1_1023_2105`, launched 2026-10-23T21:05Z from the initial checkpoint, seed 1, 16n64g on pool-4. The log closed at 00:21:36Z, about 3 h 16 min including start-up, six evals and shutdown.

**Validity.**
- Driver `exit_code=0`, with 100/100 steps and all six evals.
- The guard check passes for all 64 ranks. The only traceback is the known exit-phase `DataLoader worker … killed`.
- The stop monitor saw every step, with a median metric latency of 31 s (153 s at most, at eval steps). No rule triggered and no alert was raised.
- All 100 logged `actor/lr` values equal the export (maximum relative error 0): 1.9998766e-6 at step 1, 1.7071068e-6 at step 50, 1e-6 at step 100.
- Training prompts are identical to A3a's at every step (validated per step).

**Held-out accuracy (fixed endpoint, step 100).** Paired by validated `qid` on the frozen `eval_5000`. Evidence: [`a4_vs_a3a_s100.md`](recipe_optimization/a4_cosine/a4_vs_a3a_s100.md), [plot](recipe_optimization/a4_cosine/a4_vs_a3a_s100.png), [eval report](recipe_optimization/a4_cosine/a4_eval_report.md).

| Eval step | A3a | A4 | A4 − A3a, pp (95% CI) | A4 vs own step 0, pp (95% CI) |
|---:|---:|---:|---|---|
| 0 | 36.06 | 35.82 | −0.24 [−1.14, +0.72] (identical weights) | — |
| 20 | 39.26 | 38.52 | −0.74 [−1.74, +0.28] | +2.70 [+1.62, +3.76] |
| 40 | 39.24 | 39.14 | −0.10 [−1.14, +0.92] | +3.32 [+2.18, +4.34] |
| 60 | 40.52 | 39.30 | −1.22 [−2.20, −0.18] | +3.48 [+2.38, +4.62] |
| 80 | 40.92 | 40.14 | −0.78 [−1.84, +0.26] | +4.32 [+3.26, +5.44] |
| **100** | **40.96** | **40.38** | **−0.58 [−1.60, +0.36]** | **+4.56 [+3.50, +5.66]** |

- A4 learns from its own step 0, +4.56 pp at step 100.
- At the primary endpoint there is no detectable difference from A3a: the interval includes 0. That does not establish equivalence.
- A4's point estimate is below A3a at all five post-0 checkpoints, and the step-60 interval excludes 0. Those checks are repeated and the curves are correlated, so this is descriptive.
- For context, A4 vs A0 at step 100 is +0.76 pp [−0.30, +1.80].

**Training diagnostics vs A3a.** Medians over the A3a / A4 windows:

| Window | `rollout_corr/k3_kl` | `actor/entropy` | Response length | Training accuracy, same prompts |
|---|---|---|---|---|
| 1–20 | 0.000723 / 0.000707 | 0.207 / 0.204 | 1,730 / 1,825 | 60.1 / 60.2 % |
| 81–100 | 0.000962 / 0.000996 | 0.328 / 0.337 | 1,313 / 1,548 | 62.7 / 62.1 % |

- Training accuracy on identical prompts: −0.17 pp mean over steps 1–100. Descriptive only; consecutive steps are not independent.
- Correctness-mixed groups: 25–27% in both runs.
- Format: about 98–99% held-out in both runs.

**Stability-candidate screen (§5).** Medians of steps 81–100 against steps 1–20 ([`stability_screen.json`](recipe_optimization/a4_cosine/stability_screen.json)):

| Metric | A3a early → late (drift) | A4 early → late (drift) | Late median lower? | Drift reduction |
|---|---|---|---|---|
| Entropy | 0.2069 → 0.3285 (+0.1215) | 0.2039 → 0.3368 (+0.1329) | no | −9.4% (more drift) |
| `k3_kl` | 0.000723 → 0.000962 (+0.000239) | 0.000707 → 0.000996 (+0.000289) | no | −20.8% (more drift) |

Neither condition holds, so **A4 is not a stability candidate.** Over these 100 updates, the half-cosine's lower LR (mean update-used 1.64e-6, reaching 1.02e-6 by update 100) did not reduce A3a's entropy or mismatch drift.

**Cost** (`timing_s/step` mean / median / p90):

| | A4 | A3a |
|---|---|---|
| `timing_s/step` | 66.7 / 64.3 / 80.1 s | 63.1 / 61.4 / 75.8 s |
| Total tokens | 3.67e8 | 3.31e8 |

A4's responses stayed about 230 tokens longer late in training, so it was slightly more expensive.

**Decision: not a candidate.**
- No detectable accuracy difference at the fixed endpoint, with a point estimate below A3a at every checkpoint.
- No stability benefit.
- A3a remains the best completed 100-step seed-1 recipe.

At the user's request (2026-10-24), A5 and A6 are **paused** and PR #12 is set aside while the GSM8K baseline (PR #14) proceeds. The A6 smoke, monitor and tooling remain ready for a later resume.

## Result ledger

| Arm | Exact command/config and experiment ID | Validity / completed steps | Step-0 acc | Step-100 acc | Difference vs A3a, pp (paired 95% CI) | Tokens / GPU-hours / decision |
|---|---|---|---:|---:|---|---|
| A3a control | PR #11 run `gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753`; [manifest](recipe_optimization/a3a_control/manifest.json) | Complete: 100 steps, all six evals | 36.06% | 40.96% | Reference | Reuse recorded evidence; stability caveat |
| A4 native cosine | `gemma4_e2b_rl05fp32_a4cos_16n64g_s100_seed1_1023_2105` via `g4run_pr12.sh A4`; [config diff](recipe_optimization/a4_cosine/config_diff.json), [comparison](recipe_optimization/a4_cosine/a4_vs_a3a_s100.md) | **Complete**: 100 steps, all six evals; guards PASS; no stop rule; LR sequence exact | 35.82% | **40.38%** | **−0.58 [−1.60, +0.36]** | 3.67e8 tokens (A3a 3.31e8); 3 h 16 min; **not a candidate** (§A4 result) |
| A5 T=1 | Not launched; `g4run_pr12.sh A5` | **Paused** (user decision 2026-10-24: PR #12 set aside while GSM8K, PR #14, proceeds) | — | — | — | — |
| A6 smoke (2 steps) | `gemma4_e2b_rl05fp32_a6n16smoke_16n64g_s2_seed1_1023_2039` via `g4run_pr12.sh A6SMOKE` | **Passed** (§4.3); not an accuracy result | — | — | — | 2 steps: 1.28× step time and 2.0× tokens at step 2; memory unchanged |
| A6 n=16 | Not launched; `g4run_pr12.sh A6` | **Paused** (same decision) | — | — | — | — |

Keep completed, null, failed and aborted arms in this ledger. Link commands, manifests, logs, per-question results and plots beside each result. Update the PR description around the final experiments as they are completed.

## Initial validation and scope

The initial plan commits were documentation only, with no runtime change or GPU job. The draft was checked against merged PR #11, the Qwen rulebook/guide/launcher, the existing Gemma comparison evidence and frozen eval protocol; relative document links were checked. The later prelaunch decision selects native half-cosine, records explicit abort authorization and rules, and keeps A6 eval every 20 steps; no monitor or scheduler runtime implementation is claimed by these documentation edits. No new runtime test is warranted for this plan-only commit; implementation checks are listed above.

### Implementation validation (no GPU job)

Run locally with Python 3.13; numpy only for `gemma4_ab_compare.py`.

| Test | Result |
|---|---|
| [`test_gemma4_stop_monitor.py`](test_gemma4_stop_monitor.py) | **38/38**, including the replay of the committed PR #11 driver logs |
| [`test_gemma4_ab_compare_groups.py`](test_gemma4_ab_compare_groups.py) | **7/7** |
| [`test_gemma4_ab_compare_template.py`](test_gemma4_ab_compare_template.py) (existing; unchanged behavior) | **7/7** |
| [`test_gemma4_signal_density_shapes.py`](test_gemma4_signal_density_shapes.py) | **8/8** |
| [`test_gemma4_config_diff.py`](test_gemma4_config_diff.py) | **10/10**, including A3b vs A3a from the committed logs |
| `gemma4_lr_schedule_export.py` (pure) | 6/6 checks |

- **Frozen files unchanged:** launcher `7d511735…`, wrapper `2a37aad5…` and reward `d0bde7b8…` hash to the values A3a recorded.
- **Not yet run:** anything needing the pod image or GPUs. That covers the reward-scope and RoPE-guard tests (`torch`, `math_verify`), the verl scheduler export, dump-based monitor replays, the A6 smoke and every launch-time check above.

This follows the completed PR #11 screen. No overlapping open experimental PR was found in this fork or the upstream `gemma4 ablation` search during closeout. AI assistance: Codex drafted the plan at the user's request; experiment implementation, human review and runtime evidence remain to be added.
