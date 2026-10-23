# Gemma-4 recipe optimization, round 2 (PR #12 draft)

## Gist

Find a better **100-step Gemma-4 E2B FP32 recipe** by testing learning-rate decay and sampling/signal-density hypotheses against PR #11's strongest completed candidate, **A3a: LR 2e-6 constant, step-100 accuracy 40.96%**. Continue seed-1 screening before spending on seeds 2/3 or declaring the final GPU reference.

PR #11 is merged at `39ca638358c121b83b0f8a1c7646932ca993250a`. Its [result ledger](RECIPE_ABLATION_PLAN.md#result-ledger-update-as-experiments-finish) is the frozen source for A0–A3. A3a improved from **36.06% to 40.96% (+4.90 pp)**; vs A0 at step 100 it gained **+1.34 pp, question-paired 95% CI [+0.32, +2.44]**. LR 4e-6 collapsed, and removing format reward or the inner training wrapper did not produce a detectable accuracy improvement under frozen eval.

**Status: plan only. No new experiment or launcher change is included yet.** The plan and PR description will be updated with implementation, commands, checks and results as work proceeds.

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
| **A4 — LR decay** | LR 2e-6 through update 40, then linear decay to 1e-6 at update 100; no warmup | Keep early progress while reducing late update size | Update-used LR sequence and scheduler/logging index check; full stability trajectory |
| **A5 — temperature** | Training T **0.8 → 1.0**; keep top-p=0.95, top-k=50 and n=8 | More sampling diversity may increase mixed-correctness groups and useful learning signal | Actual sampler/trainer temperature and log-prob semantics; correctness/zero-advantage group shares |
| **A6 — completions** | n **8 → 16**; keep 256 prompts and A3a's T=0.8, constant LR 2e-6 | More answers per question may expose correctness differences within a group | 4,096 completions/update, 256 groups × 16, one update; measured token and time cost |

A4 first addresses the observed late drift. A5 then tests the most direct sampling difference suggested by the closer Qwen OMI2 comparator. A6 is a separate, higher-cost signal-density screen. **Each compares against A3a; A5 does not inherit A4's decay, and A6 does not inherit either preceding change.** Record an arm's exact command/config before launch. After each result, record whether the next planned arm still addresses an unresolved question; no automatic large sweep.

**A4 schedule definition.** Let `k` be the 1-based optimizer update and `lr_used(k)` the LR actually used for that update:

```text
k = 1..40:    lr_used(k) = 2e-6
k = 41..100:  lr_used(k) = 2e-6 - (k - 40) * 1e-6 / 60
```

Check update 1 = 2e-6, 40 = 2e-6, 41 = 1.983333333e-6 and 100 = 1e-6. A post-scheduler logged LR may describe the next update; do not shift this schedule to match a label. Implement and test this exact schedule before launching; do not claim that an existing cosine flag implements it. This arm tests decay only, not adding warmup.

**A5 is deliberately not the entire Qwen sampling bundle.** Top-p/top-k stay fixed so temperature is the only recipe factor. Removing either filter would be a separately declared follow-up, not a silent addition. At the new temperature verify that teacher-forced trainer log-probs and saved sampler log-probs use the intended matching probability convention. Trainer entropy at different T is not an apples-to-apples numerical comparison.

**A6 is not a performance-only change.** It keeps question coverage at 25,600 prompt slots over 100 steps but increases completions from 204,800 to 409,600. It changes the statistical training batch. Report actual generated/trained tokens and GPU-hours; wall time need not double exactly. A larger mixed-group fraction alone is not success: it can rise mechanically with n and must translate to accuracy or measured cost efficiency. Do not halve prompts to keep 2,048 completions and call it the same experiment.

### 4. Implement and validate before the next GPU runs

- [ ] Archive the A3a control manifest, exact launch command/resolved config, model/data/code hashes and result paths.
- [ ] Add the A4 schedule with meaningful boundary and update-index checks; verify its actual trainer path on a short smoke.
- [ ] Make each arm's override explicit in the launcher and resolved config; compare against A3a and reject unintended recipe changes. Keep training-only changes out of eval settings.
- [ ] Verify A5's probability convention at T=1, including raw/processed sampler log-probs and retained top-k/top-p filters. A training log-prob mismatch must not be mistaken for a useful exploration change.
- [ ] Generalize comparison/signal tools that currently assume 2,048 rows and groups of eight. For A6 require **4,096 rows, exactly 256 groups × 16**, consistent prompt/answer per group and unchanged ordered prompt membership. Add a valid n=16 case and negative incomplete/incorrect-group cases; do not disable identity checks.
- [ ] Confirm global token-mean loss normalization and one optimizer update across A6's full batch. Record gradient-accumulation/microbatch settings; more completions must not silently become multiple optimizer steps.
- [ ] Predeclare and implement the run's monitoring/stop rules before launch. Node/RoPE guard failures and non-finite numerics stop the run. Large sustained entropy/log-prob/cap-hit deterioration triggers investigation; thresholds must be labeled engineering screening rules, not a GPU/TPU parity gate. Preserve any aborted arm as incomplete/failed, with no step-100 result invented.
- [ ] Run the existing reward-scope/template-identity tests and the changed schedule/batch-contract checks. Keep code and manifests identical across nodes; record probe results.
- [ ] Launch each authorized arm from the initial checkpoint only after its preflight passes.

No production launch commands are promised in this initial draft: exact overrides will be added with the implementation, rather than presenting unsupported scheduler flags as runnable.

### 5. Judge accuracy, signal quality, stability and cost together

**Primary endpoint:** greedy held-out accuracy at **fixed step 100 vs A3a's 40.96%**. Report question-paired gained/lost counts and bootstrap 95% CI, with exact eval identity checks. Also show each arm's gain from its own step 0 and the complete six-point curve. Do not tune eval prompts, question membership or scorers after seeing results.

**Learning diagnostics:** sampled training accuracy and format separately from total reward; all-correct/all-wrong/mixed-correctness groups; zero-advantage and format-only groups; advantage-mass shares; response length/cap-hit rate; entropy, gradient norm, PPO clipping and trainer–sampler mismatch. Track whether the chosen arm makes correctness-bearing updates more useful, not only more numerous. Trainer–sampler `k3_kl` is not KL to the initial policy.

**Cost:** report time from the same launch boundary through completion of step-100 eval, GPU-hours, generated response tokens and trainer tokens. For steady throughput give full-window mean plus median/p90 with a declared warmup/eval exclusion policy. Plot accuracy against cumulative tokens and time as well as step. A6's twofold completion budget must be visible; fixed-step improvement alone does not establish better compute efficiency.

**Selection:** a positive paired interval, a consistent full curve and acceptable diagnostics nominate a candidate. If the interval includes zero, describe the result as inconclusive; a cheaper arm with comparable observed accuracy can still be recorded as a cost candidate, without claiming statistical equivalence. These intervals describe fixed trained models, not seed variance. Repeated recipe screening can overfit the validation set; freeze the candidate before fresh-seed confirmation and do not advertise a seed-1 screen as final generalization proof.

### 6. Keep larger algorithm changes and final validation separate

Do not escalate constant LR beyond 2e-6 after the A3b failure. REINFORCE-style loss + token TIS is a valid future algorithm hypothesis, but Qwen OMI2 did not require it. Such an experiment must specify the full objective, importance-weight detachment/truncation, log-prob sources and bypass path; it is not an equivalent single-flag change.

If A4–A6 leave a clear unresolved question, write the next single-factor hypothesis before running it (for example one sampling filter or a separately specified objective). Combining winners requires its own labeled run because benefits need not add.

After choosing a recipe, freeze it and evaluate the longer horizon and additional seeds. The existing customer protocol remains **step 200, three-seed mean, TPU ≥ GPU − 1 percentage point**. PR #12's 100-step seed-1 screens do not complete or redefine that gate.

## Result ledger

| Arm | Exact command/config and experiment ID | Validity / completed steps | Step-0 acc | Step-100 acc | Difference vs A3a, pp (paired 95% CI) | Tokens / GPU-hours / decision |
|---|---|---|---:|---:|---|---|
| A3a control | PR #11 run `gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753` | Complete: 100 steps, all six evals | 36.06% | 40.96% | Reference | Reuse recorded evidence; stability caveat |
| A4 decay | Not launched | Pending implementation/preflight | — | — | — | — |
| A5 T=1 | Not launched | Pending implementation/preflight | — | — | — | — |
| A6 n=16 | Not launched | Pending implementation/preflight | — | — | — | — |

Keep completed, null, failed and aborted arms in this ledger. Link commands, manifests, logs, per-question results and plots beside each result. Update the PR description around the final experiments as they are completed.

## Initial validation and scope

Documentation only: no runtime change or GPU job in this commit. The draft was checked against merged PR #11, the Qwen rulebook/guide/launcher, the existing Gemma comparison evidence and frozen eval protocol; relative document links were checked. No new runtime test is warranted for this plan-only commit; implementation checks are listed above.

This follows the completed PR #11 screen. No overlapping open experimental PR was found in this fork or the upstream `gemma4 ablation` search during closeout. AI assistance: Codex drafted the plan at the user's request; experiment implementation, human review and runtime evidence remain to be added.
