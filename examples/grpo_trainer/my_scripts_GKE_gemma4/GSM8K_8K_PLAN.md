# Gemma-4 E2B on GSM8K: colocated, 8K, no length penalty

## Gist

Establish a **Gemma-4 E2B GPU learning baseline on the GSM8K 8K/no-length-penalty recipe previously delivered for Qwen3-0.6B**. Reuse that task, reward, loss, batch and evaluation contract, with explicit adaptations for Gemma's tokenizer, chat template, stopping behavior and validated trainer fixes. First establish whether it learns; optimize only after the initial baseline is understood.

The historical Qwen reference used 64 GB200, 250 updates × three seeds, and full GSM8K test evaluation every 20 updates: **74.78% → 80.57% mean accuracy, approximately +5.8 pp**, with final seeds **82.26% / 79.61% / 79.83%**. It is a customer-delivered workload reference, not a numerical target for a different model. See the [historical 8K rulebook](../my_scripts_GKE_gsm8k/TPU_GPU_RL_Parity_Rulebook_gsm8k_8k.md) and [Qwen result consolidation PR #13](https://github.com/syangx39/verl/pull/13).

**Status: launcher, preflight and a 64-GPU functional smoke are done (§7). The seed-1 run waits on a scorer decision (§7.3):** the frozen boxed scorer marks 36 pp of Gemma's correct step-0 answers wrong only because of `\$` / units inside the box. The model is `google/gemma-4-E2B-it`, not a generic Gemma 4B checkpoint. This is a separate experiment from the OMI2 A4–A6 optimization in [PR #12](https://github.com/syangx39/verl/pull/12); no OMI2 result, eval protocol or experiment configuration is overwritten.

## Plan

1. Freeze the GSM8K recipe below and build Gemma-native prompt fixtures from the existing boxed data.
2. Reuse Gemma node preparation, FP32/RoPE guards and attention fixes; verify the new reward and single-forward TIS path with a short smoke.
3. Start seed 1 from the initial model with the **250-update schedule**, review the complete 0–100 curve, and continue to the fixed step-250 endpoint if the run is valid and informative.
4. Decide whether the recipe is worth freezing before scheduling seeds 2/3. If it underperforms, preserve the evidence and make any subsequent recipe change a separate arm.
5. Package a model-specific GPU reference and TPU reproduction instructions after the candidate is established.

## 1. Initial recipe contract

The first arm transfers the successful **GSM8K** recipe. It does not combine Gemma's OMI2 A3a settings with selected pieces of the Qwen recipe.

| Item | Initial Gemma GSM8K recipe | Reason / verification |
|---|---|---|
| Model | `google/gemma-4-E2B-it`, original post-trained checkpoint | Pin checkpoint/config/tokenizer/processor hashes and revision; never initialize from OMI2-trained weights |
| Topology | Colocated, 16 nodes × 4 GB200 = 64 GPUs; rollout TP=1 | Same broad resource topology as Qwen; use the current validated Gemma stack |
| Train / eval data | GSM8K train 7,473 / full test 1,319 | Reuse boxed data question IDs and gold answers; no OMI2 or test512 substitution |
| Prompt batch / group size | 128 prompt instances × 16 completions = 2,048 responses | Do not inherit OMI2's 256×8 |
| Updates | One optimizer update per full batch; one epoch per batch | Microbatches only accumulate gradients; verify global normalization |
| Advantage | GRPO group normalization, sample std + `1e-6` | Preserve zero-advantage groups in the global token denominator |
| Loss | REINFORCE-style policy gradient + detached **token TIS**, upper cap 3 | `bypass_mode=true`, `loss_type=reinforce`; no PPO/dual clipping in this objective |
| KL / entropy regularizer | Both zero | Log entropy as a diagnostic; do not add it to loss |
| Optimizer | AdamW, peak LR `2e-6`, β=(0.9,0.999), ε=`1e-8`, wd=0, global grad clip=1 | This is the historical GSM optimizer, not the OMI2 constant-LR/wd=0.01 configuration |
| LR schedule | Warmup 10 updates, cosine to 0 over 250 | Check update-used LR separately from logged post-scheduler LR |
| Precision | FP32 persistent/master weights and Adam moments; BF16 compute / rollout | Keep the proven Gemma precision fix and FP32 RoPE buffers |
| Attention / packing | Existing validated Gemma `flex_attention` path and patches | Node preparation and per-rank RoPE checks are mandatory |
| Train sampling | T=1, top_p=1, top_k=-1, n=16 | No top-k/top-p truncation; consistent trainer/sampler probability convention |
| Response cap | **8,192 generated response tokens** for both train and eval | Not prompt+response=8K; no length penalty or truncated-response masking |
| Prompt limit | Start at historical 512, verify every Gemma-rendered prompt | No silent truncation/filtering; if any exceeds it, explicitly revise the limit before freezing and report the deviation |
| Reward | Existing GSM8K boxed scorer: correct=1, parseable wrong=0.1, unparseable=0 | Exclusive categories, not additive `acc + 0.1·fmt`; reuse scorer fixtures |
| Length penalty | Disabled for train and eval | `REWARD_PENALTY_SOURCES` empty; record zero penalty on every sampled row |
| EOS / generation prompt | Gemma-native settings, frozen in manifest | Do not copy Qwen token IDs, `<think>` tokens or Qwen-specific template switches |
| Eval | Greedy, T=0, top_p=1, top_k=-1, n=1; full 1,319 questions | Steps 0,20,…,240,250; report raw answer accuracy separately from reward/format |
| Checkpoints / evidence | Save every 50 steps and final; full train/eval dumps, TensorBoard, timing and resolved config | Enables step-100 review, paired statistics and later TPU handoff |

Gemma's dynamic token microbatch limit can start at its existing conservative **16,384 tokens/GPU**, rather than Qwen's static microbatch layout. This is a memory implementation choice: preserve the 2,048-response batch, one optimizer update and global valid-token denominator. Record the value and any memory-only change; do not shrink the global batch to make a smoke pass.

### Loss contract

For a sampled response token $t$ in completion $i$, response mask $m_{it}$ and detached GRPO advantage $A_i$:

$$
w_{it}=\operatorname{stopgrad}\left[\min\left(\exp\left(\operatorname{clip}(\log\pi_{\theta,it}-\log\pi_{\mathrm{sampler},it},-20,20)\right),3\right)\right],
\qquad
L_{\rm total}=L_{\rm PG}=-\frac{\sum_{i,t}m_{it}w_{it}A_i\log\pi_{\theta,it}}{\sum_{i,t}m_{it}}.
$$

Both log-probabilities describe the same response tokens at T=1. The sampler values are recorded during rollout; current trainer values are computed before the batch's update. Gradients flow through `log πθ`, not through weights or advantages. EOS/masks and capped responses must follow the historical token-inclusion contract.

The configuration intent to validate in the installed verl path is:

```yaml
algorithm:
  adv_estimator: grpo
  rollout_correction:
    bypass_mode: true
    loss_type: reinforce
    rollout_is: token
    rollout_is_threshold: 3.0
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: bypass_mode
    loss_agg_mode: token-mean
    use_kl_loss: false
    entropy_coeff: 0.0
```

This fragment is **not a complete runnable config**. Implementation must validate the resolved keys, disable IS normalization/rejection, prove this objective is the active code path, and check that an old-log-prob trainer forward is not inadvertently retained. Merely toggling `bypass_mode` in the OMI2 PPO launcher is not sufficient.

## 2. Data, prompt and scorer adaptation

Reuse the archived GSM8K train/test question and answer payloads and their stable IDs. Verify 7,473 train and 1,319 test questions, uniqueness, disjoint question identities, and checksums. Training may repeat questions across epochs, but each step must record its actual 128 prompt instances and 16 completions per group. Prefer the archived Qwen per-seed prompt order when available; otherwise export and freeze the newly generated order rather than claim it is identical.

Preserve the semantic boxed-answer instruction, including “Please reason step by step, and put your final answer within `\boxed{}`.” Render it **once** with the Gemma processor/chat template used by the validated rollout path. If the template does not accept a separate system role, fold that content into the user message explicitly and record the adaptation. Do not insert pre-rendered Qwen/Gemma chat tokens inside another chat template or carry over OMI2's `<reasoning>/<answer>` wrapper.

Create fixtures containing raw question ID, gold answer, messages, rendered text and token IDs. Inspect short/long examples, check the full prompt-length distribution, and compare fixture IDs against online rollout input IDs. Qwen and Gemma token IDs are expected to differ; question/gold identity and intended instructions must remain consistent. **Do not run the historical Qwen data builder's `--model_out` operation on Gemma:** that path writes Qwen-specific EOS IDs. Preserve Gemma's own model/generation configuration.

Reuse [boxed_math_reward.py](../my_scripts_GKE_gsm8k/boxed_math_reward.py), rather than swapping in `maxtext_math_reward.py` or math-verify. Test correct/incorrect/missing/multiple/nested boxes, number normalization and capped responses. Correct output must receive **1.0, not 1.1**; a parseable wrong box receives 0.1; length penalty is zero at all lengths, including 8,192. Reuse tokenizer-independent reward fixtures but regenerate Gemma-specific token/length/EOS fixtures. Log `acc`, `fmt`, raw/final score, response length and cap-hit separately.

## 3. Staged execution and decisions

| Phase | Work | Evidence needed before proceeding |
|---|---|---|
| Preparation | New dedicated Gemma GSM8K launcher/config; data/prompt/scorer checks; runtime/model manifest | All recipe fields explicit; no inherited OMI2 variables; actual Gemma import/model path; resolved configuration archived |
| Functional smoke | Two updates, initially 4 nodes / 16 GPUs if convenient, same global batch and 250-step LR horizon | Node/RoPE checks on every participating rank; correct full-batch grouping; finite loss/gradients/TIS; valid sampler log-probs; second update persists a nonzero parameter change |
| Full-topology smoke | 64-GPU short run, up to 20 updates, same recipe/schedule | No OOM, correct weight sync and single-forward path, diagnostics/dumps complete, memory and timing measured; no quality conclusion from this smoke |
| Seed 1 baseline | Fresh initial checkpoint, schedule horizon 250, full eval every 20 | Review fixed step 100 and complete earlier curve; continue unchanged to 250 if technically valid and informative |
| Freeze decision | Review seed-1 step250 and cost | Decide whether to freeze or make a separately named revision; no automatic seeds2/3 based only on a smoke pass |
| Multi-seed / handoff | Fresh seeds2/3 only after recipe choice | Three complete 250-step curves; per-question dumps, spread, quality/cost summary and reproducible package |

**Schedule indexing:** update 1 uses LR 0, update 2 uses `2e-7`, update 11 uses peak `2e-6`. The logged value may instead be the next update's LR; the post-update-250 schedule value is zero. Export/check the entire schedule against the implementation. A two-step smoke must exercise a nonzero update and cannot certify correctness just from its zero-LR first step.

**The step-100 review does not change the scheduler horizon to 100.** Run the first 100 updates of the intended 250-update schedule. If paused, resume only with optimizer, scheduler, RNG and data-loader state preserved; otherwise restart from the initial checkpoint and label it as a new run. Do not splice a completed 100-step cosine into a longer curve. Short smokes are separate runs and are not pooled into the baseline.

A single-seed screen answers whether this recipe is promising, not whether the model universally learns less than Qwen. Inspect absolute accuracy and gain from Gemma's own step 0, paired correct/incorrect flips, the entire curve, and format/length/signal diagnostics. If Gemma starts near the task ceiling, smaller gain can be expected. No requirement to reach Qwen's 80.57% or reproduce its +5.8 pp gain is imposed.

### Numerical validity and monitoring

Reuse [Gemma node preparation and RoPE guards](ROPE_GUARD.md) and the completed [log-prob parity investigation](LOGPROB_PARITY.md). This experiment needs an integration check of those fixes with the new GSM8K/TIS path, not a claim that all previous debugging must be repeated.

- Inspect trainer-vs-sampler `k3_kl`, signed log-prob difference, TIS mean/clipped fraction/ESS and finite gradient norms. Compare the same tokens, temperature and masks. Use a small frozen-batch check if the new path behaves unexpectedly; do not use a universal loss-percentage threshold or require Qwen's exact k3 value.
- Verify `sum(valid response tokens)` is the global full-batch denominator, including zero-advantage groups; dynamic microbatch changes must not reweight sequences. Check actual optimizer-step count.
- Persist the first nonzero parameter update and optimizer precision evidence. Successful loading, a finite scalar loss or matching gradient norms alone do not establish update correctness.
- Monitor complete per-step dumps and promptly flushed trainer metrics. Missing or partial evidence is unknown, never a measured zero. Preserve failure logs and the exact config when aborting.
- Fail fast on failed guards, non-finite loss/gradients/log-probs, corrupt/incomplete batch identity or a failed update. Sustained reward/format collapse or rapidly growing mismatch triggers a pause and diagnosis. Any automated numerical thresholds must be written and validated before an unattended run; this plan does not silently reuse OMI2's `</answer>`-based stop rule for boxed outputs.

If 250-step seed 1 is uninformative or unstable, do not spend two more seeds merely to complete a table. First distinguish scorer/template failures, lack of mixed-reward groups, truncation, numerical drift and genuine optimization behavior. Change one factor per follow-up arm and preserve the first baseline, including unsuccessful outcomes.

## 4. Evaluation and required figures

Freeze a **new GSM8K protocol** with the full 1,319 questions and 8K greedy response cap. It does not replace Gemma's existing OMI2 `eval_5000` / step-200 protocol.

| Figure / table | Content | Interpretation |
|---|---|---|
| Held-out accuracy vs update | Gemma step0/20/…/250, each seed individually; mean/spread once available; Qwen8K curves as labeled context | Fixed endpoint, no best-checkpoint selection; compare model-specific gains |
| Training diagnostics | Sampled correctness and actual reward, format rate, response length/cap-hit, entropy, gradient norm | Reward is not accuracy; train T=1 is not greedy eval |
| Group signal | Full 128×16 batches: reward-std-zero, all-correct/all-wrong/mixed-correctness groups | Preserve distinct prompt-instance IDs, including repeated questions |
| Numerical diagnostics | Trainer/sampler k3, log-prob tails where available, TIS clipped fraction/ESS, precision/guard status | Debugging evidence, not a replacement for learning results |
| Quality vs cost | Accuracy vs elapsed time, cumulative generated tokens and GPU-hours | Different models/tokenizers have different realized token cost |
| Timing | Steady arithmetic mean, median, p90 and full end-to-end duration; component breakdown | Report measurement window and eval/checkpoint overhead explicitly |

For the final 250-step timing table, reuse the historical **steps20–250 excluding eval/checkpoint steps** (216 updates), while also reporting full launch-to-finish duration. Collect coordinator completion timestamps for a consistent elapsed-per-update measure; distinguish this from existing `timing_s/step` if their boundaries differ. Do not infer end-to-end speedup from a median or add component medians.

Archive raw per-question eval outputs at all 14 checkpoints. At each checkpoint report accuracy and a paired interval for change from that run's own step0; with three seeds also report seed-wise results and between-seed spread. Question-paired intervals alone do not include training-seed uncertainty. Matching question IDs permits comparisons, but model-specific rendered prompts/tokenization are deliberately different and must not be rejected by a same-input-token assertion.

The customer **TPU ≥ GPU − 1 percentage point** rule would compare Gemma against Gemma on an agreed frozen GSM8K protocol. Confirm endpoint, seed aggregation and uncertainty reporting with the customer before applying it here; do not silently transplant OMI2's step200 gate or the old Qwen empirical envelope to a new model/task baseline.

## 5. Implementation checklist and deliverables

Proposed filenames below are work to implement, not existing runnable commands:

- [x] `run_gemma4_e2b_gsm8k_8k.sh`: dedicated launcher/config using existing Gemma node prep and historical GSM8K loss/reward settings; isolate OMI2 environment variables and output directories. — See §7.1.
- [x] `gemma4_gsm8k_preflight.py`: model/data/template/scorer/config checks, global-batch/update/schedule validation and guard evidence collection. — Plus the runtime-config contract [`gsm8k_8k/runtime_config_asserts.txt`](gsm8k_8k/runtime_config_asserts.txt), checked with [`gemma4_config_diff.py`](gemma4_config_diff.py) `--assert`.
- [ ] `gsm8k_8k/`: data/model/code manifest, Gemma prompt fixture, scorer results, resolved configs and exact launch commands. — Smoke evidence is in [`gsm8k_8k/smoke/`](gsm8k_8k/smoke/), and the run launcher is [`gsm8k_8k/launch/g4run_gsm8k.sh`](gsm8k_8k/launch/g4run_gsm8k.sh). The model/data manifest for the baseline is still to come.
- [ ] `GSM8K_8K_RESULTS.md`: seed1 fixed-step results and decision, followed by seeds2/3 only after freeze; report full curves and failed runs.
- [ ] Extend comparison/plotting tools to accept the 128×16 batch, boxed scorer fields, full GSM8K eval IDs and explicit cross-model template differences.
- [ ] Reproducible handoff: immutable model/tokenizer/processor and code/image identity, ordered training prompts, full eval dumps, train diagnostics, checkpoints and SHA256 manifest; model-specific TPU guide.

## 6. Current progress

- [x] Select the customer-delivered GSM8K colocated8K/no-length-penalty workload as the next model-transfer experiment.
- [x] Record the initial recipe, Gemma adaptations, fixed endpoint and staged decisions.
- [x] Implement/validate the dedicated launcher and preflight (§7.1).
- [x] Run and archive the short smokes (§7.2). The functional smoke ran on the full 64-GPU topology; a separate up-to-20-update smoke is optional (§7.4).
- [ ] **Decide the scorer handling (§7.3)**, then start seed 1.
- [ ] Review seed1 at step100, then fixed step250 if continuing.
- [ ] Freeze or revise the candidate; decide on seeds2/3.
- [ ] Produce the final GPU baseline and TPU handoff.

Initial source review: historical Qwen 8K rulebook/scorer, merged Gemma README / precision and RoPE reports, and current OMI2 optimization scope. No GPU test or launch is included in the plan-only change.

## 7. Implementation and functional smoke (2026-10-24)

### 7.1 Implementation

- **[`run_gemma4_e2b_gsm8k_8k.sh`](run_gemma4_e2b_gsm8k_8k.sh)** combines the Gemma OMI2 line's system settings with the §1 recipe.
  - Gemma system settings: node prep and RoPE guard at every launch, FP32 weights and Adam, flex attention, dynamic micro-batching at 16,384 tokens per GPU.
  - Data and model: the handoff package's `gsm8k_boxed_{train,test}.parquet`, whose sha256 matches `PACKAGE_MANIFEST.sha256`, and the original `gemma-4-E2B-it`.
  - Batch and loss: 128 × 16, single-forward REINFORCE + token TIS 3.0 (`bypass_mode`, `loss_type=reinforce`, `loss_mode=bypass_mode`), no IS normalization or rejection, token-mean.
  - Optimizer: AdamW 2e-6, weight decay 0, clip 1, warmup 10 + cosine over 250.
  - Sampling and eval: T=1, top-p 1, top-k −1, 8,192 cap; greedy full-1,319 eval every 20 steps.
  - Reward: boxed scorer with `REWARD_PENALTY_SOURCES` empty. The OMI2 reward variables are unset.
- **[`gemma4_gsm8k_preflight.py`](gemma4_gsm8k_preflight.py)** runs before node prep at every launch, without GPUs:
  - data counts, IDs and overlap, and hashes against the handoff manifest;
  - every prompt rendered by the Gemma template exactly as verl does, checked for length and form;
  - the scorer: Meta's fixtures, plus the run settings without penalty;
  - the full LR schedule.
- **[`gsm8k_8k/launch/g4run_gsm8k.sh`](gsm8k_8k/launch/g4run_gsm8k.sh)** runs the modes `SMOKE` and `SEED1..3`. `SMOKE` is the production run, including the 250-step horizon, stopped after update 3 by a watcher that SIGTERMs only its own driver. It also runs the optimizer probe for the first 3 AdamW steps.

### 7.2 Functional smoke (64 GB200)

Run `gemma4_e2b_gsm8k8k_smoke_16n64g_s250stop3_seed1_1024_0036`. Evidence is in [`gsm8k_8k/smoke/`](gsm8k_8k/smoke/).

| Check | Result |
|---|---|
| Preflight | PASS: 7,473 / 1,319 rows; IDs unique and disjoint; hashes match the handoff. Rendered prompts are at most 249 / 225 tokens, against a 512 limit. Scorer: Meta fixtures PASS; correct 1.0, wrong-boxed 0.1, unparseable 0, penalty 0 |
| System turn | Gemma's template accepts it natively: `<bos><\|turn>system\n…<turn\|>\n<\|turn>user\n…<turn\|>\n<\|turn>model\n`. No folding is needed |
| Stop tokens | Gemma-native `generation_config` ids 1 `<eos>`, 106 `<turn\|>`, 50 `<\|tool_response>`; no Qwen ids |
| Runtime config | All 54 contract keys hold (`runtime_config_check.json`), including the rollout correction in both `algorithm` and the worker `policy_loss`. Rank 0 printed `Total steps: 250, num_warmup_steps: 10` |
| Single-forward path | No `old_log_prob` phase. The TIS diagnostics come from the update pass (`actor/rollout_corr/*`). At step 2: IS mean 1.00001, max 1.55, fraction above 3.0 = 0, ESS 0.9992, trainer–sampler KL 3.7e-4 |
| Batch | Every dump has 2,048 rows = 128 groups × 16. The step 1–3 prompt sets are **identical to Qwen seed 1's archived order** (`train_order_seed1.parquet`) |
| LR | Logged post-update values 2e-7, 4e-7, 6e-7 at steps 1–3, i.e. update 1 used 0 |
| Parameter updates (optimizer probe, 64/64 ranks) | Update 1 (LR 0) changed 0 elements. Update 2 (LR 2e-7) changed 42.5% of elements, with ‖applied‖/‖ideal fp32‖ = 1.0001 and cosine 1.0000. AdamW eps 1e-8, weight decay 0, FP32 parameters and moments |
| Guards | RoPE guard PASS on 64 ranks. The guard check's only FAIL is `exit_code=0`, expected because the smoke was stopped by design |
| Length and reward | Response mean about 470–500 tokens, max 3,777, cap-hit 0, length penalty 0 on every row |
| Cost | Step 2–3 took about 30–33 s each (generation 16 s, update 8–12 s, weight sync 4 s), and the step-0 eval about 2–3 min. Peak actor memory 29.4 GB allocated / 57.2 GB reserved |

### 7.3 Scorer finding: decision needed before seed 1

At step 0 (greedy, full 1,319 questions), Gemma's accuracy under the frozen `boxed_math_reward.py` is **44.81%**. Gemma usually writes a correct number with formatting inside the box (for example `\boxed{\$18}`, `\boxed{540 \text{ meters}}`, `\boxed{60\%}`), and the scorer counts those as wrong. Re-scoring the same dumps with a lenient normalization that strips `\$`, `$`, `\text{…}` units, `\%` and `^\circ` before the numeric comparison gives **80.67%** ([audit](gsm8k_8k/smoke/scorer_audit_gemma_step0.txt)).

| Category of Gemma's step-0 answers | Share |
|---|---:|
| Correct under the frozen scorer | 44.81% |
| Correct after removing `\$` / `$` from the box | 21.61% |
| Correct after removing `\text{units}` | 12.81% |
| Correct after removing `\%` / `^\circ` | 1.44% |
| No `\boxed` at all | 12.29% |
| Boxed and genuinely wrong | 7.05% |

The same lenient normalization barely moves Qwen's archived seed-3 eval: 75.82% → 76.35% at step 0, and 79.83% → 79.83% at step 250 ([audit](gsm8k_8k/smoke/scorer_audit_qwen_seed3_step0.txt)). So the frozen scorer treats the two models' answer formats very differently.

Under the frozen scorer, a correct answer with `\$` earns 0.1 instead of 1.0. Training therefore rewards dropping `\$` and units inside the box, and the smoke's 60–69% correctness-mixed groups partly reflect this formatting signal. The reported gain would then mix format conformance with math improvement.

Options:

- **(A) Keep the frozen scorer**, the strict `gsm8k_8k_v1` contract. Report a lenient answer-accuracy column at every eval, computed offline from the dumps, so that format adaptation and reasoning gains can be separated.
- **(B) A Gemma-aware normalization in the scorer**, for both training and eval. This is a new scorer version: strip only `\$`, `$`, `\text{…}`, `\%` and `^\circ` inside the box, with fixtures. Re-scoring Qwen's archived dumps would show that the Qwen reference curves are essentially unchanged.
- **(C) A prompt change**, for example "put only the final number in `\boxed{}`". This changes the prompt contract, and is the riskier option.

### 7.4 Next steps

- **Before seed 1:** the scorer decision (§7.3) and the checkpoint policy. A full checkpoint (FP32 weights plus Adam m, v) is about **61.5 GB** for 5.12B parameters, so `SAVE_FREQ=50` writes about 307 GB over 5 saves.
- **Seed-1 cost estimate:** 250 updates at about 30–35 s, plus 14 evals at about 2–3 min, roughly 3 h before checkpoint time. That is an estimate from two steps; the run itself will give the measurement.
- **Automated stopping:** the OMI2 stop monitor's `</answer>` and A3a rules do not apply here. Seed 1 runs with verl's own failure behaviour plus periodic checks: non-finite metrics, guards, IS fraction above 3, unparseable rate and held-out vs step 0. GSM8K-specific automated thresholds would need to be written and validated first.
