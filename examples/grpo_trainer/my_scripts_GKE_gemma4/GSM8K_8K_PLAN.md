# Gemma-4 E2B on GSM8K: colocated, 8K, no length penalty

## Gist

Establish a **Gemma-4 E2B GPU learning baseline on the GSM8K 8K/no-length-penalty recipe previously delivered for Qwen3-0.6B**. Reuse that task, reward, loss, batch and evaluation contract, with explicit adaptations for Gemma's tokenizer, chat template, stopping behavior and validated trainer fixes. First establish whether it learns; optimize only after the initial baseline is understood.

The historical Qwen reference used 64 GB200, 250 updates × three seeds, and full GSM8K test evaluation every 20 updates: **74.78% → 80.57% mean accuracy, approximately +5.8 pp**, with final seeds **82.26% / 79.61% / 79.83%**. It is a customer-delivered workload reference, not a numerical target for a different model. See the [historical 8K rulebook](../my_scripts_GKE_gsm8k/TPU_GPU_RL_Parity_Rulebook_gsm8k_8k.md) and [Qwen result consolidation PR #13](https://github.com/syangx39/verl/pull/13).

**Status: plan only.** No Gemma GSM8K run, new launcher, scorer validation or result is claimed by this initial PR. The model is `google/gemma-4-E2B-it`, not a generic Gemma 4B checkpoint. This is a separate experiment from the OMI2 A4–A6 optimization in [PR #12](https://github.com/syangx39/verl/pull/12); no OMI2 result, eval protocol or experiment configuration is overwritten.

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

- [ ] `run_gemma4_e2b_gsm8k_8k.sh`: dedicated launcher/config using existing Gemma node prep and historical GSM8K loss/reward settings; isolate OMI2 environment variables and output directories.
- [ ] `gemma4_gsm8k_preflight.py`: model/data/template/scorer/config checks, global-batch/update/schedule validation and guard evidence collection.
- [ ] `gsm8k_8k/`: data/model/code manifest, Gemma prompt fixture, scorer results, resolved configs and exact launch commands.
- [ ] `GSM8K_8K_RESULTS.md`: seed1 fixed-step results and decision, followed by seeds2/3 only after freeze; report full curves and failed runs.
- [ ] Extend comparison/plotting tools to accept the 128×16 batch, boxed scorer fields, full GSM8K eval IDs and explicit cross-model template differences.
- [ ] Reproducible handoff: immutable model/tokenizer/processor and code/image identity, ordered training prompts, full eval dumps, train diagnostics, checkpoints and SHA256 manifest; model-specific TPU guide.

## 6. Current progress

- [x] Select the customer-delivered GSM8K colocated8K/no-length-penalty workload as the next model-transfer experiment.
- [x] Record the initial recipe, Gemma adaptations, fixed endpoint and staged decisions.
- [ ] Implement/validate the dedicated launcher and preflight.
- [ ] Run and archive the short smokes.
- [ ] Review seed1 at step100, then fixed step250 if continuing.
- [ ] Freeze or revise the candidate; decide on seeds2/3.
- [ ] Produce the final GPU baseline and TPU handoff.

Initial source review: historical Qwen 8K rulebook/scorer, merged Gemma README / precision and RoPE reports, and current OMI2 optimization scope. No GPU test or launch is included in the plan-only change.
