# Gemma-4 FP32 recipe ablation plan (PR #11)

## Gist

Optimize the already-learning Gemma-4 E2B FP32 recipe by isolating the effects of training format reward, prompt wrapping, and learning-rate magnitude. Use independent 100-step seed-1 screens against PR #10's fixed steps 0–100, then freeze a candidate before multi-seed validation.

PR #10 established learning, not a plateau: held-out accuracy rose **35.76% → 39.62% (+3.86 pp)** at step 100, with question-paired bootstrap 95% **[+2.82, +4.96] pp**. Its run was deliberately stopped at step 107; only steps 0–100 are the comparison reference. Qwen results motivate optimization but are not a Gemma acceptance threshold.

**Status:** A1 (reward) is prepared, checked and running; see the [execution record](#execution-record). A2 and A3 have not started. The initial commit was plan only.

## Plan

### 1. Freeze the comparison reference

- [x] Anchor the plan to merged PR #10, commit `11c9486ecbdd2f47222a2b38db6d9292961db7c6`.
- [x] Reference run: `gemma4_e2b_rl05fp32_16n64g_s200_seed1_1022_2019`; use only steps 0–100.
- [ ] Record each arm's command, resolved configuration, code/launcher/reward hashes, model identity, dataset hashes and experiment ID before reading its result.
- [ ] Verify the only intended recipe difference from the reference; keep the reference files and evidence unchanged.

Reference evidence: [precision results §4](UPDATE_PRECISION_RESULTS.md#4-fp32-seed-1-steps-0100-plan-item-4--stopped-at-step-107), [100-step artifacts](update_precision/fp32_seed1_100steps/), and [frozen evaluation protocol](EVAL_PROTOCOL.md).

### 2. Run independent single-factor screens in this order

| Arm | Only intended change vs PR #10 FP32 reference | Explicitly unchanged |
|---|---|---|
| A0: reference | None; reuse PR #10 steps 0–100 | Original FP32 recipe |
| A1: reward | Training format coefficient **0.1 → 0** | Correctness scorer/extraction/normalization, format detection, length penalty, prompt, sampling, LR and eval |
| A2: prompt | Remove the embedded inner turn-template wrapper from **training prompts only**; apply the Gemma chat template once | Raw question text/order/IDs, answer, instruction wording and requested answer format, reward coefficient **0.1**, LR and eval prompts |
| A3: LR | Change only LR magnitude from **1e-6**; record the candidate value(s) before launch | Constant schedule with no warmup, original reward **0.1**, original double-wrapped prompt, all other settings |

**Every arm starts fresh from the same initial checkpoint, with seed 1 and reset optimizer state.** Do not resume the step-100 checkpoint or carry over the previous arm's changes. A2 does not inherit A1's reward change; A3 does not inherit A1 or A2. Any later combined recipe is a separately labeled experiment, not one of these single-factor arms.

Common settings: `16n64g`; FP32 persistent optimizer weights and moments, BF16 compute/rollout; 256 prompts × 8 completions; one optimizer update per rollout batch; unchanged GRPO/DAPO objective and clipping, no KL; prompt/response caps 8192/8192; sampling T=0.8, top-k=50, top-p=0.95; unchanged attention, node prep and RoPE guards. Preserve seed settings and actual per-step question membership/order.

All screens use **100 training steps**, eval at **0, 20, 40, 60, 80, 100**, and the existing 5,000-question greedy evaluation. Set the experiment name to identify the arm and `s100`; the existing wrapper's default name still contains `s200`.

### 3. Prepare each changed path before its long run

**A1 — reward**

- [ ] Make the format-coefficient change explicit in the actual Ray reward workers; a head-shell environment variable alone is not sufficient evidence. — implemented (see [A1 record](#a1-record-training-format-coefficient-0)); the check in the run's own train/eval dumps is pending.
- [x] Preserve `acc` and binary `fmt` reporting. Check representative completions: training score changes from `acc + 0.1 * fmt` to `acc`, while correctness and format flags stay unchanged; length penalty remains disabled. — [`test_maxtext_reward_fmt_scope.py`](test_maxtext_reward_fmt_scope.py) PASS.
- [x] Keep eval grading unchanged: `acc`, `fmt`, and canonical `score = acc + 0.1 * fmt`. The current scorer reads `REWARD_FMT_WEIGHT` process-wide, so a global zero override would also change eval total reward. Scope the ablation to training, and verify that eval keeps its frozen coefficient. — scoped by eval-question membership; tested; the check in the run's validation dumps is pending.
- [x] Check both correct/incorrect and formatted/unformatted cases before the 100-step run. — all four cases, for both training and eval rows.

Removing format credit mechanically lowers total reward. Compare accuracy and format rate separately; do not treat a lower total reward as a learning regression.

**A2 — prompt**

- [ ] Build a separate training-data artifact; preserve the reference parquet. Remove only the known inner template markers and fake model header, retaining the original instructions, question and requested response format.
- [ ] Preserve row IDs/order and ground truths. Save before/after rendered samples and token counts; check no unexpected truncation, row filtering or question-content changes.
- [ ] Extend [`gemma4_ab_compare.py`](gemma4_ab_compare.py) with an explicit training-template-change comparison mode. Its current strict `qid -> (rendered input, ground truth)` check correctly rejects this arm.
- [ ] In that mode, validate original row/question identity and raw question/answer hashes against a provenance mapping; validate complete steps, 2,048 rows, 256 groups × 8 completions and within-group consistency. Allow only the declared rendered training-template change; do not merely disable input checks or trust `qid` alone.
- [ ] Keep strict rendered-input equality for eval and for reward/LR comparisons. Include a positive expected-template-change check and negative mismatched-question/answer checks.
- [ ] Label training plots “same raw questions, different training templates,” not “identical prompts.”

Eval prompt tokens remain frozen for this screen. This measures the effect of changing training prompts under the existing eval condition. A future experiment that also changes eval prompts requires a separately versioned protocol and a re-evaluated reference; it must not be mixed into this comparison.

**A3 — LR**

- [ ] Predeclare LR candidate value(s) in the result ledger; each value is its own fresh run.
- [ ] Confirm resolved and logged LR match the intended constant value, with no warmup or scheduler change.
- [ ] Restore A0 reward and prompt settings before launching.

### 4. Report all arms on the same basis

For every completed arm:

- **Primary screen endpoint:** held-out greedy accuracy at fixed **step 100** vs A0 at step 100, paired by validated question identity. Also report each run's change from its own step 0, paired bootstrap intervals and gained/lost counts.
- **Full trajectory:** all six eval checkpoints, format rate, and training accuracy in the same 20-step windows. Do not substitute the best checkpoint for step 100.
- **Learning diagnostics:** mixed-correctness groups, all-correct/all-wrong groups, zero-advantage groups, format-only signal, response length/cap-hit rate, entropy, gradient norm and trainer–sampler log-prob metrics.
- **Cost:** accuracy vs elapsed time from a consistently defined launch boundary, total time through step-100 eval, and step-time mean/median/p90 on the same declared window with eval/checkpoint inclusion stated.
- **Validity:** successful completion, node/RoPE guards, finite diagnostics, complete train/eval dumps and data-identity checks. Distinguish failed/incomplete runs from valid negative ablations.

Question-paired intervals do not measure training-seed uncertainty. Consecutive training steps are not independent replicates. Read marginal differences against PR #10's same-recipe repeat evidence; seed-1 screens nominate candidates rather than establish a final winner.

### 5. Select, freeze and validate

- [ ] Complete and record the reward screen before advancing to prompt, then LR; keep the independent A0 comparison even if an earlier arm improves.
- [ ] Choose a candidate based on fixed-step accuracy, the full trajectory, stability and cost. Document the selection without using Qwen's gain as a pass/fail threshold.
- [ ] If combining useful changes, run and label the combined recipe separately; single-factor results do not establish that their benefits add.
- [ ] Freeze the selected recipe and validate with multiple seeds before claiming the final GPU reference.
- [ ] Keep every change reproducible on TPU. The existing customer protocol remains **step 200, three-seed mean, TPU ≥ GPU − 1 pp** unless explicitly revised before final validation. These 100-step screens do not complete or redefine that gate.

## Result ledger (update as experiments finish)

| Arm | Experiment ID / exact override | Run validity | Step-0 acc | Step-100 acc | Δ vs A0 at step 100 (paired 95%) | Time / decision |
|---|---|---|---:|---:|---|---|
| A0 | PR #10 FP32 seed 1, steps 0–100 | Reported window verified; full run stopped at 107 | 35.76% | 39.62% | Reference | See PR #10 |
| A1 | `gemma4_e2b_rl05fp32_a1fmt0_16n64g_s100_seed1_1023_0058`: `REWARD_FMT_WEIGHT_TRAIN=0` + `REWARD_EVAL_QUESTIONS` (eval keeps 0.1) | Running (started 2026-10-23T00:58Z) | — | — | — | — |
| A2 | Pending: training template applied once | Not run | — | — | — | — |
| A3 | Pending: predeclared constant LR value(s) | Not run | — | — | — | — |

Add reproducible launch/config evidence, per-question/per-step summaries and plots alongside each result. Preserve failed or inconclusive observations rather than deleting or pooling them.

## Execution record

### A1 record: training format coefficient 0

**Why the scorer needed a new knob.** `compute_score` receives the same fields for training and validation rows: `data_source` is `openmathinstruct2_maxtext` in both parquets, and `extra_info` is `{question, index}` in both. Validation indices also overlap training indices. verl tracks `validate` in the agent loop, but the pinned reward loop does not pass it to the scorer. A process-wide `REWARD_FMT_WEIGHT=0` would therefore also change the eval score.

**Implementation** ([`maxtext_math_reward.py`](maxtext_math_reward.py), sha256 `d0bde7b8…`). With no new environment variables the behavior is unchanged. Two variables enable the scope:
- `REWARD_FMT_WEIGHT_TRAIN`: the format coefficient for training rows.
- `REWARD_EVAL_QUESTIONS`: points to [`eval_5k/eval_5000_question_sha256.json`](eval_5k/eval_5000_question_sha256.json), the sha256 of every `eval_5000` question.

A row whose `extra_info["question"]` hashes into that set keeps the canonical 0.1; every other row gets the training coefficient. Each row also records which coefficient it got (`fmt_coef`) and whether it was treated as an eval row (`eval_scope`), so the train and validation dumps show the coefficient each row received. The scorer fails at import if the training knob is set without the manifest, or if the manifest's count disagrees with its hashes. It fails per row if `extra_info` has no question.

**Pre-run checks** (on the Ray head pod, training image):

| Check | Result |
|---|---|
| Question manifest ([`gemma4_eval_question_hashes.py`](gemma4_eval_question_hashes.py)) | 5,000 unique questions; source sha256 `d092a9c5…` equals the frozen `eval_5000.parquet` |
| Overlap with training questions | 0 of 950,000 `train.parquet` rows (sha256 `5404981a…`) hash into the eval set |
| [`test_maxtext_reward_fmt_scope.py`](test_maxtext_reward_fmt_scope.py) | PASS. Covers correct / incorrect × formatted / unformatted, each as a training and as an eval row. Training rows score `acc` (`fmt_coef` 0); eval rows score `acc + 0.1·fmt` (`fmt_coef` 0.1); `acc` and `fmt` flags unchanged; length penalty off. Also checks the three failure modes above. |
| Built-in reward self-test, default knobs | all cases OK |
| Default path vs reference dumps | 2,000 rows from the A0 run (500 each from train steps 1 and 100, eval steps 0 and 100) re-scored: 0 changes in `score`, `acc` or `fmt` |

**Launch.** Started 2026-10-23T00:58Z on an idle cluster (0 / 64 GPUs in use).
- Frozen launcher `7d511735…` and wrapper `2a37aad5…`, `SEED=1`, `NNODES=16`, `TOTAL_STEPS=100`.
- Only additions: the two Ray runtime-environment variables:
  - `++ray_kwargs.ray_init.runtime_env.env_vars.REWARD_FMT_WEIGHT_TRAIN="0"`
  - `++ray_kwargs.ray_init.runtime_env.env_vars.REWARD_EVAL_QUESTIONS="<repo>/eval_5k/eval_5000_question_sha256.json"`

## Initial validation and scope

This initial commit adds the experiment plan only. It changes no launcher, scorer, training data or eval protocol and runs no GPU jobs. The plan was checked against the current launcher, reward implementation, comparison tool, PR #10 results and frozen eval contract.

No overlapping experimental PR was found among this fork's open PRs; the upstream open-PR search for `gemma4 ablation` returned no match. This work follows PR #10's precision investigation and has a distinct scope: recipe ablations.

AI assistance: Codex drafted this plan at the user's request. Implementation review and experimental validation remain to be recorded as work proceeds.
