# Qwen3 TPU/GPU RL parity rulebook — GSM8K, 2K response cap, disaggregated asynchronous GRPO (recipe `gsm8k_2k_async1`)

This rulebook defines the TPU workload corresponding to the GB200 reference validated by three 250-step runs. It is the GPU baseline requested by the TPU TorchTitan team: the same GSM8K task as the Meta Ads round, but run **disaggregated** (separate trainer and sampler accelerator pools) and **asynchronously** (the sampler generates the next batch while the trainer updates on the current one, weights pushed to the sampler after every update). The algorithm (data, prompt, reward, GRPO, single-forward REINFORCE with a truncated importance weight, optimizer, evaluation) is the colocated `gsm8k_2k_v1` recipe unchanged; what is new is the execution model and its observable consequences (one-step policy lag, responses that span two policy versions, rare drops), which this document specifies and quantifies. It separates the training semantics both sides must preserve (Tier 1), the execution model (Tier 2), and the implementation choices each platform may optimize (Tier 3).

## Rulebook anchor

Both sides implement the same versioned recipe and evaluation protocol. The anchor is the GB200 `gsm8k_2k_async1` recipe: three seeds, 250 optimizer steps, 64 GB200 split 32 trainer + 32 rollout. A change to a frozen invariant, or to the asynchrony parameters in Tier 2 (sync period, warm-up, staleness threshold, drop policy), creates a new recipe version and requires a matching reference run.

The colocated synchronous `gsm8k_2k_v1` runs (two seeds, same algorithm, 64 GB200 shared) are delivered as context only (`band/gb200_vs_colocated_2k.png`); they are the on-policy baseline that this asynchronous reference is compared against, not an acceptance target.

## Tier 1 — Frozen invariants

Resolved run configuration (`env/resolved_config_seed{1,2,3}.yaml`) and the executed code (`env/VERL_PIN.txt`, `code/`) take precedence over framework defaults, comments, or preset names.

| Invariant | Frozen value or requirement |
|---|---|
| Initial model | `Qwen/Qwen3-0.6B` — the **post-trained** checkpoint. Fingerprint: 311 tensors, untied `lm_head`, `sum(model.norm.weight) = 3932.626953125`; `generation_config.eos_token_id = [151645, 151643]`. Freeze weights, config, tokenizer, chat template (`model/`, `MODEL_SHA256`). Every seed starts from that checkpoint. |
| Training data | GSM8K train, 7,473 rows, Meta's boxed format (`data/gsm8k_boxed_train.parquet`). Zero prompt-text overlap with the test set (`DATA_COUNTS.json`). |
| Data order | The sampler dispatches prompts in the order of a seeded shuffle of the training set (`data.seed = k`), **one prompt at a time**; the trainer consumes prompt groups in the order their 16 generations complete. Batch membership at a given step therefore depends on completion timing and is not reproducible across runs; `data/step_manifest_seed{k}.json` records the training rows each GPU seed actually consumed at each step, as a reference. TPU: use a seeded shuffle, dispatch in that order, and log per-step question IDs. Do not sort or bucket prompts by length. |
| Prompt | System + user messages exactly as in the parquet, rendered by the Qwen3 chat template with `add_generation_prompt=True` and `enable_thinking=True` (template default: no `<think>` tokens injected). Rendered text ends with `<|im_start|>assistant\n`. Token IDs frozen by `fixtures/prompt_fixture.json`. Prompt limit 512 tokens. |
| Evaluation data | `data/gsm8k_boxed_test.parquet`: all 1,319 GSM8K test questions. |
| Step accounting | 128 prompts × 16 generations = 2,048 completions per optimizer step; exactly one update per 128 complete groups (`mini = batch`, one epoch, μ = 1); 250 steps ≈ 4.3 passes over the training questions. |
| Optimizer | AdamW; peak LR **2e-6**; linear warm-up over 10 updates (update 1 at LR 0, update 2 at 2e-7 — the reference logs the LR *after* `scheduler.step()`, so `actor/lr` at step *t* is the LR the *next* update uses); cosine decay to 0 at step 250; betas (0.9, 0.999); eps 1e-8; weight decay 0; global gradient-norm clipping at 1.0 after accumulation. |
| Advantage | GRPO: A = (R − mean_group) / (std_group + 1e-6), sample std (correction = 1), groups of 16 by prompt UID; broadcast to every response token. Zero-std groups stay in the batch and in the loss denominator. No dynamic sampling / group filtering. |
| Policy loss | **Single-forward REINFORCE with a detached truncated importance weight**: loss = token-mean over all valid response tokens of −A·w·log π_θ, w = min(exp(clamp(log π_θ.detach() − log π_sampler, −20, 20)), **3.0**), no-grad. The PPO ratio is exactly 1 (no separate old-policy pass; `bypass_mode = True`, `loss_type = reinforce`); clip and dual-clip are inert. **log π_sampler is the sampler's per-token log-probability of the sampled token under the policy version that generated that token** (see Tier 2 for why this matters here). No lower truncation, no batch normalization, no rejection sampling. |
| Loss denominator | Global token mean over the whole 2,048-completion batch (terminating token included, prompt tokens excluded). Micro-batching or dynamic batching must reproduce this global mean (verified in the reference code: the denominator is computed before the batch is split). |
| Reference KL / entropy | β = 0: no reference model, no KL loss or reward shaping; entropy coefficient 0. |
| Training sampling | Temperature 1.0; top_k −1; top_p 1.0; no repetition/presence penalties; stop tokens [151645, 151643]; `ignore_eos = False`. |
| Length limits | Prompt 512 tokens; **response 2,048 generated tokens**; a response that reaches the cap is kept in the loss with all its tokens. |
| Training reward | `boxed_math` (`code/boxed_math_reward.py`): last balanced `\boxed{…}`, minimal normalization; exact match → 1.0; parseable but wrong → 0.1; no parseable box → 0.0. **Plus the overlong penalty for the training source only**: penalty = −min(1, max(0, (L − 1536) / 512)) for response length L (linear from 0 at 1,536 tokens to −1.0 at 2,048; `REWARD_OVERLONG_BUFFER 512`, `REWARD_OVERLONG_PENALTY 1.0`, `REWARD_MAX_RESP_LEN 2048`, `REWARD_PENALTY_SOURCES gsm8k_boxed_train`). Training score = raw reward + penalty; L counts the generated tokens including the terminating token. |
| Evaluation | Greedy (temperature 0, top_p 1, top_k −1), n = 1, cap 2,048, **no penalty**, on the sampler pool with the latest synchronized weights, at steps 0, 20, …, 240, 250 (14 evaluations). Metric: `acc` = (raw reward == 1.0). No best-checkpoint selection. |
| Precision | FP32 master weights and Adam moments; BF16 compute; BF16 sampler weights and KV cache; log-probabilities in FP32 from the logits. |
| Run | 250 optimizer steps per seed; seeds 1, 2, 3. |

## Tier 2 — Execution model (this is what changed)

**Topology.** 64 GB200 = 32 trainer GPUs (FSDP2, data-parallel 32) + 32 rollout GPUs (32 standalone vLLM replicas, TP 1, one per GPU). No rollout instance runs on a trainer GPU (`actor_rollout_ref.hybrid_engine = false`; the reference logs `hybrid replicas disabled … rollout served by 32 standalone replicas only`). The TPU counterpart is two slices: one for training, one for sampling; chip counts are a declared, labeled scaling point.

**Pipeline (who does what).**
1. Before the first update the sampler generates one full batch under θ₀ (`num_warmup_batches = 1`); the trainer's first update uses it (staleness 0).
2. From then on the sampler always generates the *next* batch while the trainer updates on the *current* one. After every optimizer update the trainer pushes θ_t to all 32 replicas over NCCL (`parameter_sync_step = 1`; 0.8–1.0 s per sync on GB200). The sampler does **not** abort or restart in-flight requests at a sync: tokens already generated stay, later tokens of the same response are sampled under the new weights.
3. Prompts are dispatched one at a time (`gen_batch_size = 1`) with n = 16 each; complete groups are queued (transfer queue) and the trainer takes the next 128 complete groups in completion order. The update is one step of Tier 1 on exactly those 2,048 completions.
4. A group whose *oldest* token was generated more than **2** policy versions before the update that would consume it is **dropped** (`max_off_policy_threshold = 2`, strategy `drop`), never trained on.
5. Evaluation runs on the sampler pool with the weights of the checkpoint being evaluated (the trainer waits).

**What the configuration guarantees vs what was observed.** The configuration bounds the lag only through the threshold: a consumed group may have its oldest token generated up to **2** policy versions before the update that consumes it (staleness ≤ 2), and a response may therefore span up to 3 versions. Mixed-version responses are allowed and expected. The three reference runs stayed well inside these bounds (`band/summary.json`, `runs/seed*/check_smoke.json`), and these are **observations, not guarantees**:
- `trajectory_staleness_worst/max` = 1 at every step after the first (0 at step 1): every consumed token was generated under θ_{t−1} or θ_t. A TPU run that consumes some groups at staleness 2 is within the recipe; report the distribution.
- `trajectory_spans/max` = 2 in 136 / 138 / 144 of the 250 steps: at least one response in more than half of the batches contains tokens from two consecutive policy versions (the sync happened while it was in flight). This is the normal state of the pipeline, not an anomaly.
- Dropped groups: **11 / 8 / 8 of 32,000 per seed (≈ 0.03 %)**, all with staleness 3 — long responses (near the 2,048 cap) whose group did not complete within two updates. No other eviction reason.
- The importance weight absorbs both effects. Because log π_sampler is the sampler's own log-probability for each token under the version that produced it, w is a correct per-token importance weight even for a response spanning two versions; the observed weight statistics are `rollout_is_mean` 1.000, effective sample fraction 0.9986, fraction of tokens with pre-truncation ratio > 3 ≈ 1e-6, `k3_kl` (trainer vs sampler, per token) ≈ 7e-4 in steady state with a transient rise to ≈ 1.6e-3 during steps 15–40 when the policy changes fastest. On the colocated reference `k3_kl` is a flat 6–7e-4 (kernel mismatch only); the rise is the lag signature and is the only visible difference in the training diagnostics.

**Rules for the TPU implementation.**
- Same pipeline: sampler one batch ahead, weights pushed after *every* update, one warm-up batch, threshold 2 with drop. Different values are a different recipe version. What must match is this configuration; the observed lag/span/drop statistics are expected to be similar (worst lag mostly 1, spans ≤ 2, drops ≈ 0.03 %) but are reported, not required. Record the per-step counts of consumed groups by staleness, spans per response, and dropped groups; the GPU tags are `training/off_policy/trajectory_staleness{,_worst}/{mean,max,min}`, `training/off_policy/trajectory_spans/{mean,max,min}`, `training/off_policy/evicted_samples`.
- log π_sampler must be the sampler's per-token value under the generating version (not recomputed by the trainer, not a single version per response). If the TPU sampler cannot continue an in-flight request across a weight update and instead restarts or aborts it, say so: spans stay 1, the drop rate and the batch composition change, and that is a documented deviation, not a violation.
- Dropped groups are discarded, not resampled and not moved to the next batch.
- Nothing about asynchrony changes Tier 1: the update is still one global-token-mean REINFORCE step over 128 complete groups with detached TIS at 3.0.

## Tier 3 — Free implementation variables

Recorded for the reference; not requirements. Container `env/IMAGE_REF.txt` (derived from `verlai/verl:uv-cu130-arm64`), verl `verl-project/verl @ ace775e8` (`env/VERL_PIN.txt`), Python 3.12, Torch 2.13.0+cu130, vLLM 0.29.0, Transformers 5.12.1, TransferQueue 0.1.10, FlashAttention 2.8.3 (`env/versions.txt`, `env/uv.lock`). Trainer: FSDP2 with FP32 params/BF16 compute, remove-padding, gradient checkpointing, **dynamic batching at 32,768 tokens (prompt + response) per GPU** instead of a fixed micro-batch. Sampler: vLLM async server, TP 1, `gpu_memory_utilization 0.6`, `max_num_seqs 256`, `max_num_batched_tokens 8192`, prefix caching on, cudagraph `FULL_AND_PIECEWISE`, `logprobs_mode processed_logprobs`. Weight transfer: NCCL checkpoint engine, 1 GB buckets. Transfer queue: `SimpleStorage`, 32 units. The Ray environment also carries `VLLM_USE_V1=1`, which vLLM 0.29 ignores (kept so the recipe file resolves to exactly the reference runs' configuration).

Topology and batching were chosen by three 20-step trials (steps 11–19, median / p90 step time): 16 trainer + 48 rollout with fixed micro-batch 8 → 13.0 / 13.5 s (trainer-bound, MFU ≈ 2 %); 16 + 48 with dynamic batching → 6.8 / 12.8 s; **32 + 32 with dynamic batching → 7.0 / 8.8 s** (chosen: same median, much tighter tail). These are performance choices with the same objective; the reference band was produced only with the chosen one.

## Alignment sequence

1. **Prompt rendering (gate 1, CPU).** Reproduce `fixtures/prompt_fixture.json` token-exactly. Unchanged from the 8K package.
2. **Scorer (gate 2, CPU).** Reproduce `fixtures/meta_reward_fixtures.json` (21 rules) and `fixtures/scorer_fixture_2k.jsonl` (800 real evaluation responses from seed 1 at steps 0 and 250 with their dumped scores: 1.0 / 0.1 / 0.0) exactly; reproduce the training-mode penalty on the lengths in `fixtures/reward_selftest_2k_penalty.log`.
3. **Step-0 evaluation (gate 3, sampler only).** Greedy on the initial checkpoint: GB200 0.7066 / 0.7066 / 0.7028 (mean 0.7053). Per-question agreement between GB200 seeds is recorded in `band/step0_greedy_variability.json`; compare distributions, not questions.
4. **Asynchrony semantics (gate 4, short run, new in this round).** A 20-step run with evaluation off must show: worst-case staleness 0 at step 1 and ≤ 2 afterwards (GB200: 1 at every step); spans ≤ 3 (GB200: ≤ 2); drops at threshold 2 rare (GB200 20-step runs had none; ≈ 10 per 250 steps); importance-weight mean ≈ 1.00, effective sample fraction ≥ 0.99, fraction of tokens above the threshold ≲ 1e-5; `k3_kl` ≲ 2e-3. The LR sequence of the first three updates is 0 / 2e-7 / 4e-7. `code/check_smoke.py` encodes these checks for the GPU log format and documents each threshold.
5. **Loss contract (gate 5, recommended).** The 8K package's gradient-replay fixtures rely on the colocated code path's log-probability dumps, which the V1 asynchronous trainer does not emit; there is **no per-parameter gradient fixture in this package**. Verify the loss contract on the TPU side directly: on any batch, the trainer's loss must equal the NumPy evaluation of the Tier 1 formula from the per-token (log π_θ, log π_sampler, advantage, mask) it used, to BF16 tolerance, and the denominator must equal the batch's total valid response tokens. The two-pass/single-forward equivalence (gradient cosine 0.99995 on an identical batch) was established for this loss in the colocated round and carries over unchanged.
6. **Three 250-step runs (gate 6).** Seeds 1–3 with the Tier 1/Tier 2 recipe; evaluation at the 14 scheduled steps.

## Quality and performance reporting

### Predeclared reference (three GB200 seeds, `gsm8k_2k_async1`)

GSM8K test (1,319) accuracy, greedy, per scheduled step (min / mean / max across seeds; per-seed values in `band/summary.json`):

| step | min | mean | max |
|---|---|---|---|
| 0 | 0.7028 | 0.7053 | 0.7066 |
| 20 | 0.7142 | 0.7296 | 0.7430 |
| 40 | 0.7665 | 0.7680 | 0.7703 |
| 60 | 0.7877 | 0.7958 | 0.8059 |
| 80 | 0.8036 | 0.8105 | 0.8165 |
| 100 | 0.8006 | 0.8150 | 0.8241 |
| 120 | 0.8074 | 0.8175 | 0.8294 |
| 140 | 0.8074 | 0.8185 | 0.8332 |
| 160 | 0.8135 | 0.8221 | 0.8347 |
| 180 | 0.8127 | 0.8289 | 0.8370 |
| 200 | 0.8143 | 0.8193 | 0.8279 |
| 220 | 0.8211 | 0.8271 | 0.8324 |
| 240 | 0.8180 | 0.8261 | 0.8423 |
| 250 | 0.8234 | 0.8274 | 0.8317 |

Per seed: step 0 → 250 = 0.7066 → 0.8317, 0.7066 → 0.8271, 0.7028 → 0.8234 (gain +12.5 / +12.1 / +12.1 pp). Band width across the 14 checkpoints: median 2.0 pp, p90 2.5 pp, max 2.9 pp (step 20). First of two consecutive evaluations ≥ 0.80 (the confirming one): step 80 / 100 / 100. For context, the colocated `gsm8k_2k_v1` seeds ended at 0.8226 / 0.8180 and confirmed 0.80 at steps 80 / 100: the asynchronous reference learns at the same rate and to at least the same level; three seeds do not establish that it learns better.

Training diagnostics (means over the last 10 steps): entropy ≈ 0.32–0.33 nats/token (from 0.48); response length ≈ 560–600 tokens (from ≈ 1,000 at step 1, minimum ≈ 400 around step 30); cap-hit fraction < 1 % after step 30; training score ≈ 0.87–0.88; pre-clip gradient norm ≈ 0.20; `k3_kl` ≈ 7e-4. All coincide with the colocated seeds except the `k3_kl` transient described in Tier 2.

### Evaluation variability

Seeds 1 and 2 produced identical step-0 accuracy (0.7066); seed 3 differs by 0.4 pp. The per-question agreement across seeds at step 0 is in `band/step0_greedy_variability.json`. As in the 8K round, per-question TPU/GPU comparison is diagnostic only; compare the accuracy distribution and the curves.

### Comparison rule for this round

Full 250-step curves of three TPU seeds are overlaid on the three GPU seeds at the 14 shared checkpoints, with the same recipe and asynchrony parameters. The GPU min–max band is a **descriptive** range from three seeds, not a confidence interval and not a parity criterion by itself; a TPU curve inside it is compatible, one outside it at several checkpoints or with a different final level is a finding to investigate, not automatically a failure. Report alongside: the asynchrony statistics of gate 4 for the full runs (staleness, spans, drops), the importance-weight statistics, and the step-time decomposition. Deviations of Tier 2 behaviour (e.g. spans always 1 because in-flight requests are aborted) must be stated with the result.

### Performance

Reference wall clock (end to end from launch, including ≈ 4 min of startup, 14 evaluations and 5 checkpoints): 59.4 / 59.2 / 59.3 min per seed; the confirming 0.80 evaluation at 21 / 27 / 25 min. Steady-state step time (steps 20–250 excluding evaluation and checkpoint steps, n = 216): median **7.16 / 7.34 / 7.22 s**, p90 9.5–10.3 s; components (median): actor update 3.0 s, weight sync 0.8 s, advantage + queue fetch 0.9 s, log-probability bookkeeping 0.4 s, trainer waiting for the sampler 0.1 s (p90 2.7 s — the sampler is the intermittent bottleneck on long-tail batches). Evaluation costs ≈ 70 s per pass on the 32-GPU sampler pool (≈ 16 min of the 59); the colocated reference evaluates in ≈ 20 s on 64 GPUs. Colocated `gsm8k_2k_v1` on the same 64 GB200: 14.15 / 14.06 s per step, 71.0 / 69.9 min end to end — the asynchronous pipeline is 1.95× faster per step and ≈ 15 % faster end to end at this scale; the gap is smaller end to end because evaluation and startup are unchanged or slower.

Report TPU step time in the same decomposition (`band/gb200_step_time.png`), state the clock origin, and give time to the confirming 0.80 evaluation with the sustained-two rule.

## GB200 reference and handoff package

`gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_2k_async1/` (`README.md` lists every directory; `sha256sum -c PACKAGE_MANIFEST.sha256` verifies it). Key contents: `model/`, `data/` with `step_manifest_seed{1,2,3}.json`, `fixtures/` (gates 1–2), `code/` (recipe YAML, launcher, preflight, reward, `check_smoke.py`, plots, Dockerfile, pin and image reference), `env/` (resolved configs, `CONFIG_DIFF.txt`, `uv.lock`, versions, cluster manifest), `runs/seed{1,2,3}/` (driver log, TensorBoard, 14 evaluation dumps, 250 rollout dumps with per-sample `uid`/`score`/`input`/`output`, `check_smoke.json`), `band/` (band, colocated comparison, step-time and diagnostics figures; `summary.json` with every number quoted above), `checkpoints/seed{1,2,3}_step250/`.

### TPU result formats

1. Per-step evaluation accuracy at the 14 checkpoints per seed, plus the training-score, response-length, entropy, gradient-norm, `k3_kl`, staleness / spans / drops and step-time series, as TensorBoard event files with the GPU tag names or as one CSV per seed with those columns.
2. Per-sample evaluation dumps (question, output, score) at steps 0 and 250 at least; per-step rollout dumps if available.
3. The resolved configuration, software versions, and the asynchrony parameters actually used.

## Known gaps in this round

- No per-parameter gradient fixture (gate 5 is a contract check, not a replay).
- Batch membership is completion-order dependent; the step manifests are references, not a fixed order to reproduce.
- The overlong penalty makes the training score a mixture of accuracy and length; the rollout dumps carry the score only (no separate `acc` field), so training-accuracy curves in the diagnostics figures are score curves.
- Evaluation on the sampler pool is 3–4× slower than colocated evaluation; the end-to-end numbers include it.