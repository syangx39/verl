# Gemma-4 E2B (`rl-05`) Frozen Evaluation & 200-Step GPU Baseline Protocol (PR #9, step 1)

> [!IMPORTANT]
> **Status:**
> - Evaluation dataset (`eval_5000.parquet`), prompt-token ID fixture (`eval_5000_prompt_tokens.parquet`), and 200-step GPU baseline launcher (`run_gemma4_e2b_rl05_accuracy_baseline.sh`) are frozen.
> - **BF16 seed 1 (`gemma4_e2b_rl05_16n64g_s200_seed1_1021_1955`, `16n64g`) is complete** (`exit_code=0`, 200/200 steps, 64/64 ranks guard PASS, §5) and is kept as the original-recipe reference. Its step-200 change (+0.54 pp) is not distinguishable from zero. Seeds 2/3 of this recipe are on hold (seed 2 stopped at step 147) while PR #10 investigates optimizer update precision ([`UPDATE_PRECISION_RESULTS.md`](UPDATE_PRECISION_RESULTS.md)); a revised recipe starts again from seed 1 and is not pooled with these BF16 seeds.
> - **Customer gate (Meta):** $\overline{\text{Acc}}_{\text{TPU}}(200) \ge \overline{\text{Acc}}_{\text{GPU}}(200) - 1.0\text{ pp}$, where $\overline{\text{Acc}}(200) = \frac{1}{3}\sum_{s=1}^{3} \text{Acc}_s(200)$ is the **3-seed arithmetic mean** held-out greedy accuracy on `eval_5000.parquet` at the pre-declared **final checkpoint (step 200)** under identical recipe, training budget, prompt token IDs, and grading (§3.1).

---

## 1. Frozen Held-Out Evaluation Set (`eval_5000.parquet`)

Built deterministically by [`build_gemma4_eval_set.py`](build_gemma4_eval_set.py) (`sha256: f0b1837cb2c6c0b98478c3c4b2858bd89375805b6612be0fb1c6efac7f68f260`); full provenance in [`eval_5k/manifest.json`](eval_5k/manifest.json).

### 1.1 Why raw `val.parquet` had to be filtered & deduplicated

| Dataset | Path | SHA256 | Rows | Unique normalized questions | Overlap with `train.parquet` |
|---|---|---|---:|---:|---:|
| `train.parquet` | `/workspace/meta-RL/data/openmathinstruct2/train.parquet` | `5404981a07c78a5c02d4f493293fc30bd1ddb42908376566fa9a1020abbdf01b` | 950,000 | 591,477 | — |
| `val.parquet` | `/workspace/meta-RL/data/openmathinstruct2/val.parquet` | `ded5dc67f4c70401c749155cd2a1392e85631ed24b098e1c7686f735619161f5` | 50,000 | — | **37,372 / 50,000 (74.7%)** |
| **Non-overlapping `val` pool** | — | — | 12,628 | **11,652** | **0 (0.0%)** |
| **Frozen `eval_5000.parquet`** | `/workspace/meta-RL/data/openmathinstruct2/gemma4_eval_5k/eval_5000.parquet` | `d092a9c5e80aa2b3cf8e4e12d8367769e770f69ee6627ff1285fd5693efc662d` | **5,000** | **5,000** | **0 (0.0%)** |

### 1.2 Selection rule (deterministic, seed-free)

1. Extract the question body from `row["prompt"][0]["content"]` (between the instruction prefix and `<end_of_turn>`).
2. Normalize to `key = re.sub(r"[\W_]+", "", question.lower())`.
3. Exclude every `val.parquet` row whose `key` appears in `train.parquet` (37,372 rows removed) or duplicates an earlier `val.parquet` row with the same `key` (keeping the lowest `val_index`, leaving 11,652 unique non-overlapping problems).
4. Sort the 11,652 survivors by `sha256(key.encode("utf-8")).hexdigest()` ascending and take the first **5,000** rows, preserving the exact `val.parquet` Arrow schema.

At $N = 5{,}000$ and $\hat{p} \approx 0.55$, the single-run binomial standard error is $\sqrt{\hat{p}(1-\hat{p})/N} \approx 0.70\text{ pp}$ (and paired differences across steps or seeds on the same 5,000 problems have even smaller standard error, typically $0.3\text{–}0.4\text{ pp}$).

---

## 2. Frozen Prompt-Token ID Fixture (`eval_5000_prompt_tokens.parquet`)

- **Committed file:** [`eval_5k/eval_5000_prompt_tokens.parquet`](eval_5k/eval_5000_prompt_tokens.parquet) (also on FUSE at `/workspace/meta-RL/data/openmathinstruct2/gemma4_eval_5k/eval_5000_prompt_tokens.parquet`)
- **SHA256:** `a54fbb327d9c60545c49edfc26107b21a1e55f3ef9e673d80024c79e821b5ca4`
- **Columns:** `eval_pos` (`0..4999`), `val_index` (original row index in `val.parquet`), `key_sha256`, `n_prompt_tokens`, `prompt_token_ids` (`list<int32>`).
- **Prompt rendering (double-wrapped, deliberately, for MaxText parity):** the user message content in `openmathinstruct2/*.parquet` already contains a Gemma-3-style turn template, `<start_of_turn>user\n{system_prompt}\n\n{question}<end_of_turn>\n<start_of_turn>model`. This is MaxText's `examples/chat_templates/gsm8k_rl.json` `TEMPLATE`, mirrored by `my_scripts/preprocess_openmathinstruct2.py`. Both MaxText (`trainers/post_train/rl/utils_rl.py`: `format_maxtext_messages` puts it in a user message, then `apply_chat_template(..., add_generation_prompt=True)`) and verl then apply the model's own chat template on top. The Gemma-4 prompt is therefore `<bos><|turn>user\n` + literal `<start_of_turn>user\n` + system prompt + question + literal `<end_of_turn>\n<start_of_turn>model` + `<turn|>\n<|turn>model\n`. Thinking is off (`enable_thinking` defaults to false; there is no thought channel). Earlier text here showed a `<|thought>` suffix; that was wrong.
  - In the Gemma-4 tokenizer, `<start_of_turn>` / `<end_of_turn>` are **not** special tokens; each is 7 ordinary text pieces (`<`, `start`, `_`, `of`, `_`, `turn`, `>`). That is about 25 prompt tokens of literal marker text, plus a fake `model` header inside the user turn. Only `<bos>`, `<|turn>` (id 105) and `<turn|>` (id 106) are special.
  - Model outputs do not echo the markers: 0 / 5,000 eval outputs at step 0 and 0 / 2,048 training outputs at step 1 contain them, and 97.5% start with `<reasoning>`. The reward scores only the response, so it is unaffected.
  - Not a GPU/TPU mismatch, since both pipelines build the same prompt; it is a recipe-quality quirk for Gemma-4. A single-wrapped prompt (markers removed, the Gemma-4 template applied once) is a candidate single-variable change. It would change the eval prompt-token fixture (a new eval-protocol version) and must be mirrored on the TPU side.
  - Verified against the 256 online rollout prompts in `/workspace/meta-RL/logprob_parity/fixture_fixed_1021_0813/fixture_step1.npz`: **256 / 256 prompts matched token-for-token (`n_exact = 256, n_mismatch = 0`)**.
  - Token length distribution across the 5,000 eval prompts: `min = 91`, `p50 = 148`, `mean = 167.0`, `max = 2,268` (zero prompts truncated at `max_prompt_length = 8,192`).
  - For GPU–TPU comparison, the TPU evaluation pipeline can either consume `prompt_token_ids` directly from `eval_5000_prompt_tokens.parquet` or verify that its rendered prompt token IDs match `prompt_token_ids` bit-for-bit.

---

## 3. Frozen 200-Step GPU Baseline Protocol (`run_gemma4_e2b_rl05_accuracy_baseline.sh`)

| Knob | Value | Notes |
|---|---|---|
| Base recipe | [`run_gemma4_e2b_rl05_parity.sh`](run_gemma4_e2b_rl05_parity.sh) (PR #9 BF16 reference: sha256 `8646c173…`, `model_dtype=bfloat16`; from PR #10: sha256 `7d511735…`, `model_dtype=fp32`, otherwise identical) | Unchanged frozen `rl-05` training parameters (`train_batch_size=256`, `rollout.n=8`, `ppo_mini_batch_size=256` [$\mu=1$], `lr=1e-6`, `clip=0.2/0.28`, $\beta=0$, `max_prompt=8192`, `max_response=8192`, `T=0.8`, `top_k=50`, `top_p=0.95`, `ATTN_IMPL=flex_attention`) |
| RoPE fix & guard | Step 0 (`gemma4_node_prep.py` + `[GEMMA4-ROPE v2]`) | Enforced automatically on every launch ([ROPE_GUARD.md](ROPE_GUARD.md)) |
| Training horizon | `TOTAL_STEPS=200` | 200 GRPO steps $\times$ 2,048 completions/step = 409,600 training completions (51,200 prompts) |
| Eval dataset | `eval_5000.parquet` | 5,000 non-overlapping unique problems (§1) |
| Eval schedule | `trainer.val_before_train=True`, `trainer.test_freq=20` | 11 checkpoints: step **0** (base model before any update) and steps **20, 40, 60, 80, 100, 120, 140, 160, 180, 200** |
| Eval decoding | Greedy (`do_sample=False, temperature=0, top_k=-1, top_p=1.0, n=1`) | Deterministic argmax decoding (`max_response_length=8192`) |
| Grading | [`maxtext_math_reward.py`](maxtext_math_reward.py) (`compute_score`) | `math-verify==0.9.0`; logs `acc` ($\in \{0,1\}$), `fmt` ($\in \{0,1\}$), and `score = acc + 0.1 * fmt` |
| Per-sample dumps | `trainer.validation_data_dir`, `trainer.rollout_data_dir` | Writes `<eval_dump_dir>/val/<step>.jsonl` (5,000 rows per eval step) and `<eval_dump_dir>/train/<step>.jsonl` (2,048 rows per train step) |
| Seeds | `SEED=1`, `SEED=2`, `SEED=3` | Sets `data.seed`, `actor_rollout_ref.rollout.seed`, `actor_rollout_ref.actor.data_loader_seed`, and `actor_rollout_ref.actor.fsdp_config.seed` |

### 3.1 Pre-Registered Statistical Gate Definition (`TPU ≥ GPU − 1.0 pp`)

To avoid post-hoc cherry-picking across intermediate checkpoints or individual seeds, the customer acceptance gate is pre-registered as follows:

1. **Primary Pass/Fail Criterion (Step 200, 3-Seed Mean):**
   - Let $\text{Acc}_{\text{dev}, s}(t) \in [0, 100]$ denote the greedy held-out accuracy (%) on `eval_5000.parquet` for platform $\text{dev} \in \{\text{GPU}, \text{TPU}\}$, seed $s \in \{1, 2, 3\}$, at training step $t \in \{0, 20, \dots, 200\}$.
   - The primary benchmark metric is the **3-seed arithmetic mean at the final checkpoint ($t = 200$)**:
     $$\overline{\text{Acc}}_{\text{dev}}(200) = \frac{1}{3}\sum_{s=1}^{3} \text{Acc}_{\text{dev}, s}(200)$$
   - The gate passes if and only if:
     $$\overline{\text{Acc}}_{\text{TPU}}(200) \ge \overline{\text{Acc}}_{\text{GPU}}(200) - 1.0\text{ pp}$$
   - Intermediate checkpoints ($t \in \{20, 40, \dots, 180\}$) are **not** used to pick a best-of-run checkpoint ($\max_t$) for the gate decision.
2. **Required Companion Reporting (Full Trajectory & Seed Dispersion):**
   - **Full 11-point learning curve (`steps 0, 20, 40, ..., 200`):** Report every individual seed's curve $\text{Acc}_{\text{dev}, s}(t)$ alongside the 3-seed mean curve $\overline{\text{Acc}}_{\text{dev}}(t)$.
   - **Across-seed dispersion:** At every evaluated step $t$ (and specifically at $t = 200$), report the across-seed sample standard deviation
     $$s_{\text{seed}}(t) = \sqrt{\frac{1}{3 - 1}\sum_{s=1}^{3}\bigl(\text{Acc}_{\text{dev}, s}(t) - \overline{\text{Acc}}_{\text{dev}}(t)\bigr)^2}$$
     and the 3-seed min/max range $[\min_s \text{Acc}_{\text{dev}, s}(t),\ \max_s \text{Acc}_{\text{dev}, s}(t)]$ (produced automatically by [`gemma4_eval_report.py`](gemma4_eval_report.py) under `## Multi-seed summary`).
   - **Within-seed & paired uncertainty:** Report the single-run binomial SE $\sqrt{\hat{p}(1-\hat{p})/5000}$ and the paired SE of $\Delta_s(t) = \text{Acc}_{\text{dev}, s}(t) - \text{Acc}_{\text{dev}, s}(0)$ (`+Gained / -Lost` on the identical 5,000 problems).
   - **Step 0 base-model sanity check:** Verify $\text{Acc}_{\text{TPU}}(0)$ vs $\text{Acc}_{\text{GPU}}(0)$ before any gradient update on the identical `eval_5000_prompt_tokens.parquet`.

---

## 4. Launch & Verification Commands (Ray head pod)

```bash
source /workspace/setup_env.sh
SCRIPT_DIR=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gemma4

# Launch 200-step Seed 1 baseline on 16n64g (or NNODES=4 for 4n16g)
SEED=1 NNODES=16 TOTAL_STEPS=200 \
  bash "$SCRIPT_DIR/run_gemma4_e2b_rl05_accuracy_baseline.sh" \
  2>&1 | tee /workspace/meta-RL/logs/gemma4_e2b_rl05_16n64g_s200_seed1.log

# Verify guard evidence across all nodes and ranks
python3 "$SCRIPT_DIR/gemma4_guard_check.py" \
  --log /workspace/meta-RL/logs/gemma4_e2b_rl05_16n64g_s200_seed1.log \
  --guard-dir /workspace/meta-RL/logs/gemma4_guard/<EXPERIMENT_NAME> \
  --world 64 --steps 200

# Summarize held-out accuracy curve (steps 0, 20, ..., 200) and paired SEs
python3 "$SCRIPT_DIR/gemma4_eval_report.py" \
  --run seed1=/workspace/meta-RL/logs/gemma4_eval/<EXPERIMENT_NAME> \
  --log seed1=/workspace/meta-RL/logs/gemma4_e2b_rl05_16n64g_s200_seed1.log
```

---

## 5. 200-Step GPU Baseline Results (`16n64g`, 64 GB200 GPUs)

### 5.1 Run Provenance & Guard Verification

| Seed | Experiment Name | Driver Log | Guard Check | Node Prep | All-Rank Init Guard | First-Forward Guard | `k3_kl` (steps 1–200 min / med / max) | `grad_norm` (steps 1–200 min / med / max) | Step Time (steps 2–200 med / mean / p90) |
|---:|---|---|---|---|---|---|---|---|---|
| **1** | `gemma4_e2b_rl05_16n64g_s200_seed1_1021_1955` | [`logs/gemma4_e2b_rl05_16n64g_s200_seed1_1021_1955.log`](logs/gemma4_e2b_rl05_16n64g_s200_seed1_1021_1955.log) (`exit_code=0`) | [`eval_5k/seed1_16n64g/guard_check.txt`](eval_5k/seed1_16n64g/guard_check.txt) (**16/16 PASS**) | [`eval_5k/seed1_16n64g/node_prep.json`](eval_5k/seed1_16n64g/node_prep.json) (`16/16` alive nodes PASS) | [`eval_5k/seed1_16n64g/init_actor_aggregate.json`](eval_5k/seed1_16n64g/init_actor_aggregate.json) (`64/64` unique ranks PASS, `256/256` buffers bit-exact FP32) | `64/64` unique ranks PASS (`restored=[]`) | `0.000640 / 0.000684 / 0.000739` | `0.1289 / 0.2588 / 3.3281` | `70.14s / 73.85s / 83.96s`; outliers: step 1 `111.1s` (compile), step 182 `397.4s` (rollout `370.8s`, a response hit the 8,192-token cap) |
| **2** | `gemma4_e2b_rl05_16n64g_s200_seed2_1022_0301` — **stopped at step 147** (BF16 recipe put on hold for the PR #10 investigation); evals at steps 0–140 kept as a replicate | — | — | — | — | — | — | — | — |
| **3** | *not launched* (deferred until a revised recipe is frozen; PR #10) | — | — | — | — | — | — | — | — |

### 5.2 Seed 1 Held-Out Accuracy Trajectory (`eval_5000`, Greedy Decoding)

Exact output of [`gemma4_eval_report.py`](gemma4_eval_report.py) ([`eval_5k/seed1_16n64g/seed1_report.md`](eval_5k/seed1_16n64g/seed1_report.md), [`eval_5k/seed1_16n64g/seed1_report.json`](eval_5k/seed1_16n64g/seed1_report.json)):

| Step | N (`eval_5000`) | Acc (%) | Binomial SE (pp) | $\Delta$ vs step 0 (pp) | Paired SE (pp) | +Gained / -Lost | Format (%) | Mean reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | 5000 | **36.26** | 0.68 | +0.00 | 0.00 | +0 / -0 | 93.94 | 0.4565 |
| **20** | 5000 | **36.34** | 0.68 | +0.08 | 0.48 | +290 / -286 | 95.26 | 0.4587 |
| **40** | 5000 | **36.56** | 0.68 | +0.30 | 0.49 | +303 / -288 | 95.94 | 0.4615 |
| **60** | 5000 | **37.32** | 0.68 | +1.06 | 0.48 | +317 / -264 | 95.88 | 0.4691 |
| **80** | 5000 | **37.16** | 0.68 | +0.90 | 0.48 | +309 / -264 | 95.64 | 0.4672 |
| **100** | 5000 | **36.92** | 0.68 | +0.66 | 0.48 | +304 / -271 | 96.10 | 0.4653 |
| **120** | 5000 | **37.50** | 0.68 | +1.24 | 0.49 | +333 / -271 | 96.38 | 0.4714 |
| **140** | 5000 | **36.68** | 0.68 | +0.42 | 0.49 | +308 / -287 | 96.48 | 0.4633 |
| **160** | 5000 | **37.22** | 0.68 | +0.96 | 0.48 | +318 / -270 | 96.30 | 0.4685 |
| **180** | 5000 | **37.34** | 0.68 | +1.08 | 0.48 | +315 / -261 | 96.64 | 0.4700 |
| **200** | 5000 | **36.80** | 0.68 | **+0.54** | 0.49 | +309 / -282 | **96.70** | **0.4647** |

- **Step 0 → Step 200 summary (Seed 1):**
  - Base held-out accuracy (`step 0`): **36.26% ± 0.68 pp** (`1,813 / 5,000`), format compliance **93.94%**, mean reward **0.4565**.
  - Predeclared endpoint (`step 200`): **36.80% ± 0.68 pp** (`1,840 / 5,000`, $\Delta = +0.54\text{ pp}$, paired SE `0.49 pp`, question-paired bootstrap 95% interval **[−0.48, +1.48] pp**, `+309 / -282`), format compliance **96.70%**, mean reward **0.4647**. Pairing is by validated question identity (`qid`, input and ground-truth hashes; [`update_precision/audit_seed1/audit.md`](update_precision/audit_seed1/audit.md)).
  - Intermediate checkpoints (steps 60–200) range **36.68% – 37.50%**; they are reported for the trajectory only and are not substituted for the endpoint.
  - Online training reward (`critic/score/mean`, $T=0.8$, 2,048 completions/step): **0.6906** (steps 1–20 mean; sampled accuracy **59.27%**, format **97.96%**) and **0.7124** (steps 181–200 mean; sampled accuracy **61.31%**, format **99.34%**).

![Seed 1 Held-Out Evaluation Curve](eval_5k/seed1_16n64g/seed1_eval_curve.png)

![Seed 1 Online Training & Numerical Guard Diagnostics](eval_5k/seed1_16n64g/seed1_training_diagnostics.png)

Panel D of the diagnostics figure clips the step-time axis at 115 s: step 182 took **397.4 s** (rollout 370.8 s) and is off the top of the plot.

### 5.3 Seed 1 observations (revised in PR #10)

> [!IMPORTANT]
> PR #10 audited this run ([`UPDATE_PRECISION_RESULTS.md`](UPDATE_PRECISION_RESULTS.md)). The run is a numerically stable BF16 reference, **not** evidence of a learning gain: the step-200 change is within the run-to-run nondeterminism of the greedy eval, and the optimizer in this recipe wrote back only a small fraction of each AdamW update (BF16 master weights and moments). Treat any revised-precision run as a separate cohort.

1. **Numerical stability and RoPE guard (`16n64g`, 64 ranks):**
   - All 16 nodes passed `gemma4_node_prep.py` in `10.6s`, all 64 unique FSDP actor ranks rebuilt `256/256` RoPE `inv_freq` buffers to bit-exact FP32 at `init_checkpoint`, and all 64 unique ranks passed the `first_forward` guard with zero buffer drift (`restored=[]`).
   - Across all 200 training steps (`409,600` completions), `rollout_corr/k3_kl` stayed in **`[0.000640, 0.000739]`** (median **`0.000684`**, vs. pre-fix `~0.98–1.35`), and `training/rollout_actor_probs_pearson_corr` in **`[0.99714, 0.99730]`** (median **`0.99721`**).
   - `actor/grad_norm` stayed in **`[0.1289, 3.3281]`** (median **`0.2588`**, vs. pre-fix `10–350`); the run exited `0`. Steps 2–200 median step time **`70.14s`** with one outlier (step 182, `397.4s`, rollout-bound). Nonzero gradients do not imply effective weight updates ([`UPDATE_PRECISION_RESULTS.md`](UPDATE_PRECISION_RESULTS.md) §2).
2. **Held-out `eval_5000`:**
   - **Format compliance** went from **`93.94%`** (step 0) to **`96.70%`** (step 200), **not monotonically** (it dipped between steps 40→60, 60→80 and 140→160).
   - **Accuracy:** the predeclared step-200 change is **+0.54 pp** (bootstrap 95% **[−0.48, +1.48] pp**), not distinguishable from zero. Two greedy evals of the **identical** step-0 weights (seed 1 vs the seed-2 run) differ by **+0.48 pp** with **+305 / −281** flipped questions and only **288 / 5,000** identical outputs, i.e. the same magnitude as step 0 → 200 (`+309 / −282`). Intermediate checkpoints are within the same band.
3. **Earlier attribution withdrawn:** PR #9 explained the ~300 flips per direction by questions sitting "near the greedy decision boundary". No logit evidence supports that; the identical-weights replicate shows that run-to-run nondeterminism of batched greedy vLLM decoding alone produces flips of this size. The predeclared step-200, 3-seed endpoint (§3.1) and question-paired uncertainty remain the reporting rule.
