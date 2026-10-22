# Gemma-4 RoPE fix: persistence and all-rank guard (PR #9, step 0)

> [!IMPORTANT]
> **Status:** the RoPE fix from PR #8 is installed on every Ray node at every launch, and a guard inside the trainer checks it on every rank and fails the whole job if any rank is wrong. Verified by deterministic unit tests, a 2-step positive smoke (`4n16g`), and a deterministic negative test (one corrupted rank → the whole job exits non-zero before the optimizer is built). Accuracy / convergence: still pending (PR #9, later steps).

PR #8 ([LOGPROB_PARITY.md](LOGPROB_PARITY.md)) fixed the trainer–sampler log-prob mismatch, but the fix lived only in each pod's image-local `/workspace/verl-pin`. It was applied by hand, lost when a pod restarted, and nothing checked it at run time. A restarted pod would silently put ~half of the ranks back on scrambled RoPE frequencies. This step closes that gap in two parts:

- **A. Node prep** (`gemma4_node_prep.py`, run by the launcher before `main_ppo`): every alive Ray node is brought to "verl pin + managed patches", with per-node provenance. The launch aborts unless every node passes.
- **B. Guard** (`[GEMMA4-ROPE]` block installed by `patch_verl_gemma4_rope_buffers.py` v2): checks at registration, at an all-rank init checkpoint right after the FSDP wrap, and at the first forward.

The fixes themselves are unchanged from PR #8: buffers are sorted before the FSDP wrap, and the RoPE `inv_freq` is rebuilt in fp32. The one difference is that the rebuild now happens at the init checkpoint instead of on the first forward.

---

## 1. Node prep (A)

| Step (per node, one Ray task each, `NodeAffinitySchedulingStrategy(soft=False)`) | Detail |
|---|---|
| Pinned packages | `math-verify==0.9.0`, `latex2sympy2_extended==1.11.0`, `antlr4-python3-runtime==4.9.3`, the versions on the validated pods. `uv pip install` runs only if the installed versions differ |
| Managed files | `verl/models/transformers/monkey_patch.py` (flex tiles + `[GEMMA4-ROPE]` block) and `verl/workers/engine/fsdp/transformer_impl.py` (2-line init-checkpoint hook). Rebuilt from the pin (`git show HEAD:<file>`) + the patchers, and written only if the live file differs |
| Other tracked files | Restored to the pin (`git checkout`), e.g. the PR #8 diagnostic patches, so every node runs exactly "pin + managed patches" |
| Import check | A fresh interpreter of the Ray worker's python imports `verl`, `monkey_patch`, and `transformer_impl`. It checks the module file paths, patch version, block sha256, hook present in `FSDPEngine._build_model_optimizer`, flex patch present, and package versions |
| Provenance (per node) | Ray node id / IP / pod, actual python, verl root and HEAD (and the image's `/workspace/VERL_PIN`), module paths, patch version, block sha256, sha256 of every tracked file that differs from the pin, package versions, actions taken |

The driver reads the patchers once on the head, records their sha256, and ships them to the tasks by value, so worker pods never read them from the FUSE mount. The driver requires a result from every alive node, each from the node it was pinned to. Every result must be ok, and python, verl HEAD, managed-file sha256, module paths, and package versions must be identical across nodes. It writes `<guard_dir>/node_prep.json` and exits 0 only on PASS. The launcher runs under `set -e`, so a FAIL stops the launch.

**Upgrade semantics.** `patch_verl_gemma4_rope_buffers.py` v2 first removes every earlier block (the v1 `[GEMMA4] rank-independent buffer order + fp32 RoPE frequencies` block, or a `[GEMMA4-ROPE]` block of any version) and every earlier hook line, then installs the current version. Run in place on a v1 file, it reports `UPGRADED from v1`. Node prep goes further: it rebuilds from the pin, so nodes end byte-identical whatever their history.

---

## 2. Guard (B)

| Phase | Where | What | On failure |
|---|---|---|---|
| Registration | `apply_monkey_patch()` (wrapped), before the FSDP wrap | A Gemma-4 model must expose ≥ 1 `Gemma4TextRotaryEmbedding` with `rope_init_fns` and both buffers per layer type. One first-forward hook per module; hook count == module count | Raise immediately. Same code and config on every rank, so every rank raises, before any collective |
| Init checkpoint | `FSDPEngine._build_model_optimizer`, right after `_build_fsdp_module()`, **before the optimizer is built** | Every rank classifies each RoPE buffer against an fp32 reference and its bf16 rounding (table below), rebuilds it in fp32, and re-verifies it (dtype float32, bit-exact to the reference, reference agrees with a CPU fp32 recomputation). Local exceptions become a FAIL status, so every rank still reaches the collective | `all_gather_object` of the per-rank results. Rank 0 prints one `GEMMA4-ROPE-GUARD-REPORT {json}` line covering every rank and writes `init_actor_aggregate.json`. If any rank failed, **all ranks raise the same error after the collective**: `init_model` fails and the driver exits non-zero |
| First forward | Forward pre-hook on each rotary module | Re-checks the buffers against the rebuilt reference (catches anything that changes them between init and use) | Never raises inside the forward. It restores the reference and writes a per-rank record; `gemma4_guard_check.py` fails the run's evidence if any rank restored something |

Evidence goes to `$VERL_GEMMA4_ROPE_REPORT_DIR`: per-rank `init_actor_rank<RRRR>.json` and `first_forward_actor_rank<RRRR>.json`, plus `init_actor_aggregate.json`. The launcher sets the directory to `/workspace/meta-RL/logs/gemma4_guard/<EXPERIMENT_NAME>/` via the Ray `runtime_env`, next to `node_prep.json`. Per-rank stdout lines are for humans only: Ray de-duplicates lines that differ only in numbers (`[repeated 15x across cluster]`), so the evidence is keyed by rank.

### 2.1 Buffer classification (per buffer, every entry)

| State | Condition | Accepted before rebuild |
|---|---|---|
| `fp32` | $\lvert c-r\rvert \le 10^{-10} + 4\cdot10^{-6}\,\lvert r\rvert$ | yes |
| `bf16` | $c$ is exactly bf16-representable **and** $\lvert c-\mathrm{bf16}(r)\rvert \le 10^{-10} + 2^{-7}\,\lvert \mathrm{bf16}(r)\rvert$ (≤ 1 bf16 ulp) | yes (expected: verl casts the module to `model_dtype=bfloat16`) |
| `mismatch` | anything else | no |
| `nonfinite` | any NaN / Inf | no |
| `shape_mismatch` | shape differs from the reference | no |

Here $c$ is the current buffer and $r$ the fp32 reference, recomputed with the module's own init function and kwargs. Relative errors are reported over non-zero reference entries only. A zero reference entry can only match through the absolute floor ($10^{-10}$); Gemma-4's proportional RoPE pads `full_attention_inv_freq` with 192 zeros. After the rebuild, every buffer must be `fp32`, dtype float32, and bit-exact to the reference, and the device reference must agree with a CPU fp32 recomputation within the `fp32` tolerance.

Why not one relative threshold: a legitimately bf16-rounded buffer is up to 0.37% off the fp32 values here, while a 0.2% error on fp32 values is not a rounding artifact. The classifier accepts the first and flags the second (unit test below).

### 2.2 Test-only fault injection

`VERL_GEMMA4_ROPE_FAULT_RANK=<global rank>` plus `VERL_GEMMA4_ROPE_FAULT_MODE`:
- `reverse_order` (default): reverses that rank's RoPE buffer order *before* the FSDP wrap, so the real FSDP1 positional broadcast scrambles it. This is the original bug, made deterministic.
- `scale`: ×1.05 after the sync.
- `nan`: one NaN after the sync.

The guard report records the fault configuration, and the per-rank record names the rank where it was applied.

---

## 3. Evidence

### 3.1 Provenance & SHA256s (16 Ray nodes / 64 GB200 GPUs)

| Item | Value / SHA256 |
|---|---|
| `verl` root & `HEAD` (all 16 nodes) | `/workspace/verl-pin` @ `ace775e8bc192836a3e9cfddbc3fa6f31de2c6ee` (`/workspace/VERL_PIN` matches) |
| Python executable (all 16 nodes) | `/workspace/verl-pin/.venv/bin/python` (`3.12.12`) |
| Pinned packages (all 16 nodes) | `math-verify==0.9.0`, `latex2sympy2_extended==1.11.0`, `antlr4-python3-runtime==4.9.3` |
| [`patch_verl_gemma4_rope_buffers.py`](patch_verl_gemma4_rope_buffers.py) | `6d22ce33a767c1029bf94264ffeb9bb07bf13843e1f80939f6f1ef985ffaf5e2` (`version="2"`, `block_sha256="2d750b2b0c718b581831c6aa9130717e5dcf47e5938715de58caa2d42ae0b8f4"`) |
| [`patch_verl_gemma4_flex_attn.py`](patch_verl_gemma4_flex_attn.py) | `4533443330c29e37eb1454849dc4bb23446c179ff77c3e5a3a9d6a5648863596` |
| [`gemma4_node_prep.py`](gemma4_node_prep.py) | `65b572dd6d2b4f36b7936404c6587da740a7716ca490def44d001897000732ac` |
| [`gemma4_guard_check.py`](gemma4_guard_check.py) | `240f4f49cf38c50f1c91db8555f2aa373e76dbcc08d67d5dae04da709a3ada9a` |
| [`test_gemma4_rope_guard.py`](test_gemma4_rope_guard.py) | `419ae1485781f3ef9c358470561e820927aa8f854d95e272051e3b73c6c99207` |
| [`run_gemma4_e2b_rl05_parity.sh`](run_gemma4_e2b_rl05_parity.sh) | `8646c17331d84ac95a5183ffae4880f2150889a864fe1ff88ba5c79390ea66d0` |
| Managed `verl/models/transformers/monkey_patch.py` (all 16 nodes) | `9be46b096f096ab17596f91880929398b9919179a74851a1505e23a5803bfbb7` |
| Managed `verl/workers/engine/fsdp/transformer_impl.py` (all 16 nodes) | `9a43adb81f9e8d39dc77a715e341190778da43f95152127f8a43e5f6c08c3263` |

### 3.2 Patcher upgrade & idempotence test

Exact transcript in [`rope_guard/patcher_upgrade_test.txt`](rope_guard/patcher_upgrade_test.txt):
- First install on `ace775e8` + flex reports `APPLIED (version 2)` on both `monkey_patch.py` and `transformer_impl.py`.
- Second run reports `UP-TO-DATE (version 2)` without touching mtimes.
- Running on a pod file carrying the PR #8 `v1` block (`# [GEMMA4] rank-independent buffer order + fp32 RoPE frequencies`) strips the `v1` block (`0` v1 markers remaining) and installs `v2` (`UPGRADED from v1`).

### 3.3 Single-process deterministic unit tests (`test_gemma4_rope_guard.py`)

Ran on `/workspace/meta-RL/models/gemma-4-E2B-it` (`Gemma4TextRotaryEmbedding` with `full_attention` and `sliding_attention`). Full JSON output in [`rope_guard/unit_test.json`](rope_guard/unit_test.json) (**14 / 14 PASS**):

1. `classify_exact_fp32`: exact FP32 reference classified as `fp32` (`max_rel_err_fp32 = 0.0`, `zero_ref_count = 192` on `full_attention_inv_freq`).
2. `classify_fp32_sub_tolerance`: $2\cdot10^{-6}$ relative perturbation within `rtol_fp32 = 4e-6` classified as `fp32`.
3. `classify_bf16_rounded`: `.to(bfloat16).to(float32)` classified as `bf16` (`max_rel_err_fp32 = 0.003653`, `max_rel_err_bf16 = 0.0`).
4. `classify_rejects_fp32_at_0p2pct`: $0.2\%$ relative perturbation on non-BF16-representable FP32 tensor rejected as `mismatch` (demonstrates why a flat $0.4\%$ threshold is not used).
5. `classify_rejects_scrambled_layer_type`: `sliding_attention` values placed in `full_attention` buffer rejected as `mismatch` (`max_rel_err_fp32 = 0.678`).
6. `classify_rejects_nonzero_at_zero_ref`: $10^{-5}$ placed at a zero-padded proportional-RoPE entry rejected as `mismatch` via the `atol = 1e-10` floor.
7. `classify_rejects_nan_and_inf`: `NaN` and `Inf` entries rejected as `nonfinite`.
8. `classify_rejects_shape_mismatch`: truncated buffer rejected as `shape_mismatch`.
9. `check_module_rebuilds_bf16_to_bit_exact_fp32`: after `.to(bfloat16)`, `_g4_check_module(..., rebuild=True)` restores bit-exact FP32 (`post.state = "fp32"`, `bit_exact = True`, `cpu_oracle_state = "fp32"`).
10. `register_guard_and_checkpoint_pass`: `_g4_register_guard` + `_verl_gemma4_rope_init_checkpoint` + forward pass produce `init_actor_rank0000.json`, `init_actor_aggregate.json`, and `first_forward_actor_rank0000.json` with `status = "PASS"`.
11. `register_guard_fails_when_rotary_missing`: Gemma-4 model without `Gemma4TextRotaryEmbedding` raises `RuntimeError` at registration.
12. `simulated_fsdp_sync_without_sort_caught`: reversing buffer registration order and copying rank-0 buffers by position (simulating FSDP1 `sync_module_states`) causes `_verl_gemma4_rope_init_checkpoint` to raise `RuntimeError` and record `status = "FAIL"`.
13. `fault_injection_modes`: `reverse_order`, `scale`, and `nan` fault modes all trigger `RuntimeError` at `_verl_gemma4_rope_init_checkpoint`.
14. `first_forward_hook_restores_post_init_corruption`: corrupting a buffer after the init checkpoint is caught and restored by `_g4_first_forward_pre_hook` (`status = "FAIL"`, `restored = ["full_attention_inv_freq"]`).

### 3.4 16-rank positive 2-step smoke (`4n16g`)

- **Run:** `gemma4_e2b_4n16g_guard_smoke2_1021_1926`
- **Log:** [`logs/gemma4_e2b_4n16g_guard_smoke2_1021_1926.log`](logs/gemma4_e2b_4n16g_guard_smoke2_1021_1926.log)
- **Structured evidence:** [`rope_guard/positive_smoke/`](rope_guard/positive_smoke/) (`node_prep.json`, `init_actor_aggregate.json`, `guard_check.json`, `guard_check.txt`)
- **Summary (`gemma4_guard_check.py`, 16 / 16 checks PASS):**
  - `node_prep`: `PASS` across all 16 alive Ray nodes (16 GPU nodes), identical managed file SHA256s (`9be46b096f09…`, `9a43adb81f9e…`).
  - `init_checkpoint`: all 16 unique ranks (`0..15`) wrote `init_actor_rank<RRRR>.json` and passed (`n_pass = 16, n_fail = 0`); all 64 buffers (4 per rank) were `bf16` before rebuild (`max_rel_err_vs_fp32 = 0.003653`) and `fp32` bit-exact after rebuild.
  - `first_forward`: all 16 unique ranks (`0..15`) wrote `first_forward_actor_rank<RRRR>.json` with `status = "PASS"` and `restored = []` (whereas Ray's worker stdout stream in the driver log only surfaced one rank's line due to worker log deduplication/buffering).
  - Training completed 2 steps cleanly (`step 1`: `rollout_corr/k3_kl = 0.000684`, `actor/grad_norm = 0.198`; `step 2`: `rollout_corr/k3_kl = 0.000718`, `actor/grad_norm = 0.334`) and exited with `exit_code = 0`.

### 3.5 16-rank deterministic negative test (`4n16g`, rank-5 fault injection)

- **Run:** `gemma4_e2b_4n16g_guard_negative_r5_1021_1936`
- **Command overrides:** `++ray_kwargs.ray_init.runtime_env.env_vars.VERL_GEMMA4_ROPE_FAULT_RANK="5" ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_GEMMA4_ROPE_FAULT_MODE="reverse_order"`
- **Log:** [`logs/gemma4_e2b_4n16g_guard_negative_r5_1021_1936.log`](logs/gemma4_e2b_4n16g_guard_negative_r5_1021_1936.log)
- **Structured evidence:** [`rope_guard/negative_fault_rank5/`](rope_guard/negative_fault_rank5/) (`node_prep.json`, `init_actor_aggregate.json`, `init_actor_rank0005.json`, `guard_check.json`, `guard_check.txt`)
- **Summary (`gemma4_guard_check.py --expect-fail-rank 5`, 15 / 15 checks PASS):**
  - Rank 5 reversed its RoPE buffer order before the FSDP1 wrap (`fault_applied = "reverse_order(before FSDP sync)"`), so FSDP1 positional broadcast overwrote rank 5's buffers with the wrong layer types (`full_attention_inv_freq` `max_rel_err = 0.678`, `sliding_attention_inv_freq` `max_rel_err = 1.0`, `sliding_attention_original_inv_freq` `max_rel_err = 2.113`).
  - All 16 unique ranks (`0..15`) wrote `init_actor_rank<RRRR>.json` and reached the `all_gather_object` checkpoint (`n_pass = 15, n_fail = 1, failed_ranks = [5]`).
  - Every rank raised `RuntimeError: [GEMMA4-ROPE] guard FAILED on 1/16 rank(s) (role=actor, phase=init_checkpoint, before the optimizer is built)` right after the collective, before `_build_optimizer`.
  - No `first_forward_actor_rank*.json` files were written, zero training steps or optimizer updates occurred (`steps = []`), the driver exited with `exit_code = 1`, and all 64 GPUs returned cleanly to idle.

---

## 4. Limits and follow-ups

- Verified for the FSDP1 engine path this recipe uses (`actor.strategy=fsdp`, no ref model since $\beta=0$). The guard also runs for a ref engine (`role=ref`), but no ref run was exercised. The FSDP2 path has no positional buffer broadcast and was not exercised either.
- **transformers ≥ 5.16** registers the buffers in sorted order ([v5.16.0 `modeling_gemma4.py` L1092](https://github.com/huggingface/transformers/blob/v5.16.0/src/transformers/models/gemma4/modeling_gemma4.py#L1092)), but verl main still casts the module to `model_dtype` after `apply_monkey_patch()` ([`transformer_impl.py` L328–337](https://github.com/verl-project/verl/blob/f2923cbf154731918b1e36db8c567433f349fede/verl/workers/engine/fsdp/transformer_impl.py#L328-L337)). After an upgrade, the bf16 cast must be re-validated; the guard reports it as `bf16` before the rebuild.
- **C. Bake into the image**: deferred to the next image rebuild. It must be a separate derived image, because the shared `verl-disagg` image is documented as "no fork, no patches". Node prep then only verifies.
- **D. Upstream**: the verl-side bf16 cast of non-persistent fp32 buffers can be reported in parallel; the transformers ordering is already fixed in 5.16.
- The PR #8 diagnostic patchers (`patch_verl_pin_logprob_fixture.py`, `patch_verl_pin_rank_probe.py`) are not part of the managed set. Node prep restores the files they touch, so a diagnostic run has to apply them after node prep and start `main_ppo` directly.
