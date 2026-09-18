# my_scripts_GKE_gsm8k — Meta GSM8K-boxed reproduction (Qwen3-0.6B-Base, GRPO, clip-higher)

Goal: reproduce Meta's GPU reference run `qwen3-0p6b-base-boxed-cliphigh-0c2e52db` (8x GB200, Meta's
trainer) on verl, match its curves, then hand the frozen recipe to the TPU side. Same methodology as
`my_scripts_GKE` (rulebook, gates, band), different recipe.

## Files
| file | purpose |
|---|---|
| `build_gsm8k_boxed_data.py` | Meta jsonl -> verl parquets (train 7,473; test512 = Meta's eval set = head; test 1,319 full), model copy with the frozen stop set (eos [151645,151643]), prompt fixture, token-exact check against Meta's prompt_example.json, MANIFEST |
| `boxed_math_reward.py` | Meta's `boxed_math` per REPRODUCTION.md §1 (exclusive 1.0 / 0.1 / 0, minimal normalization, last box, unbalanced -> 0) + overlong penalty (training sources only); embeds and passes Meta's 21 fixtures |
| `run_qwen3_0p6b_base_gsm8k_boxed.sh` | launcher; FROZEN block = Meta's yaml + REPRODUCTION.md resolved defaults (token IS 3.0, dual clip 5.0, wd 0, token-mean); three pre-flights |
| `compare_to_meta.py` | overlay ours vs Meta's `train_metrics.csv` / `eval_metrics.csv` (named columns): train-batch acc, eval acc, plus length / cap-hit / zero-std / mean-reward channels |
| `collapse_guard.py`, `plot_phase0.py`, `band_plot.py`, `paired_eval_bootstrap.py` | shared tools, now parameterized (`--groups/--group_size`, `--metrics`, `--sources`); these copies supersede the `my_scripts_GKE` ones |
| `patch_verl_dump_uid.py`, `patch_verl_reward_response_len.py`, `patch_verl_logprob_fixture.py` | fork patches (already applied on the pod; launcher only checks) |

## Meta config -> verl mapping (FROZEN; sources: configs yaml + REPRODUCTION.md v1.0)
| Meta | verl arg | source |
|---|---|---|
| model Qwen/Qwen3-0.6B-Base, its own ChatML template, system+user | `MODEL_PATH` = copy with eos_token_id [151645,151643] | prompt_example.json (99 ids for test row 0, checked by the builder) |
| stop: `vllm_stop_token_ids [151645]` + native EOS 151643 both terminate | eos list above; `ignore_eos` False | REPRODUCTION.md §5 |
| global_batch 128 x num_generations 16 = 2048; ppo_epochs 1 | `train_batch_size=128 rollout.n=16 ppo_mini_batch_size=128 ppo_epochs=1` | yaml |
| micro_batch_size 8 (per GPU), no dynamic batching | `use_dynamic_bsz=False ppo_micro_batch_size_per_gpu=8` | yaml |
| prompt 512 / completion 2048 | `max_prompt_length=512 max_response_length=2048` | yaml |
| AdamW lr 2e-5, cosine to 0 at 250, warmup 10; betas (0.9,0.999), eps 1e-8, **wd 0.0**, fused False, clip 1.0 | `optim.lr=2e-5 lr_scheduler_type=cosine lr_warmup_steps=10 min_lr_ratio=0 num_cycles=0.5 betas=[0.9,0.999] weight_decay=0.0 clip_grad=1.0` (verl eps default 1e-8; not fused under FSDP) | REPRODUCTION.md §2 |
| fp32 params + bf16 autocast | `fsdp_config.model_dtype=fp32` + FSDP bf16 compute | yaml |
| GRPO advantages: group mean, / (std ddof=1 + 1e-6), broadcast, no batch norm | verl grpo (`norm_adv_by_std_in_grpo=True`, torch std unbiased, eps 1e-6) | §3 -- matches |
| loss_agg "token" (global token mean) | `loss_agg_mode=token-mean` | §2 |
| clip 0.2/0.28, dual clip 5.0 -- all inert (ratio == 1 by construction) | `clip_ratio_low=0.2 clip_ratio_high=0.28 clip_ratio_c=5.0` | §3 |
| old_log_probs = new.detach() (no separate pass) | verl recomputes old with the trainer (separate no-grad pass); ratio == 1 exactly (measured repeat error 0). Same gradient; extra time to account for in perf | §3 |
| token-level IS: w = min(exp(clamp(logp_actor - logp_rollout, ±20)), 3.0), no renorm, no RS | `rollout_correction.rollout_is=token rollout_is_threshold=3.0 rollout_rs=null rollout_is_batch_normalize=False bypass_mode=False` (weight = exp(clamp(old_trainer - rollout, ±20)) truncated at 3.0, detached; verl's backend has the same ±20 guard) | §4; **PENDING: is Meta's w detached?** |
| kl_coeff 0, no reference model | `use_kl_loss=False` | yaml |
| temperature 1.0; top_p / top_k / seed UNSET (backend = vLLM defaults) | `temperature=1.0 top_p=1.0 top_k=-1` (vLLM defaults, same backend) | §2 gap 3 |
| eval: rows 0-511 of test.jsonl (head), greedy, n=1, 2048 tokens, every 20 steps, raw reward (no overlong penalty) | test512 parquet (head), `val_kwargs` greedy n=1, `test_freq=20`; reward applies the penalty only to `gsm8k_boxed_train` | §5 |
| boxed_math: last `\boxed{`, brace-match, unbalanced -> 0; normalize strip/rstrip('.')/no ','/no '$'; 1.0 / 0.1 / 0 exclusive; `18.0`!=`18`; `\boxed{}` -> 0 | `boxed_math_reward.py` -- passes all 15 + 6 fixtures | §1 + reward_fixtures.json |
| overlong: min(0, -(len-1536)/512), added to the training reward | `REWARD_OVERLONG_BUFFER=512 REWARD_OVERLONG_PENALTY=1.0 REWARD_MAX_RESP_LEN=2048` | §1 |
| 8x GB200, single node | `NNODES=2 GPUS_PER_NODE=4` (two GKE nodes; NVLink-domain difference is a perf note, not a recipe difference) | — |
| controller pipelined (generation of batch t+1 overlaps training of batch t) | verl is synchronous colocated. **PENDING: which weights generate batch t+1 (one-step lag?)** | §7 |

Confirmed by Meta and no longer PENDING: eval rows = head 512; stop set = both ids; weight decay 0; token-mean; ddof=1; format
credit exclusive; `reward/accuracy` in the screenshot is the training-rollout channel, not eval.

## Still PENDING with Meta (asked)
1. Gold normalization: text says gold is not normalized, fixture 2 requires the gold's "," removed. We follow the fixture.
2. Whether the IS weight is detached / no_grad (decides the gradient: detached -> -adv*w*grad logpi, which is what verl's TIS does).
3. Pipelining: policy version used to generate each batch; any lag. Affects the algorithm and the perf comparison.
4. Timing figures (withheld pending owner approval).

## Run
```bash
export DATA_DIR=/workspace/meta-RL/data/gsm8k_boxed MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B-Base-stop META=/workspace/meta-RL/meta_pkg
python3 build_gsm8k_boxed_data.py --meta_data $META/data --out $DATA_DIR \
    --model_in /workspace/meta-RL/models/Qwen3-0.6B-Base --model_out $MODEL_PATH --prompt_example $META/reference/prompt_example.json
python3 boxed_math_reward.py --fixtures $META/reference/reward_fixtures.json     # 21/21 + penalty scope; non-zero exit on failure
TOTAL_STEPS=3 TEST_FREQ=1 SAVE_FREQ=2 bash run_qwen3_0p6b_base_gsm8k_boxed.sh 2>&1 | tee $LOG_DIR/meta_smoke.log
SEED=1 bash run_qwen3_0p6b_base_gsm8k_boxed.sh 2>&1 | tee $LOG_DIR/meta_boxed_seed1_$(date +%m%d_%H%M).log     # full 250-step cosine schedule; compare the first 80
python3 compare_to_meta.py --tb $TB_DIR --rollout $ROLLOUT_DUMP_DIR --meta_train $META/reference/train_metrics.csv --meta_eval $META/reference/eval_metrics.csv --out cmp.png
```
The launcher runs three pre-flights before touching the GPUs: Meta's reward fixtures, the collapse guard config (128x16), and a
hydra `--cfg job` render of the **full launch argument list** (`TRAIN_ARGS`, the same array the launch uses) written to
`resolved_config_preflight.txt`; it must contain the IS / dual-clip / micro-batch / wd / LR / warmup / batch / steps / length values above.
`compare_to_meta.py` computes `frac_zero_std` from the training reward (`score` = raw + penalty, what advantages see), and reports the
raw-reward mean separately.
Analysis tools take the batch shape: `plot_phase0.py --groups 128 --group_size 16 --cap 2048`, `collapse_guard.py --groups 128 --group_size 16`,
`band_plot.py --steps 0:240:20,250 --metrics 'val-core/gsm8k_boxed_test512/acc/mean@1=GSM8K test512 acc (greedy)'`,
`paired_eval_bootstrap.py --key qid --sources gsm8k_boxed_test512=... gsm8k_boxed_test=...` (no pooled row across overlapping sets).

## What the reference actually is (read before comparing)
- One unseeded run; Meta has no multi-seed band. Logs cover steps 1-80 (train) and evals at 0/20/40/60 only; the run was configured for 250.
- Two channels, never to be mixed: `reward/accuracy` (training rollouts, T=1, 16/prompt, every step; 0.255 at step 1, ~0.80 by step 16)
  and `eval/accuracy` (512 held-out, greedy; 0.5527 at step 0, 0.7363 / 0.7129 / 0.7500 at 20 / 40 / 60).
- `reward/frac_zero_std` climbs from 0 to ~0.6-0.74 by step 40-80 with no dynamic sampling: three-quarters of the batch is inert late.
  Our run must show the same climb; enabling any group filtering breaks comparability.
- Step 0 eval is the best first gate: same 512 rows, greedy, same reward -> our test512 step-0 accuracy should be ~0.553 (deterministic up to
  numerics). Step-1 train accuracy ~0.25 is a weaker check (sampled, one draw).
- Eval accounting: each of our evals runs 512 + 1,319 questions; `EVAL_FULL=0` for Meta-comparable timing. Meta's step_time is pipelined
  wall-clock; phase times do not sum to it.

## Acceptance -- PENDING
To be declared in writing with Henry after the smoke run and before any TPU number: per-step train-accuracy and per-checkpoint eval
differences vs Meta's single run, with the caveat that Meta's own seed-to-seed spread is unknown. `compare_to_meta.py` prints binomial
2-sigma scales as reference only. Three seeds on our side for the band, as in Track A.