# my_scripts_GKE_gsm8k — Meta GSM8K-boxed reproduction (Qwen3-0.6B-Base, GRPO, clip-higher)

Goal: reproduce Meta's GPU reference run `qwen3-0p6b-base-boxed-cliphigh-0c2e52db` (8x GB200, Meta's
trainer) on verl, match its curves, then hand the frozen recipe to the TPU side. Same methodology as
`my_scripts_GKE` (rulebook, gates, band), different recipe.

## Files
| file | purpose |
|---|---|
| `build_gsm8k_boxed_data.py` | Meta jsonl -> verl parquets (train 7,473; test512 = Meta's eval set; test 1,319 full), model copy with the frozen stop set (eos [151645,151643]), prompt fixture, MANIFEST |
| `boxed_math_reward.py` | **PROVISIONAL** port of Meta's `boxed_math` (1.0 / format_score 0.1 / 0) + DAPO overlong penalty (512 on cap 2048). Replace with Meta's verbatim code when received; rules marked GUESS |
| `run_qwen3_0p6b_base_gsm8k_boxed.sh` | launcher; FROZEN block = Meta's yaml; unspecified items are env knobs (see below) |
| `compare_to_meta.py` | overlay our train-batch acc / eval acc against Meta's TensorBoard CSV export |
| `collapse_guard.py`, `plot_phase0.py`, `band_plot.py`, `paired_eval_bootstrap.py` | shared tools, now parameterized (`--groups/--group_size`, `--metrics`, `--sources`); these copies supersede the `my_scripts_GKE` ones |
| `patch_verl_dump_uid.py`, `patch_verl_reward_response_len.py`, `patch_verl_logprob_fixture.py` | fork patches (already applied on the pod; launcher only checks) |

## Meta yaml -> verl mapping (FROZEN)
| Meta | verl arg |
|---|---|
| model Qwen/Qwen3-0.6B-Base | `MODEL_PATH` = patched copy (eos_token_id [151645,151643] reproduces `vllm_stop_token_ids: [151645]` + tokenizer eos) |
| global_batch_size 128, num_generations 16 | `data.train_batch_size=128`, `rollout.n=16` (2048 seq/step) |
| ppo_epochs 1 | `ppo_mini_batch_size=128`, `ppo_epochs=1` (mu=1) |
| micro_batch_size 8 (per GPU) | `use_dynamic_bsz=False`, `ppo_micro_batch_size_per_gpu=8` |
| max_seq_length 2560 | `max_prompt_length=512`, `max_response_length=2048` |
| lr 2e-5, cosine, warmup 10, max_steps 250 | `optim.lr=2e-5 lr_scheduler_type=cosine lr_warmup_steps=10 min_lr_ratio=0 num_cycles=0.5`, `total_training_steps=250` |
| max_grad_norm 1.0 | `optim.clip_grad=1.0` |
| bf16: true (mixed precision) | `fsdp_config.model_dtype=fp32` + FSDP bf16 compute (verl default) |
| clip 0.2 / 0.28, kl_coeff 0 | `clip_ratio_low/high`, `use_kl_loss=False` |
| temperature 1.0 | `rollout.temperature=1.0` |
| eval: greedy, n=1, 512 samples, every 20 steps | `val_kwargs` do_sample False / T 0 / n 1, `test_freq=20`, val files test512 (+ full test as our diagnostic) |
| overlong_buffer 512 / 1.0 | `REWARD_OVERLONG_BUFFER=512 REWARD_OVERLONG_PENALTY=1.0 REWARD_MAX_RESP_LEN=2048` |
| boxed_math format_score 0.1 | `REWARD_FORMAT_SCORE=0.1` |
| 8x GB200 | `NNODES=2 GPUS_PER_NODE=4` |

## Not specified by Meta -> env knobs, CONFIRM before freezing
`ROLLOUT_TOP_P` (1.0), `ROLLOUT_TOP_K` (-1), `WEIGHT_DECAY` (0.0 = HF default), `LOSS_AGG_MODE` (token-mean),
chat-template thinking flag (tokenizer default), whether eval uses the first 512 or a random 512.

## Run
```bash
export DATA_DIR=/workspace/meta-RL/data/gsm8k_boxed MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B-Base-stop
python3 build_gsm8k_boxed_data.py --meta_data /workspace/meta-RL/meta_pkg/data --out $DATA_DIR \
    --model_in /workspace/meta-RL/models/Qwen3-0.6B-Base --model_out $MODEL_PATH      # refuses to touch --model_in
python3 boxed_math_reward.py                                        # self-test (PROVISIONAL rules; non-zero exit on failure)
TOTAL_STEPS=3 TEST_FREQ=1 SAVE_FREQ=2 bash run_qwen3_0p6b_base_gsm8k_boxed.sh 2>&1 | tee $LOG_DIR/meta_smoke.log
SEED=1 bash run_qwen3_0p6b_base_gsm8k_boxed.sh 2>&1 | tee $LOG_DIR/meta_boxed_seed1_$(date +%m%d_%H%M).log
```
The launcher ignores the Track A environment (`SCRIPTS_DIR`, `REWARD_FN_PATH`, `REWARD_*`) and sets its own; overrides use
`META_*` names only. Its pre-flight checks the reward at 1536 -> 0, 1792 -> -0.5, 2048 -> -1 and that the buffer is 512,
and verifies the collapse guard started with `--groups 128 --group_size 16`.

Analysis tools take the batch shape explicitly for this recipe:
`plot_phase0.py --groups 128 --group_size 16`, `collapse_guard.py --groups 128 --group_size 16`,
`band_plot.py --metrics 'val-core/gsm8k_boxed_test512/acc/mean@1=GSM8K test512 acc (greedy)' ...`,
`paired_eval_bootstrap.py --sources gsm8k_boxed_test512=$DATA_DIR/gsm8k_boxed_test512.parquet`.

## Eval accounting
Each eval runs 512 (Meta's set) + 1,319 (our full-set diagnostic) = 1,831 questions. For wall-time comparisons with
Meta, either use `EVAL_FULL=0` (512 only) or exclude eval time; steady-state step time excludes eval either way.

## Status of assumptions -- all PENDING until Meta confirms
- The screenshot's `reward/accuracy` (first point ~0.25) is assumed to be train-batch accuracy at T=1; its identity
  (accuracy vs total reward; sampled vs greedy) is unconfirmed, so it is NOT yet a gate.
- "The first 512 test rows are Meta's eval set" (max_eval_samples 512) is assumed, not confirmed.
- Stop set: Meta `vllm_stop_token_ids=[151645]` + tokenizer eos 151643 assumed to equal our patched eos list;
  confirm Meta's engine also stops on 151643.
- top_p / top_k / weight_decay / loss aggregation / chat-template thinking flag: our defaults, unconfirmed.
- The boxed_math extraction and normalization rules are a provisional reading; Meta's code replaces them.

## Acceptance -- PENDING
Numbers for "verl reproduces Meta's trainer on the same GPU" (per-step train-acc and per-checkpoint eval differences)
will be declared in writing after Meta's CSV export and reward code arrive and before any TPU number is inspected.
`compare_to_meta.py` prints binomial 2-sigma values for the batch sizes as a reference scale only; they are not
run-to-run noise estimates and are not acceptance thresholds. Three seeds for the band, as in Track A.