#!/usr/bin/env bash
# GRPO step-time benchmark on 64 GB200, colocated verl (upstream pin in the verl-pin image).
# usage: MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B TP=2 [STEPS=11] [GRPO_DATA=...] bash run_grpo_colocated.sh [extra hydra overrides]
# Workload: 256 prompts x 8 generations = 2048 completions / step, one update per step (mu=1); T 0.8, top-k 50, top-p 0.95;
#   response cap 8192, prompt cap 4096; GRPO, beta=0 (no reference model), PPO clip 0.2 / 0.28 (clip-higher), token-mean loss;
#   no eval, no checkpoint. The verl steps are 1-based; the default 11 steps time steps 4-11 (1-3 warmup).
# Policy loss: SINGLE_FWD=1 (default) = single forward: the sampler's per-token log-probs are the old log-probs
#   (bypass_mode, PPO clip against them, no IS weights), so the trainer skips the old-logprob forward pass.
#   SINGLE_FWD=0 = the two-pass default (the trainer recomputes old log-probs), as in the earlier TP sweeps.
# Optimizer: as MaxText rl.yml/base.yml (the TPU reference does not override them): AdamW betas (0.9, 0.99), wd 0.1,
#   lr 1e-6, linear warmup over 10% of the steps, cosine decay to 10% of the peak; grad clip 1.0.
# System knobs (report them): TP (DP = 64/TP), GPU_MEM_UTIL (vLLM share, default 0.6), PPO_MAX_TOK (dynamic batching).
# Logs: the console log goes to $RUN_DIR/driver.log; TensorBoard is written to node-local disk (gcsfuse appends block the
#   trainer) and copied to $RUN_DIR/tensorboard when the run ends (collect_dir.py; it can also be run by hand mid-run).
set -euo pipefail
: "${MODEL_PATH:?}" "${TP:?}" "${DISAGG_PYTHON:?}" "${VERL_REPO:?}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
DATA=${GRPO_DATA:-/workspace/meta-RL/data/grpo_perf/omi2_5120.parquet}
LOG_ROOT=${GRPO_LOG_DIR:-/workspace/meta-RL/logs/grpo_perf}; mkdir -p "$LOG_ROOT"
M=$(basename "$MODEL_PATH"); EXP=${EXPERIMENT_NAME:-grpo_${M}_tp${TP}_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=$LOG_ROOT/$EXP; mkdir -p "$RUN_DIR"
TB_LOCAL=${TB_LOCAL_ROOT:-/tmp/tb_local}/$EXP
STEPS=${STEPS:-11}; MEM=${GPU_MEM_UTIL:-0.6}; MAXTOK=${PPO_MAX_TOK:-32768}
WARMUP=${LR_WARMUP_STEPS:-$(( (STEPS + 9) / 10 ))}          # 10% of the steps, rounded up (11 -> 2, 200 -> 20)
RE=ray_kwargs.ray_init.runtime_env

# The data is read in file order (shuffle off) for exactly one epoch: it must hold STEPS x 256 prompts, or the run stops early.
ROWS=$("$DISAGG_PYTHON" -c "import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)" "$DATA")
if [ "$ROWS" -lt $(( STEPS * 256 )) ]; then
  echo "[grpo] ABORT: $DATA has $ROWS rows; $STEPS steps need $(( STEPS * 256 )). Build a larger file with prep_omi2_data.py --n."; exit 2
fi

SF_ARGS=()
if [ "${SINGLE_FWD:-1}" = "1" ]; then
  SF_ARGS=(actor_rollout_ref.rollout.calculate_log_probs=true
           algorithm.rollout_correction.bypass_mode=true algorithm.rollout_correction.loss_type=ppo_clip
           actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode
           '+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}')
fi

archive_tb() { "$DISAGG_PYTHON" "$SCRIPT_DIR/collect_dir.py" "$TB_LOCAL" "$RUN_DIR/tensorboard" \
                 || echo "[grpo] WARN: TensorBoard copy failed; the events stay under $TB_LOCAL on the node that ran the trainer"; }
trap archive_tb EXIT
echo "[grpo] run $EXP: model $M TP=$TP steps=$STEPS single_fwd=${SINGLE_FWD:-1} warmup=$WARMUP clip=0.2/0.28 mem=$MEM ppo_max_tok=$MAXTOK data=$DATA ($ROWS rows) tb=$TB_LOCAL" | tee "$RUN_DIR/recipe.txt"

"$DISAGG_PYTHON" -m verl.trainer.main_ppo \
  data.train_files=$DATA data.val_files=$DATA data.train_batch_size=256 data.shuffle=false \
  data.max_prompt_length=4096 data.max_response_length=8192 data.filter_overlong_prompts=true data.truncation=error data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=$MODEL_PATH actor_rollout_ref.model.use_remove_padding=true actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAXTOK \
  actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=$WARMUP actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1 actor_rollout_ref.actor.optim.weight_decay=0.1 'actor_rollout_ref.actor.optim.betas=[0.9,0.99]' \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=$TP actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=0.8 actor_rollout_ref.rollout.top_k=50 actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$MEM actor_rollout_ref.rollout.max_model_len=12288 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true actor_rollout_ref.rollout.enable_prefix_caching=true \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAXTOK \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAXTOK \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  reward.custom_reward_function.path=${GRPO_REWARD:-$SCRIPT_DIR/omi2_reward.py} reward.custom_reward_function.name=compute_score reward.num_workers=8 \
  trainer.nnodes=16 trainer.n_gpus_per_node=4 trainer.total_epochs=1 trainer.total_training_steps=$STEPS \
  trainer.test_freq=-1 trainer.save_freq=-1 trainer.val_before_train=false 'trainer.logger=[console,tensorboard]' \
  trainer.project_name=grpo_perf trainer.experiment_name=$EXP trainer.default_local_dir=$RUN_DIR/ckpt trainer.balance_batch=true \
  $RE.py_executable=$DISAGG_PYTHON "+$RE.env_vars.PYTHONPATH=$VERL_REPO:$SCRIPT_DIR" "+$RE.env_vars.TENSORBOARD_DIR=$TB_LOCAL" \
  "+$RE.env_vars.VLLM_NO_USAGE_STATS='1'" "+$RE.env_vars.DO_NOT_TRACK='1'" \
  ${SF_ARGS[@]+"${SF_ARGS[@]}"} "$@" 2>&1 | tee "$RUN_DIR/driver.log"
echo "[grpo] run dir $RUN_DIR"
