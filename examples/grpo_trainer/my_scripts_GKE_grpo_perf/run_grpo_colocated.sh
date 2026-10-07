#!/usr/bin/env bash
# GRPO step-time benchmark on 64 GB200, colocated verl (upstream pin in the verl-pin image).
# usage: MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B TP=1 bash run_grpo_colocated.sh [extra hydra overrides]
# Frozen workload: 256 prompts x 8 generations = 2048 completions / step, one update per step, 11 steps (verl steps are 1-based: 1-3 warmup, 4-11 timed),
# T 0.8 top-k 50 top-p 0.95, response cap 8192, prompt cap 4096, lr 1e-6 warmup 2 cosine, AdamW(0.9,0.99) wd 0.1, clip 1.0,
# GRPO beta=0, PPO clip 0.2 (ratio == 1 at mu=1), no eval, no checkpoint. Swept: TP (DP = 64/TP). Reported knobs: memory split, batching.
set -euo pipefail
: "${MODEL_PATH:?}" "${TP:?}" "${DISAGG_PYTHON:?}" "${VERL_REPO:?}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
DATA=${GRPO_DATA:-/workspace/meta-RL/data/grpo_perf/omi2_5120.parquet}
LOG_ROOT=${GRPO_LOG_DIR:-/workspace/meta-RL/logs/grpo_perf}; mkdir -p $LOG_ROOT
M=$(basename $MODEL_PATH); EXP=${EXPERIMENT_NAME:-grpo_${M}_tp${TP}_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=$LOG_ROOT/$EXP; mkdir -p $RUN_DIR
STEPS=${STEPS:-11}; MEM=${GPU_MEM_UTIL:-0.6}; MAXTOK=${PPO_MAX_TOK:-32768}
RE=ray_kwargs.ray_init.runtime_env
"$DISAGG_PYTHON" -m verl.trainer.main_ppo \
  data.train_files=$DATA data.val_files=$DATA data.train_batch_size=256 data.shuffle=false \
  data.max_prompt_length=4096 data.max_response_length=8192 data.filter_overlong_prompts=true data.truncation=error data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=$MODEL_PATH actor_rollout_ref.model.use_remove_padding=true actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAXTOK \
  actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=2 actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.weight_decay=0.1 'actor_rollout_ref.actor.optim.betas=[0.9,0.99]' actor_rollout_ref.actor.optim.clip_grad=1.0 \
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
  $RE.py_executable=$DISAGG_PYTHON "+$RE.env_vars.PYTHONPATH=$VERL_REPO:$SCRIPT_DIR" "+$RE.env_vars.TENSORBOARD_DIR=$RUN_DIR/tensorboard" \
  "+$RE.env_vars.VLLM_NO_USAGE_STATS='1'" "+$RE.env_vars.DO_NOT_TRACK='1'"  \
  "$@" 2>&1 | tee $RUN_DIR/driver.log
echo "[grpo] run dir $RUN_DIR"
