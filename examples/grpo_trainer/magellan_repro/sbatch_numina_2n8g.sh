#!/usr/bin/env bash
#SBATCH --job-name=numina-trlparity-2n8g-pd8
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mem=0

# =============================================================================
# NuminaMath TRL-parity multi-node run: 2 nodes x 4 GPU = 8 GPUs
# (W=8 -> SPG=4 under the pd=8 / round-2 accounting: 4 optimizer
# updates per rollout, mini_batch=64).
# Default 40 steps. TP control: sbatch --export=ALL,ROLLOUT_TP=2 --job-name=...-tp2
#
# Comparison anchors:
#   metaface W=8 pd=8 TRUE CYCLE = 121.2 s (gen 47.5 / update 20.1 / framework 52.5)
#   our archived pd=2 number       = 131.1 s (do NOT mix accountings)
#
# Ray scaffolding: head on node[0] -> workers on node[1..1] -> driver
# attaches via RAY_ADDRESS; /tmp:/tmp mount REQUIRED (raylet socket shared
# between head and driver container instances).
#
# Check afterwards:
#   1. "[accounting] W=8 SPG=4 ppo_mini_batch_size=64" at launch
#   2. timing_s/step vs the two anchors above
#   3. response_length/mean ~3,900-4,000, clip ~0.25 (regime check)
# =============================================================================

CONTAINER=$HOME/meta-RL/containers/verl-vllm-arm64.sqsh
MOUNTS=$HOME/meta-RL:/workspace/meta-RL,/tmp:/tmp
REPRO_DIR=/workspace/meta-RL/data/magellan_repro

# Environment block shared by every container invocation (head/worker/driver).
# Single-quoted heredoc-style variable: expanded inside the container shell.
ENV_SETUP='
  export HOME=/workspace/meta-RL/.home
  export CACHE_ROOT=/workspace/meta-RL/.cache
  mkdir -p $HOME $CACHE_ROOT
  export XDG_CACHE_HOME=$CACHE_ROOT
  export FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer
  export HF_HOME=$CACHE_ROOT/huggingface
  export TRITON_CACHE_DIR=$CACHE_ROOT/triton
  export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor
  export HF_HUB_OFFLINE=1
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
  export PYTHONPATH=/workspace/meta-RL/verl:$PYTHONPATH
  export DATA_DIR=/workspace/meta-RL/data/numina_verl
  export REWARD_FN_PATH=REPRO_PLACEHOLDER/format_reward_verl.py
  export MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B
'
ENV_SETUP=${ENV_SETUP//REPRO_PLACEHOLDER/$REPRO_DIR}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_ip=$(srun -N1 -n1 -w "$head_node" --mem=1G hostname -I | awk '{print $1}')
port=6379
echo "Ray head: $head_node ($head_ip:$port); workers: ${nodes[*]:1}"

# ---- 1) Ray head (node 0), stays up via --block ---------------------------
srun -N1 -n1 -w "$head_node" --mem=200G \
     --container-image=$CONTAINER --container-mounts=$MOUNTS \
     --container-workdir=/workspace/meta-RL \
     bash -c "$ENV_SETUP
  ray start --head --node-ip-address=$head_ip --port=$port \
    --num-gpus=4 --dashboard-host=0.0.0.0 --block
" &
sleep 40

# ---- 2) Ray workers (nodes 1..N-1) ----------------------------------------
for node in "${nodes[@]:1}"; do
  srun -N1 -n1 -w "$node" --mem=200G \
       --container-image=$CONTAINER --container-mounts=$MOUNTS \
       --container-workdir=/workspace/meta-RL \
       bash -c "$ENV_SETUP
      ray start --address=$head_ip:$port --num-gpus=4 --block
  " &
done
sleep 40

# ---- 3) Training driver on head node, attach to the cluster ---------------
srun -N1 -n1 -w "$head_node" --overlap --mem=200G \
     --container-image=$CONTAINER --container-mounts=$MOUNTS \
     --container-workdir=/workspace/meta-RL \
     bash -c "$ENV_SETUP
  export RAY_ADDRESS=$head_ip:$port
  ray status
  ROLLOUT_TP=\${ROLLOUT_TP:-1} NNODES=2 TOTAL_STEPS=\${TOTAL_STEPS:-40} bash $REPRO_DIR/run_qwen3_0p6b_trlparity.sh \
    trainer.test_freq=-1 \
    trainer.val_before_train=False
"
EXIT=$?

# Driver done -> tear down the background ray sruns (head + workers) so the
# job exits cleanly. Kill only our child srun processes, not the whole job,
# so Slurm records COMPLETED rather than CANCELLED.
pkill -TERM -P $$ srun 2>/dev/null || true
sleep 5
exit $EXIT
