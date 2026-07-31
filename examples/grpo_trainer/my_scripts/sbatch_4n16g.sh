#!/usr/bin/env bash
#SBATCH --job-name=verl-parity-4n16g
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mem=0

# =============================================================================
# Multi-node parity run: 4 nodes x 4 GPU = 16 GPUs, frozen invariants, 20 steps.
#
# Topology: verl is single-controller — one Ray cluster spanning all nodes,
# one training driver on the head node. NOT one launcher per node.
#   step 1: ray head on node[0]           (background, --block)
#   step 2: ray worker on node[1..3]      (background, --block)
#   step 3: driver on node[0], --overlap, attaches via RAY_ADDRESS
# /tmp:/tmp mount is REQUIRED: head and driver are separate enroot container
# instances on the same node; the raylet Unix socket lives under /tmp and must
# be visible to both (same fix as the TPU JobSet's hostPath shared-tmp).
#
# Check afterwards (vs 1n4g at the same engine mode):
#   1. timing_s/gen: expect ~2x, NOT 4x — long-tail bound (eager runs
#      measured 700->340s; per-replica seq count drops but the longest
#      sequence's wall time doesn't)
#   2. update_actor ~2.8x (FSDP cross-node tax); update_weights fixed ~4s
#   3. curves (score/length distributions) must match 1n4g within noise —
#      same data order, so step-by-step comparison is exact
# =============================================================================

CONTAINER=$HOME/meta-RL/containers/verl-vllm-arm64.sqsh
MOUNTS=$HOME/meta-RL:/workspace/meta-RL,/tmp:/tmp
SCRIPTS=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts

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
  export DATA_DIR=/workspace/meta-RL/data/openmathinstruct2
  export REWARD_FN_PATH=SCRIPTS_PLACEHOLDER/maxtext_math_reward.py
  export MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B
'
ENV_SETUP=${ENV_SETUP//SCRIPTS_PLACEHOLDER/$SCRIPTS}

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
  pip install --user -q math-verify==0.9.0
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
    pip install --user -q math-verify==0.9.0
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
  NNODES=4 bash $SCRIPTS/run_qwen3_0p6b_maxtext_parity.sh \
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