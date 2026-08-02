#!/usr/bin/env bash
#SBATCH --job-name=numina-trlparity-32n128g
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mem=0
#SBATCH --nodelist=a4xlustref-a4xnodeset2-[0-7,9-13,15-17],a4xlustref-a4xnodeset3-[0-15]

# =============================================================================
# NuminaMath TRL-parity multi-node run: 32 nodes x 4 GPU = 128 GPUs
# (W=128 -> SPG=1, i.e. ONE optimizer update per step, mini_batch=256).
# Default 40 steps. TRL W=128 reference: rollout_gen_s=21.17s ONLY —
# their train/framework are NOT separable at SPG=1 (no gen-free steps),
# so per their own methodology ONLY the gen phase is apples-to-apples
# at this point. Our full breakdown remains valid (verl phases are
# timer-separated regardless of SPG).
#
# ---- CROSS-NVL-DOMAIN POINT — new territory, read this ----
# 32 nodes exceeds a single NVL72 domain (18 nodes x 4 GPU = 72 GPUs).
# This run necessarily spans >=2 NVLink domains. Implications:
#   * Rollout (DP replicas) is domain-agnostic: no cross-replica traffic.
#   * FSDP collectives (all-gather/reduce-scatter over 128 ranks) and the
#     refit broadcast now cross the domain boundary over the scale-out
#     fabric (RoCE), not NVLink. At 0.6B the volumes are small; expect a
#     modest bump in update_actor/update_weights, not a cliff — measure.
#   * BALANCE the split across domains (e.g. 16+16 from two nodesets):
#     asymmetric splits (e.g. 18+14) skew collective ring latency andmake
#     the numbers harder to attribute. Use the -w override below to pin.
#   * nvidia-imex is per-domain: verify the imex service is healthy on ALL
#     participating nodesets before the run (known failure mode on A4X).
#   * NCCL: no MNNVL-specific env should assume a single domain; defaults
#     handle mixed NVLink+RoCE, but capture NCCL_DEBUG=INFO on rank 0 for
#     the first run to archive the detected topology.
# Pin nodes explicitly for a balanced split (edit to your idle sets):
#   sbatch -w a4xlustref-a4xnodeset0-[0,2-16],a4xlustref-a4xnodeset3-[0-15] \
#          sbatch_numina_32n128g.sh
# The topology-echo step below logs the nodes-per-nodeset split — check it
# in the log before trusting the numbers.
#
# Ray scaffolding identical to the other multi-node scripts; worker
# registration sleep raised to 120s (31 workers).
#
# Check afterwards:
#   1. "[accounting] W=128 SPG=1 ppo_mini_batch_size=256" at launch
#   2. gen vs TRL's 21.17s (their only comparable number here) and vs our
#      W=64 gen=19.0s — per-replica seq count is 16; if gen stops falling,
#      that is the rollout parallelism saturation point (matches TRL's
#      own flattening 23.0 -> 21.2 at 64 -> 128)
#   3. update_actor/update_weights vs W=64 (6.9/2.1s) — the cross-domain
#      collective tax, if any, shows up here
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
# ---- topology echo: nodes per nodeset (check balance across NVL domains) --
echo "[topology] nodes-per-nodeset split:"
printf '%s\n' "${nodes[@]}" | sed 's/-[0-9]*$//' | sort | uniq -c

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
sleep 120

# ---- 3) Training driver on head node, attach to the cluster ---------------
srun -N1 -n1 -w "$head_node" --overlap --mem=200G \
     --container-image=$CONTAINER --container-mounts=$MOUNTS \
     --container-workdir=/workspace/meta-RL \
     bash -c "$ENV_SETUP
  export RAY_ADDRESS=$head_ip:$port
  ray status
  NNODES=32 TOTAL_STEPS=\${TOTAL_STEPS:-40} bash $REPRO_DIR/run_qwen3_0p6b_trlparity.sh \
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
