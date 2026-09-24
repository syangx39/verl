# Derived image for the disaggregated-async GSM8K baseline (build context = my_scripts_GKE_gsm8k_disagg/).
# Base: verlai/verl:uv-cu130-arm64 (Ubuntu 24.04, CUDA 13.0.2 devel, NCCL 2.28, gdrcopy, prefetched uv cache; its own
# /workspace/verl checkout is upstream main @ f7e53133, too old: separate_async there always co-locates rollout on the
# trainer GPUs). This image clones UPSTREAM verl at VERL_PIN (hybrid_engine=False support, #7373 memory budget fix)
# into /workspace/verl-pin and builds its venv from that commit's uv.lock. No fork, no patches.
# Must be built on linux/arm64 (GB200 nodes are aarch64).
FROM verlai/verl@sha256:094943c1909590564feda8122bd090ba0a93aa42345245111a05225c1a7972f1

ARG VERL_PIN=ace775e87d8765bcdd114aac734ab71da5367a0f
# set to 1 if the lock's flash-attn wheel URL is x86-only and 404s on aarch64: sync without it, then build it from source
ARG FLASH_ATTN_FROM_SOURCE=0

ENV UV_HTTP_TIMEOUT=500 UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/workspace/verl-pin/.venv
WORKDIR /workspace
RUN git clone --filter=blob:none https://github.com/verl-project/verl.git verl-pin \
 && cd verl-pin && git checkout --detach ${VERL_PIN} && test -z "$(git status --porcelain --untracked-files=no)" \
 && git rev-parse HEAD > /workspace/VERL_PIN
WORKDIR /workspace/verl-pin
RUN if [ "${FLASH_ATTN_FROM_SOURCE}" = "1" ]; then \
      uv sync --frozen --all-packages --extra fsdp --extra vllm --no-install-package flash-attn \
      && FA=$(grep -A2 '^name = "flash-attn"' uv.lock | grep '^version' | head -1 | cut -d'"' -f2) \
      && MAX_JOBS=32 uv pip install --python .venv/bin/python --no-build-isolation "flash-attn==${FA}"; \
    else \
      uv sync --frozen --all-packages --extra fsdp --extra vllm; \
    fi
# prove the venv matches the pinned lock for the enabled extras (markers pick +cu130 torch, the right transformers) and
# that the CUDA extensions load, before the image is shipped
COPY verify_venv_lock.py /workspace/verify_venv_lock.py
RUN .venv/bin/python /workspace/verify_venv_lock.py /workspace/verl-pin --require-async
ENV VIRTUAL_ENV=/workspace/verl-pin/.venv \
    PATH=/workspace/verl-pin/.venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/workspace/verl-pin \
    VERL_PLATFORM=nvidia VLLM_USE_V1=1 PYTHONNOUSERSITE=1
LABEL verl.pin="${VERL_PIN}" verl.source="verl-project/verl main"
CMD ["bash"]
