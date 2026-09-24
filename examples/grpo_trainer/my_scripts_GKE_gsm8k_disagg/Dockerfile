# Derived image for the disaggregated-async GSM8K baseline (build context = my_scripts_GKE_gsm8k_disagg/).
# Base: verlai/verl:uv-cu130-arm64 = Ubuntu 24.04, CUDA 13.0.2 devel, NCCL 2.28, gdrcopy, and a clean checkout of
# UPSTREAM verl main at VERL_PIN in /workspace/verl with its uv.lock and a prefetched uv cache -- but no venv.
# This image builds the venv from that lock (cache hits, minutes) and verifies it. No fork, no patches.
# Must be built on linux/arm64 (GB200 nodes are aarch64).
FROM verlai/verl@sha256:094943c1909590564feda8122bd090ba0a93aa42345245111a05225c1a7972f1

ARG VERL_PIN=f7e53133bfff899e2ba7690624e6e2f387d462a5
# set to 1 if the lock's flash-attn wheel URL is x86-only and 404s on aarch64: sync without it, then build it from source
ARG FLASH_ATTN_FROM_SOURCE=0

ENV UV_HTTP_TIMEOUT=500 UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/workspace/verl/.venv
WORKDIR /workspace/verl
# the checkout shipped in the base image must be exactly the pin and clean
RUN test "$(git rev-parse HEAD)" = "${VERL_PIN}" || { echo "base image checkout is $(git rev-parse HEAD), expected ${VERL_PIN}"; exit 1; } \
 && { test -z "$(git status --porcelain --untracked-files=no)" || { echo "base image checkout has tracked modifications:"; git status --porcelain --untracked-files=no; exit 1; }; } \
 && git rev-parse HEAD > /workspace/VERL_PIN
RUN if [ "${FLASH_ATTN_FROM_SOURCE}" = "1" ]; then \
      uv sync --frozen --all-packages --extra fsdp --extra vllm --no-install-package flash-attn \
      && FA=$(grep -A2 '^name = "flash-attn"' uv.lock | grep '^version' | head -1 | cut -d'"' -f2) \
      && MAX_JOBS=32 uv pip install --python .venv/bin/python --no-build-isolation "flash-attn==${FA}"; \
    else \
      uv sync --frozen --all-packages --extra fsdp --extra vllm; \
    fi
# prove the venv is complete and matches the pinned lock (cu130 torch entry, not the +cpu one) before the image is shipped
COPY verify_venv_lock.py /workspace/verify_venv_lock.py
RUN .venv/bin/python /workspace/verify_venv_lock.py /workspace/verl --require-async
ENV VIRTUAL_ENV=/workspace/verl/.venv \
    PATH=/workspace/verl/.venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/workspace/verl \
    VERL_PLATFORM=nvidia VLLM_USE_V1=1 PYTHONNOUSERSITE=1
LABEL verl.pin="${VERL_PIN}" verl.source="verl-project/verl main (checkout shipped in verlai/verl:uv-cu130-arm64)"
CMD ["bash"]
