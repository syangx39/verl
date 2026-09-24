# Derived image for the disaggregated-async GSM8K baseline (build context = my_scripts_GKE_gsm8k_disagg/, which holds this Dockerfile, verify_venv_lock.py and build_and_push.sh): verl fork pinned at 9924801, dependencies installed from
# THAT commit's uv.lock into /workspace/verl-disagg/.venv, using the base image's prefetched uv cache.
# Base: verlai/verl:uv-cu130-arm64 (Ubuntu 24.04, CUDA 13.0.2 devel, NCCL 2.28, gdrcopy; uv cache prefetched for
# upstream main's lock -- NOT a ready venv). Must be built on linux/arm64 (GB200 nodes are aarch64).
FROM verlai/verl@sha256:094943c1909590564feda8122bd090ba0a93aa42345245111a05225c1a7972f1

# Source: a LOCAL checkout in the build context (verl-src/ = jialei777/verl-upstream@9924801 + patches/*.patch, committed
# with a fixed author/date so the resulting SHA is reproducible; see build_and_push.sh). VERL_PIN is declared only
# right before the source layer: an ARG participates in the cache key of every later RUN, so declaring it here would
# re-run the dependency install whenever a patch changes the pin.
# set to 1 if the lock's flash-attn wheel URL is x86-only and 404s on aarch64: sync without it, then build it from source
ARG FLASH_ATTN_FROM_SOURCE=0

ENV UV_HTTP_TIMEOUT=500 UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/workspace/verl-disagg/.venv
WORKDIR /workspace/verl-disagg
# Layer 1: dependencies only, keyed on the lock/pyproject so source patches do not re-trigger the install.
COPY verl-src/pyproject.toml verl-src/uv.lock ./
RUN if [ "${FLASH_ATTN_FROM_SOURCE}" = "1" ]; then \
      uv sync --frozen --no-install-project --all-packages --extra fsdp --extra vllm --no-install-package flash-attn \
      && FA=$(grep -A2 '^name = "flash-attn"' uv.lock | grep '^version' | head -1 | cut -d'"' -f2) \
      && MAX_JOBS=32 uv pip install --python .venv/bin/python --no-build-isolation "flash-attn==${FA}"; \
    else \
      uv sync --frozen --no-install-project --all-packages --extra fsdp --extra vllm; \
    fi
# Layer 2: the pinned + patched source; installs the project itself (editable) on top of the cached dependencies.
COPY verl-src /workspace/verl-disagg
ARG VERL_PIN
RUN test -n "${VERL_PIN}" && test "$(git rev-parse HEAD)" = "${VERL_PIN}" \
 && test -z "$(git status --porcelain --untracked-files=no)" && git rev-parse HEAD > /workspace/VERL_PIN \
 && uv sync --frozen --all-packages --extra fsdp --extra vllm $( [ "${FLASH_ATTN_FROM_SOURCE}" = "1" ] && echo --no-install-package flash-attn )
# prove the venv is complete and matches the pinned lock (cu130 torch entry, not the +cpu one) before the image is shipped
COPY verify_venv_lock.py /workspace/verify_venv_lock.py
RUN .venv/bin/python /workspace/verify_venv_lock.py /workspace/verl-disagg --require-async
ENV VIRTUAL_ENV=/workspace/verl-disagg/.venv \
    PATH=/workspace/verl-disagg/.venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/workspace/verl-disagg \
    VERL_PLATFORM=nvidia VLLM_USE_V1=1 PYTHONNOUSERSITE=1
LABEL verl.pin="${VERL_PIN}" verl.base="jialei777/verl-upstream@9924801779415f86c807b5716a3d4479fa60f811 + patches/"
CMD ["bash"]