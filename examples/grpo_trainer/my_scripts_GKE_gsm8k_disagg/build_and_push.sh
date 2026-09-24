#!/usr/bin/env bash
# Build the derived image ON AN ARM64 MACHINE (a GB200 node, an Arm VM, or Cloud Build with an arm64 worker pool);
# QEMU cross-builds from x86 work but compiling flash-attn under emulation takes hours.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"          # the recipe dir: Dockerfile, verify_venv_lock.py and this script live side by side
REG=${REG:-us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr}
PIN=9924801779415f86c807b5716a3d4479fa60f811
TAG=$REG/verl-disagg-cu130-arm64:${PIN:0:9}
docker buildx build --platform linux/arm64 --progress=plain \
  --build-arg FLASH_ATTN_FROM_SOURCE=${FLASH_ATTN_FROM_SOURCE:-0} \
  -f Dockerfile -t "$TAG" --push . 2>&1 | tee build.log
DIGEST=$(docker buildx imagetools inspect "$TAG" --format '{{json .Manifest.Digest}}' | tr -d '"')
echo "IMAGE=$TAG"; echo "DIGEST=$DIGEST"
echo "$TAG@$DIGEST" > IMAGE_REF.txt      # put this exact reference into the RayCluster yaml (image: x2, DISAGG_IMAGE x2)