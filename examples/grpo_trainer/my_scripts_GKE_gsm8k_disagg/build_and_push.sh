#!/usr/bin/env bash
# Build the derived image ON AN ARM64 MACHINE (a GB200 node, an Arm VM, or Cloud Build with an arm64 worker pool);
# QEMU cross-builds from x86 work but compiling flash-attn under emulation takes hours.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"          # the recipe dir: Dockerfile, verify_venv_lock.py and this script live side by side
REG=${REG:-us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr}
PIN=f7e53133bfff899e2ba7690624e6e2f387d462a5      # upstream verl main, the checkout shipped inside the base image
echo "$PIN" > VERL_PIN.txt
# keep launcher and prepare_env on the same pin
sed -i "s|^PIN=.*|PIN=$PIN   # upstream verl-project/verl main (checkout inside the image)|" run_gpu_disagg.sh
sed -i "s|^COMMIT = .*|COMMIT = \"$PIN\"|" prepare_env.py
grep -q "^PIN=$PIN" run_gpu_disagg.sh && grep -q "^COMMIT = \"$PIN\"" prepare_env.py || { echo "failed to sync PIN into run_gpu_disagg.sh / prepare_env.py"; exit 2; }
printf 'build.log\nbuild_driver.log\nlogs/\n__pycache__/\nverl-src/\n' > .dockerignore

TAG=$REG/verl-disagg-cu130-arm64:${PIN:0:9}
docker buildx build --platform linux/arm64 --progress=plain \
  --build-arg VERL_PIN="$PIN" --build-arg FLASH_ATTN_FROM_SOURCE=${FLASH_ATTN_FROM_SOURCE:-0} \
  -f Dockerfile -t "$TAG" --push . 2>&1 | tee build.log
DIGEST=$(docker buildx imagetools inspect "$TAG" --format '{{json .Manifest.Digest}}' | tr -d '"')
echo "IMAGE=$TAG"; echo "DIGEST=$DIGEST"
echo "$(echo "$TAG" | sed 's/:[^:]*$//')@$DIGEST" > IMAGE_REF.txt      # registry path @ digest: put into the RayCluster yaml (image: x2, DISAGG_IMAGE x2)