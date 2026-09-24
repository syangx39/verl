#!/usr/bin/env bash
# Build the derived image ON AN ARM64 MACHINE (a GB200 node, an Arm VM, or Cloud Build with an arm64 worker pool);
# QEMU cross-builds from x86 work but compiling flash-attn under emulation takes hours.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"          # the recipe dir: Dockerfile, verify_venv_lock.py and this script live side by side
REG=${REG:-us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr}
BASE_PIN=9924801779415f86c807b5716a3d4479fa60f811
UPSTREAM=${UPSTREAM:-https://github.com/jialei777/verl-upstream.git}

# ---- 1. local source: upstream pin + patches/*.patch, committed with a fixed identity/date -> deterministic SHA
if [ ! -d verl-src/.git ]; then
  git clone "$UPSTREAM" verl-src
fi
git -C verl-src fetch origin "$BASE_PIN" && git -C verl-src checkout --detach -q "$BASE_PIN"
for P in patches/*.patch; do git -C verl-src apply --check "../$P" && git -C verl-src apply "../$P"; done
git -C verl-src add -A
GIT_AUTHOR_NAME="verl-disagg recipe" GIT_AUTHOR_EMAIL="recipe@local" GIT_COMMITTER_NAME="verl-disagg recipe" GIT_COMMITTER_EMAIL="recipe@local" \
GIT_AUTHOR_DATE="2026-09-24T00:00:00Z" GIT_COMMITTER_DATE="2026-09-24T00:00:00Z" \
  git -C verl-src commit -q -m "GPU disagg recipe: apply patches/ on top of ${BASE_PIN}"
PIN=$(git -C verl-src rev-parse HEAD); echo "PIN=$PIN (base $BASE_PIN + $(ls patches/*.patch | wc -l) patch(es))"
echo "$PIN" > VERL_PIN.txt
# keep launcher and prepare_env on the same pin
sed -i "s|^PIN=.*|PIN=$PIN   # = $BASE_PIN + patches/ (see VERL_PIN.txt)|" run_gpu_disagg.sh
sed -i "s|^COMMIT = .*|COMMIT = \"$PIN\"|" prepare_env.py
grep -q "^PIN=$PIN" run_gpu_disagg.sh && grep -q "^COMMIT = \"$PIN\"" prepare_env.py || { echo "failed to sync PIN into run_gpu_disagg.sh / prepare_env.py"; exit 2; }
printf 'build.log\nbuild_driver.log\nlogs/\n__pycache__/\n' > .dockerignore

# ---- 2. build + push
TAG=$REG/verl-disagg-cu130-arm64:${PIN:0:9}
docker buildx build --platform linux/arm64 --progress=plain \
  --build-arg VERL_PIN="$PIN" --build-arg FLASH_ATTN_FROM_SOURCE=${FLASH_ATTN_FROM_SOURCE:-0} \
  -f Dockerfile -t "$TAG" --push . 2>&1 | tee build.log
DIGEST=$(docker buildx imagetools inspect "$TAG" --format '{{json .Manifest.Digest}}' | tr -d '"')
echo "IMAGE=$TAG"; echo "DIGEST=$DIGEST"
echo "$TAG@$DIGEST" > IMAGE_REF.txt      # put this exact reference into the RayCluster yaml (image: x2, DISAGG_IMAGE x2)