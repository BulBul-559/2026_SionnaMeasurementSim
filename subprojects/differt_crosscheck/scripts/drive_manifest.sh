#!/usr/bin/env bash
# Parallel driver: one worker per GPU card, scenes claimed atomically, resumable.
#
#   drive_manifest.sh <run-root> [workers]
#
# Each worker walks the scene list, claims a scene by creating locks/<scene>
# (mkdir is atomic), runs run_scene.sh on its own GPU, and moves on. Finished scenes
# leave work/<scene>/dataset/scenes/<scene>.h5 behind; a rerun skips them.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$1"
WORKERS="${2:-6}"

mkdir -p "${ROOT}/locks" "${ROOT}/logs" "${ROOT}/work"

# GPU cards to use: skip the ones with foreign load unless told otherwise.
if [ -z "${GPU_IDS:-}" ]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 2000 {printf "%s ", $1}')"
fi
read -r -a GPUS <<< "${GPU_IDS}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "no idle GPU; falling back to CPU-only workers" >&2
  GPUS=(0)
fi
echo "workers=${WORKERS} gpus=${GPUS[*]}"

worker() {
  local gpu="$1"
  while read -r scene; do
    [ -z "${scene}" ] && continue
    local h5="${ROOT}/work/${scene}/dataset/scenes/${scene}.h5"
    [ -f "${h5}" ] && continue
    if ! mkdir "${ROOT}/locks/${scene}" 2>/dev/null; then
      continue
    fi
    echo "[gpu${gpu}] start ${scene} $(date +%H:%M:%S)" >> "${ROOT}/logs/driver.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${HERE}/scripts/run_scene.sh" "${ROOT}" "${ROOT}/selections/${scene}.json" \
      && echo "[gpu${gpu}] ok    ${scene} $(date +%H:%M:%S)" >> "${ROOT}/logs/driver.log" \
      || echo "[gpu${gpu}] FAIL  ${scene} $(date +%H:%M:%S)" >> "${ROOT}/logs/driver.log"
  done
}

SCENES_FILE="${ROOT}/scene_ids.txt"
if [ ! -f "${SCENES_FILE}" ]; then
  python3 - "${ROOT}/scenes.json" > "${SCENES_FILE}" <<'PY'
import json, sys
print("\n".join(e["scene_id"] for e in json.load(open(sys.argv[1]))))
PY
fi

i=0
for ((w = 0; w < WORKERS; w++)); do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  i=$((i + 1))
  worker "${gpu}" < "${SCENES_FILE}" &
done
wait
echo "all workers finished $(date +%H:%M:%S)" >> "${ROOT}/logs/driver.log"
