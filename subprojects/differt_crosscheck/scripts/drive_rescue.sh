#!/usr/bin/env bash
# Parallel driver for the order-1 rescue pass: one scene per job, N workers.
#
#   drive_rescue.sh <run-root> <out-root> <manifest> [workers]
#
# Uses the Sionna-style scene layout: scenes/<scene>.h5 + status/<scene>.csv.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$1"; OUT_ROOT="$2"; MANIFEST="$3"; WORKERS="${4:-14}"

mkdir -p "${OUT_ROOT}/locks" "${OUT_ROOT}/logs" "${OUT_ROOT}/scenes" "${OUT_ROOT}/status"

SCENES_FILE="${OUT_ROOT}/scene_list.txt"
if [ ! -f "${SCENES_FILE}" ]; then
  python3 - "${RUN_ROOT}/compare/zero_links.csv" > "${SCENES_FILE}" <<'PY'
import csv, sys
seen = []
for row in csv.DictReader(open(sys.argv[1])):
    if row["scene_id"] not in seen:
        seen.append(row["scene_id"])
print("\n".join(seen))
PY
fi

worker() {
  local idx="$1"
  while read -r scene; do
    [ -z "${scene}" ] && continue
    [ -f "${OUT_ROOT}/scenes/${scene}.h5" ] && continue
    mkdir "${OUT_ROOT}/locks/${scene}" 2>/dev/null || continue
    echo "[w${idx}] start ${scene} $(date +%H:%M:%S)" >> "${OUT_ROOT}/logs/driver.log"
    if PYTHONPATH="${HERE}" JAX_PLATFORMS=cpu "${HERE}/.venv-differt/bin/python" -u \
        "${HERE}/scripts/rescue_zero_links.py" \
        --run-root "${RUN_ROOT}" --manifest "${MANIFEST}" --out-root "${OUT_ROOT}" \
        --scene "${scene}" --cell 0.05 --order 2 --types t \
        > "${OUT_ROOT}/logs/${scene}.log" 2>&1; then
      echo "[w${idx}] ok    ${scene} $(date +%H:%M:%S)" >> "${OUT_ROOT}/logs/driver.log"
    else
      echo "[w${idx}] FAIL  ${scene} $(date +%H:%M:%S)" >> "${OUT_ROOT}/logs/driver.log"
    fi
  done
}

i=0
for ((w = 0; w < WORKERS; w++)); do
  worker "${w}" < "${SCENES_FILE}" &
  i=$((i + 1))
done
wait
echo "all workers finished $(date +%H:%M:%S)" >> "${OUT_ROOT}/logs/driver.log"
