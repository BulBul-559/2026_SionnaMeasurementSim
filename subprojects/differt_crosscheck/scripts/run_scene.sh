#!/usr/bin/env bash
# Drive ONE scene through the two-stage pipeline (GPU trace -> CPU fields).
#
#   run_scene.sh <run-root> <selection-json>
#
# Candidate count grows with (triangles)^order, so a single port-chunk size cannot fit
# every scene into 24 GB of VRAM. The trace therefore walks a ladder: 4 ports -> 2 -> 1,
# and if even one port does not fit it falls back to a CPU trace (slow but unbounded
# memory). The field stage always runs on CPU.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$1"; shift
SEL="$1"; shift

SCENE="$(basename "${SEL}" .json)"
WORK="${ROOT}/work/${SCENE}"
LOGDIR="${ROOT}/logs"
mkdir -p "${WORK}" "${LOGDIR}"

export PYTHONPATH="${HERE}"
TRACE_LOG="${LOGDIR}/${SCENE}.trace.log"
: > "${TRACE_LOG}"

trace_ok=0
for chunk in 4 2 1; do
  echo "=== gpu trace rx-chunk=${chunk} ===" >> "${TRACE_LOG}"
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
     XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
     "${HERE}/.venv-differt-gpu/bin/python" -u "${HERE}/scripts/split_stage.py" \
       --stage trace --profile 2988 --output-root "${WORK}" --selection "${SEL}" \
       --tracer exhaustive --order "${ORDER:-2}" --transmission --rx-chunk "${chunk}" >> "${TRACE_LOG}" 2>&1; then
    trace_ok=1
    echo "trace ok with rx-chunk=${chunk}" >> "${TRACE_LOG}"
    break
  fi
  rm -rf "${WORK}/paths"
done

if [ "${trace_ok}" -eq 0 ]; then
  echo "=== cpu trace fallback ===" >> "${TRACE_LOG}"
  if ! JAX_PLATFORMS=cpu "${HERE}/.venv-differt/bin/python" -u "${HERE}/scripts/split_stage.py" \
        --stage trace --profile 2988 --output-root "${WORK}" --selection "${SEL}" \
        --tracer exhaustive --order "${ORDER:-2}" --transmission --rx-chunk 4 >> "${TRACE_LOG}" 2>&1; then
    echo "FAILED ${SCENE} trace" >> "${LOGDIR}/failed.txt"
    exit 1
  fi
fi

if ! JAX_PLATFORMS=cpu "${HERE}/.venv-differt/bin/python" -u "${HERE}/scripts/split_stage.py" \
      --stage fields --profile 2988 --output-root "${WORK}" --selection "${SEL}" \
      --targets-file "${WORK}/paths/targets.json" --freq-chunk 32 \
      > "${LOGDIR}/${SCENE}.fields.log" 2>&1; then
  echo "FAILED ${SCENE} fields" >> "${LOGDIR}/failed.txt"
  exit 1
fi

if [ -f "${WORK}/dataset/scenes/${SCENE}.h5" ]; then
  cp -f "${WORK}/paths/targets.json" "${LOGDIR}/${SCENE}.targets.json" 2>/dev/null || true
  rm -rf "${WORK}/paths"
  echo "DONE ${SCENE}" >> "${LOGDIR}/done.txt"
else
  echo "FAILED ${SCENE} no-h5" >> "${LOGDIR}/failed.txt"
  exit 1
fi
