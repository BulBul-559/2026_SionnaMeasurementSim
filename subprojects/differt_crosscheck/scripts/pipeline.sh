#!/usr/bin/env bash
# End-to-end: select links -> run DiffeRT -> compare against the Sionna 339 dataset.
#
# Usage:
#   ./scripts/pipeline.sh                    # full 3 scenes x 1 BS x 10 UEs
#   ./scripts/pipeline.sh --scene front3d_3682
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HERE}/.venv-differt/bin/python"

if [ ! -x "${PY}" ]; then
  echo "missing ${PY}; run ./scripts/bootstrap_env.sh first" >&2
  exit 1
fi

export PYTHONPATH="${HERE}"

"${PY}" "${HERE}/scripts/select_links.py" "$@"
"${PY}" "${HERE}/scripts/run_links.py" "$@"
"${PY}" "${HERE}/scripts/compare_with_sionna.py" "$@"
