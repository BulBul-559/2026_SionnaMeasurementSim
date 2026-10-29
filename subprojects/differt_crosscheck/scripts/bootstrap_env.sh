#!/usr/bin/env bash
# Create the isolated environment for the DiffeRT cross-check.
#
# Deliberately separate from the Sionna environment (../../.venv): DiffeRT needs
# Python >= 3.11 + JAX + Warp, and mixing it into the Sionna env risks breaking the
# existing simulation stack.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HERE}/.venv-differt"
PY_VERSION="${PY_VERSION:-3.12}"
INDEX_URL="${INDEX_URL:-https://mirrors.bfsu.edu.cn/pypi/web/simple}"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; install it first (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

if [ ! -x "${VENV}/bin/python" ]; then
  uv venv --python "${PY_VERSION}" "${VENV}"
fi

uv pip install \
  --python "${VENV}/bin/python" \
  --index-url "${INDEX_URL}" \
  -r "${HERE}/requirements.txt"

"${VENV}/bin/python" - <<'PY'
import differt, jax
print(f"differt {differt.__version__}  jax {jax.__version__}  devices={jax.devices()}")
PY
