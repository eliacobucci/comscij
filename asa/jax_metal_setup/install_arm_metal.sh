#!/usr/bin/env bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"${DIR}/preflight_arm64.sh"
echo "=== Installing pinned JAX + Metal (arm64 only) ==="
pip install -U pip wheel setuptools
if command -v conda >/dev/null 2>&1; then
  conda config --env --set subdir osx-arm64 || true
fi
pip install -r "${DIR}/requirements-metal.txt"
export JAX_PJRT_USE_PJRT_COMPATIBILITY=1
echo "=== Verifying JAX devices ==="
python - <<'PY'
import jax
print("JAX devices:", jax.devices())
assert any("Metal" in str(d) or "GPU" in str(d) for d in jax.devices()), "No Metal GPU device detected"
print("Verification OK: Metal GPU visible to JAX.")
PY
