#!/usr/bin/env bash
set -euo pipefail
echo "=== Preflight: Apple Silicon arm64 environment check ==="
echo "uname -m: $(uname -m)"
PYTHON_BIN="$(which python || true)"
echo "python path: ${PYTHON_BIN}"
if [ -n "${PYTHON_BIN}" ]; then
  file "${PYTHON_BIN}" || true
  "${PYTHON_BIN}" - <<'PY'
import platform, sys
print("platform.machine:", platform.machine())
print("python_version:", sys.version.replace("\n"," "))
print("is_64bit:", sys.maxsize > 2**32)
PY
fi
if [ "$(uname -m)" != "arm64" ]; then
  echo "ERROR: Not an arm64 shell. Quit and reopen a non‑Rosetta Terminal."
  exit 1
fi
if command -v pip >/dev/null 2>&1; then
  echo "pip debug (looking for arm64 tags):"
  pip debug --verbose 2>/dev/null | grep -i -E 'arm64|aarch64' || echo "(no explicit arm64 tag shown)"
fi
echo "Preflight OK (arm64 detected)."
