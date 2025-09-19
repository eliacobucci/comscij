#!/usr/bin/env bash
set -e

echo "== OS GPU visibility =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found (OK on non-NVIDIA systems)."
fi

echo
echo "== Python environment =="
python3 - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import torch
    print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch not importable:", e)

try:
    import cupy as cp
    print("CuPy:", cp.__version__, "CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
except Exception as e:
    print("CuPy not importable:", e)

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print("TensorFlow:", tf.__version__, "GPUs:", gpus)
except Exception as e:
    print("TensorFlow not importable:", e)
PY
