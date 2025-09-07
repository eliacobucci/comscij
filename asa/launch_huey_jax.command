#!/bin/bash
#
# 🚀 Launch Huey with JAX Acceleration
# 
# This script activates the native ARM64 environment and launches
# Huey GUI with JAX acceleration enabled.
#

echo "🚀 HUEY JAX LAUNCHER"
echo "===================="
echo ""
echo "🍎 Activating native ARM64 environment..."

# Activate the ARM64 conda environment
source ~/miniconda3/bin/activate huey_arm64

echo "✅ Environment activated"
echo "🚀 JAX version: $(python -c 'import jax; print(jax.__version__)')"
echo "🍎 Platform: $(python -c 'import platform; print(platform.machine())')"
echo ""
echo "🎨 Launching Huey GUI with JAX acceleration..."
echo ""

# Launch Huey GUI
python /Users/josephwoelfel/asa/launch_huey_gui.py

echo ""
echo "🔄 Huey GUI closed"