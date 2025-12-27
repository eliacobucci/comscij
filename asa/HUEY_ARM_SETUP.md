# Huey ARM Setup - Apple Silicon Compatible

## Quick Start

**To run Huey (recommended method):**
```bash
./start_huey_arm.sh
```

Or manually:
```bash
source huey_arm_env/bin/activate
streamlit run huey_time_working.py
```

Huey will be available at: http://localhost:8502

## What Was Fixed

### The Problem
- JAX 0.8.2 has critical bugs with Apple Silicon Metal backend
- Running x86 Python via Rosetta causes AVX instruction errors
- This broke eigendecomposition and 3D visualizations

### The Solution
1. **ARM Python**: Installed native ARM Python 3.13 via Homebrew
2. **Stable JAX**: Pinned JAX to version 0.4.34 (last stable Metal release)
3. **Virtual Environment**: Created `huey_arm_env` with correct dependencies

## Important: DO NOT UPGRADE JAX

The `requirements_huey_arm.txt` file pins JAX to **0.4.34**.

**❌ Never run:**
- `pip install --upgrade jax`
- `pip install jax` (without version)

**✅ To reinstall if needed:**
```bash
source huey_arm_env/bin/activate
pip install -r requirements_huey_arm.txt
```

## Environment Details

- **Python**: `/opt/homebrew/bin/python3.13` (ARM64)
- **Virtual Env**: `huey_arm_env/`
- **JAX Version**: 0.4.34 (PINNED - do not upgrade)
- **GPU**: JAX Metal (Apple M4)

## Troubleshooting

### If you see AVX errors again:
```bash
# Check JAX version
source huey_arm_env/bin/activate
python -c "import jax; print(jax.__version__)"

# If not 0.4.34, reinstall:
pip install -r requirements_huey_arm.txt
```

### If environment is missing:
Just run `./start_huey_arm.sh` - it will recreate it automatically.

## Files Created

- `huey_arm_env/` - ARM Python virtual environment
- `requirements_huey_arm.txt` - Pinned dependencies
- `start_huey_arm.sh` - Automatic startup script
- `HUEY_ARM_SETUP.md` - This documentation

---
**Setup Date**: December 27, 2025
**JAX Metal Status**: Working ✅
