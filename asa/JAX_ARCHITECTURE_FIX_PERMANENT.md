# JAX Architecture Fix - PERMANENT SOLUTION

## The Problem
**ALWAYS HAPPENS**: JAX fails with architecture mismatch on Apple Silicon
- Error: `cpu_feature_guard.so (mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64'))`
- Root cause: Python running under Rosetta emulation with ARM64 JAX libraries

## The Solution (TESTED AND WORKS)
Use the **force ARM64 approach** that we've successfully used before:

### Step 1: Run the ARM64 JAX installer
```bash
python install_native_arm64_jax.py
```

### Step 2: Launch Huey with forced ARM64 execution
```bash
arch -arm64 /usr/bin/python3 -m streamlit run huey_time_working.py --server.port 8505
```

### Step 3: Verify JAX Metal is working
```bash
arch -arm64 /usr/bin/python3 -c "import jax; print('Devices:', jax.devices())"
```

## Why This Works
- `arch -arm64` forces native ARM64 execution (no Rosetta)
- `/usr/bin/python3` is the system Python (not conda under emulation)
- JAX libraries match the native ARM64 architecture

## Files That Contain Working Solutions
1. `install_native_arm64_jax.py` - Main installer
2. `force_arm64_jax.py` - Alternative installer
3. `launch_huey_gpu_arm64.command` - Working launcher script

## NEVER DO THIS AGAIN
- Don't try to fix JAX in conda environment 
- Don't uninstall/reinstall JAX manually
- Don't modify JAX import code - the problem is architectural

## When This Happens
Run this exact command sequence:
```bash
python install_native_arm64_jax.py
arch -arm64 /usr/bin/python3 -m streamlit run huey_time_working.py --server.port 8505
```

## Speed Benefits
- Native ARM64: ~3-5x faster than Rosetta
- JAX Metal GPU: ~10-50x faster for matrix operations
- Total speedup: ~15-250x faster than conda/CPU

---
**BOOKMARK THIS FILE** - Next time JAX breaks, just run these commands!