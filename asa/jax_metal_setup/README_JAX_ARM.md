# JAX on Apple Silicon (Metal only) — No CPU Fallback

This setup forces an Apple Silicon arm64 environment with JAX + jax‑metal.
It refuses to proceed on x86/Rosetta and does not suggest CPU-only fallbacks.

## 0) Start in a native arm64 shell
- Terminal.app: ensure “Open using Rosetta” is unchecked.
- `uname -m` should output `arm64`.
- (Recommended) Use a fresh env: `conda create -n huey-arm python=3.11 -y && conda activate huey-arm && conda config --env --set subdir osx-arm64`

## 1) Preflight (must pass)
```bash
bash jax_metal_setup/preflight_arm64.sh
```

## 2) Install pinned JAX + Metal
```bash
bash jax_metal_setup/install_arm_metal.sh
```
This sets `JAX_PJRT_USE_PJRT_COMPATIBILITY=1` and verifies that a Metal GPU appears in `jax.devices()`.

## Pins
- jax==0.4.30
- jaxlib==0.4.30
- jax-metal==0.1.1
