# Huey Temporal Mix Package

Purpose: keep windows co-occurrence (W) clean while adding temporal order (T) safely.

Run:
    python demo_temporal_mix.py

Use from code:
    from huey.temporal_mix import (...)

See inline docstrings for details. Suggested defaults:
- alpha=0.85
- center/prune/clamp T every ~1000 windows
- TopK=256
- spectral_guardrail hi_ratio=4.0 lo_ratio=1.5 step=0.05
