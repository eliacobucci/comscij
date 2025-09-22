# Huey Tools Bundle (Minimal + Safe)

This zip contains two small tools that complement Huey while keeping complexity low:

- **Pseudo‑R Inertial Alignment**: rotate real/imag blocks independently; no scaling; align on matched (non‑free) labels only.  
- **Covariance Hebbian Updater**: streaming, sparse, signed weights so inhibitory links emerge naturally (good for self-terms).

## Quick Smoke Test

```bash
# 1) Alignment on templates
python tools/galileo_pseudoR_align.py --src tools/templates/en_coords.csv --tgt tools/templates/zh_coords.csv

# 2) Covariance Hebb demo
python -m tools.huey.examples.cov_hebb_demo
```

## Use in Huey

- Keep your **measurement matrix W0** unchanged.  
- Build a signed working matrix using the covariance updater and prune periodically.  
- For cross-language comparisons, align spaces with the pseudo‑R tool, optionally excluding **free concepts** you manipulated.

If anything feels confusing, run the **Smoke Test** first — it validates the basics in under a minute.
