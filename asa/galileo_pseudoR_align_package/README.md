# Galileo Pseudo‑R Rotation Package

Align two concept spaces into a shared **inertial reference frame** using **pure rotations** (no scaling) in a pseudo‑Riemannian setup. The tool rotates **real** and **imaginary** blocks **separately**, fits on **matched labels only**, and can exclude **free concepts** from the least‑squares criterion while still transforming them.

## Contents
- `galileo_pseudoR_align.py` — alignment tool
- `templates/en_coords.csv` — sample English coordinates (`label,r1..r3,i1..i2`)
- `templates/zh_coords.csv` — sample Mandarin coordinates (one extra unmatched label)
- `templates/mapping.csv` — example explicit label mapping
- `templates/free_concepts.txt` — example list of free concepts to exclude from LS
- `README.md` — this file

## CSV schema
```
label, r1, r2, ..., rP, i1, i2, ..., iQ
```
- `r*` = coordinates in the real (positive) block
- `i*` = coordinates in the imaginary (negative) block
- You may omit the imaginary block if unused.

## Basic usage
From this folder:
```bash
python galileo_pseudoR_align.py --src templates/en_coords.csv --tgt templates/zh_coords.csv
```

## Explicit label mapping (optional)
If labels differ across files, provide a two‑column CSV: `src_label,tgt_label` (no header):
```bash
python galileo_pseudoR_align.py --src templates/en_coords.csv --tgt templates/zh_coords.csv --map templates/mapping.csv
```

## Free concepts (exclude from LS, still transform)
You can specify **free concepts** (manipulated concepts) to exclude from the fitting criterion but still carry through the rotation/translation:
```bash
# As a file with one label per line:
python galileo_pseudoR_align.py --src templates/en_coords.csv --tgt templates/zh_coords.csv --free templates/free_concepts.txt

# Or comma‑separated labels directly:
python galileo_pseudoR_align.py --src templates/en_coords.csv --tgt templates/zh_coords.csv --free "blue,house"
```

## Outputs
- `<src>_aligned.csv` — rotated+translated source coordinates
- `<src>_aligned_diffs.csv` — discrepancies for **matched, non‑free** concepts:
  - `euclidean_norm`
  - `J_signed_sq = ||Δ_real||^2 − ||Δ_imag||^2`
  - `J_abs_norm = sqrt(|J_signed_sq|)`

## Notes
- Only **orthogonal rotations** (det=+1); **no rescaling**.
- Real and imaginary blocks are rotated **independently**; never mixed.
- Translation aligns centroids computed from matched, non‑free points in each block.
- Unmatched and free concepts are **transformed**, but **do not influence** the least‑squares fit.
- This constructs a stable **inertial frame** for comparing relative movements.
