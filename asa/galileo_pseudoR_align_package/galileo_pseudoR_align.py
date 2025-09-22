#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

def parse_blocks(df):
    # Identify label column
    label_col = None
    for c in df.columns:
        if c.lower() in ("label","concept","word","id","token"):
            label_col = c
            break
    if label_col is None:
        raise SystemExit("No 'label' (or concept/word/id/token) column found.")
    # Partition columns into real/imag by prefix
    r_cols = [c for c in df.columns if c.lower().startswith("r")]
    i_cols = [c for c in df.columns if c.lower().startswith("i")]
    if not r_cols and not i_cols:
        raise SystemExit("No coordinate columns found. Use r1..rP for real, i1..iQ for imaginary.")
    R = df[r_cols].to_numpy(float) if r_cols else None
    I = df[i_cols].to_numpy(float) if i_cols else None
    labels = df[label_col].astype(str).tolist()
    return labels, R, I, label_col, r_cols, i_cols

def kabsch_orthogonal(P, Q):
    """Return orthogonal R (det=+1) solving min ||PR - Q||_F for centered P,Q."""
    C = P.T @ Q
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    R = Vt.T @ U.T
    # Enforce det +1 (proper rotation). If det<0, flip the last column of Vt.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R

def center(M):
    mu = M.mean(axis=0, keepdims=True)
    return M - mu, mu

def align_blocks(Rs, Rt, Is, It, match_idx_src, match_idx_tgt):
    """Fit independent rotations for R and I blocks using matched points only."""
    tR = tI = None
    Rrot = Irot = None
    mu_s_R = mu_t_R = mu_s_I = mu_t_I = 0.0

    if Rs is not None and Rt is not None:
        Pm = Rs[match_idx_src, :]
        Qm = Rt[match_idx_tgt, :]
        Pc, mu_s_R = center(Pm)
        Qc, mu_t_R = center(Qm)
        Rrot = kabsch_orthogonal(Pc, Qc)
        Rs = (Rs - mu_s_R) @ Rrot + mu_t_R
        tR = (mu_s_R, mu_t_R)
    elif Rs is None and Rt is None:
        Rrot = None
    else:
        raise SystemExit("Real block dimensionality mismatch: present in one CSV but not the other.")

    if Is is not None and It is not None:
        Pm = Is[match_idx_src, :]
        Qm = It[match_idx_tgt, :]
        Pc, mu_s_I = center(Pm)
        Qc, mu_t_I = center(Qm)
        Irot = kabsch_orthogonal(Pc, Qc)
        Is = (Is - mu_s_I) @ Irot + mu_t_I
        tI = (mu_s_I, mu_t_I)
    elif Is is None and It is None:
        Irot = None
    else:
        raise SystemExit("Imag block dimensionality mismatch: present in one CSV but not the other.")

    return Rs, Is, (Rrot, Irot), (tR, tI)

def pseudo_norms(Dr, Di):
    # Signed squared pseudo‑norm: ||v||_J^2 = ||v_r||^2  -  ||v_i||^2
    er2 = np.sum(Dr**2, axis=1) if Dr is not None else 0.0
    ei2 = np.sum(Di**2, axis=1) if Di is not None else 0.0
    signed = er2 - ei2
    eu = np.sqrt(er2 + ei2)  # Euclidean combined for reference
    absJ = np.sqrt(np.abs(signed))
    return signed, eu, absJ

def load_free_labels(path_or_list):
    if not path_or_list:
        return set()
    # Accept comma‑separated string or a file path
    p = Path(path_or_list)
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        tokens = [t.strip() for t in txt.replace("\n", ",").split(",") if t.strip()]
        return set(tokens)
    else:
        tokens = [t.strip() for t in path_or_list.split(",") if t.strip()]
        return set(tokens)

def main():
    ap = argparse.ArgumentParser(description="Galileo pseudo‑R rotation (real/imag blocks); matched-only LS; optional 'free concepts'.")
    ap.add_argument("--src", required=True, help="Source CSV: label,r1..rP[,i1..iQ]")
    ap.add_argument("--tgt", required=True, help="Target CSV: label,r1..rP[,i1..iQ]")
    ap.add_argument("--map", default=None, help="Optional mapping CSV: src_label,tgt_label (no header expected)")
    ap.add_argument("--free", default=None, help="Comma-separated labels or a file of labels to EXCLUDE from LS fit (still transformed)." )
    args = ap.parse_args()

    src_df = pd.read_csv(args.src)
    tgt_df = pd.read_csv(args.tgt)
    s_labels, Rs, Is, s_label_col, s_r_cols, s_i_cols = parse_blocks(src_df)
    t_labels, Rt, It, t_label_col, t_r_cols, t_i_cols = parse_blocks(tgt_df)

    free = load_free_labels(args.free)

    # Build label mapping (intersection or explicit map)
    if args.map:
        map_df = pd.read_csv(args.map, header=None)
        pairs = [(str(a), str(b)) for a,b in zip(map_df.iloc[:,0], map_df.iloc[:,1])]
        s_index = {l:i for i,l in enumerate(s_labels)}
        t_index = {l:i for i,l in enumerate(t_labels)}
        matched_pairs = [(s_index[a], t_index[b]) for a,b in pairs if a in s_index and b in t_index]
        # Apply 'free' exclusion on source labels
        matched_pairs = [(i,j) for (i,j) in matched_pairs if s_labels[i] not in free and t_labels[j] not in free]
        if len(matched_pairs) < 3:
            raise SystemExit("Fewer than 3 matched (non‑free) items; need >= 3 for a stable rotation.")
        s_idx = np.array([i for i,_ in matched_pairs], dtype=int)
        t_idx = np.array([j for _,j in matched_pairs], dtype=int)
        common_labels = [s_labels[i] for i,_ in matched_pairs]
    else:
        common = sorted(set(s_labels) & set(t_labels))
        # Drop 'free' from common
        common = [k for k in common if k not in free]
        if len(common) < 3:
            raise SystemExit("Fewer than 3 overlapping (non‑free) labels; need >= 3.")
        s_idx = np.array([s_labels.index(k) for k in common], dtype=int)
        t_idx = np.array([t_labels.index(k) for k in common], dtype=int)
        common_labels = common

    # Align (rotations on matched points only; apply to all points)
    Rs_aligned, Is_aligned, (Rrot, Irot), translations = align_blocks(Rs, Rt, Is, It, s_idx, t_idx)

    # Build full aligned dataframe with original column names
    out_df = pd.DataFrame({ "label": s_labels })
    if Rs_aligned is not None:
        out_df = pd.concat([out_df, pd.DataFrame(Rs_aligned, columns=s_r_cols)], axis=1)
    if Is_aligned is not None:
        out_df = pd.concat([out_df, pd.DataFrame(Is_aligned, columns=s_i_cols)], axis=1)

    # Compute diffs for matched points only (NOTE: diffs reported only for non‑free matched points)
    diffs = { "label": common_labels }
    if Rs_aligned is not None:
        Dr = Rs_aligned[s_idx, :] - Rt[t_idx, :]
    else:
        Dr = None
    if Is_aligned is not None:
        Di = Is_aligned[s_idx, :] - It[t_idx, :]
    else:
        Di = None

    signed_J, eu_norm, absJ = pseudo_norms(Dr, Di)
    diff_df = pd.DataFrame({
        "label": common_labels,
        "euclidean_norm": eu_norm,
        "J_signed_sq": signed_J,
        "J_abs_norm": absJ
    })

    # Summaries
    rms_eu = float(np.sqrt(np.mean(eu_norm**2)))
    mean_signed = float(np.mean(signed_J))
    mean_absJ   = float(np.mean(absJ))
    print(f"Matched (non‑free) points used for LS: {len(common_labels)}")
    if free:
        print(f"Free concepts excluded from LS (still transformed): {sorted(list(free))}")
    print(f"Euclidean RMS displacement: {rms_eu:.6g}")
    print(f"Mean signed J‑squared: {mean_signed:.6g}")
    print(f"Mean |J|-norm: {mean_absJ:.6g}")
    if Rrot is not None: print(f"Real rotation shape: {Rrot.shape}")
    if Irot is not None: print(f"Imag rotation shape: {Irot.shape}")
    if translations[0] is not None or translations[1] is not None:
        print("Translations applied from matched centroids (real/imag) to target centroids.")

    # Save outputs
    base = Path(args.src).with_suffix("")
    aligned_path = base.with_name(base.name + "_aligned.csv")
    diffs_path   = base.with_name(base.name + "_aligned_diffs.csv")
    out_df.to_csv(aligned_path, index=False)
    diff_df.to_csv(diffs_path, index=False)
    print(f"Saved aligned source to: {aligned_path}")
    print(f"Saved matched diffs to:  {diffs_path}")

if __name__ == "__main__":
    main()
