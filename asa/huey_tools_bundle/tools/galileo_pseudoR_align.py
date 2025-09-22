#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_blocks(df):
    label_col = None
    for c in df.columns:
        if c.lower() in ("label","concept","word","id","token"):
            label_col = c; break
    if label_col is None:
        raise SystemExit("No 'label' column found (label/concept/word/id/token).")
    r_cols = [c for c in df.columns if c.lower().startswith("r")]
    i_cols = [c for c in df.columns if c.lower().startswith("i")]
    if not r_cols and not i_cols:
        raise SystemExit("No coordinate columns. Use r1..rP for real, i1..iQ for imaginary.")
    R = df[r_cols].to_numpy(float) if r_cols else None
    I = df[i_cols].to_numpy(float) if i_cols else None
    labels = df[label_col].astype(str).tolist()
    return labels, R, I, label_col, r_cols, i_cols

def kabsch_orthogonal(P, Q):
    C = P.T @ Q
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1.0
        R = Vt.T @ U.T
    return R

def center(M):
    mu = M.mean(axis=0, keepdims=True)
    return M - mu, mu

def align_blocks(Rs, Rt, Is, It, s_idx, t_idx):
    Rrot = Irot = None
    mu_s_R = mu_t_R = mu_s_I = mu_t_I = 0.0

    if Rs is not None and Rt is not None:
        Pm, Qm = Rs[s_idx,:], Rt[t_idx,:]
        Pc, mu_s_R = center(Pm); Qc, mu_t_R = center(Qm)
        Rrot = kabsch_orthogonal(Pc, Qc)
        Rs = (Rs - mu_s_R) @ Rrot + mu_t_R
    elif (Rs is None) ^ (Rt is None):
        raise SystemExit("Real block present in one CSV but not the other.")

    if Is is not None and It is not None:
        Pm, Qm = Is[s_idx,:], It[t_idx,:]
        Pc, mu_s_I = center(Pm); Qc, mu_t_I = center(Qm)
        Irot = kabsch_orthogonal(Pc, Qc)
        Is = (Is - mu_s_I) @ Irot + mu_t_I
    elif (Is is None) ^ (It is None):
        raise SystemExit("Imag block present in one CSV but not the other.")

    return Rs, Is, Rrot, Irot

def load_free_labels(arg):
    if not arg: return set()
    p = Path(arg)
    if p.exists():
        s = p.read_text(encoding="utf-8")
        tokens = [t.strip() for t in s.replace("\n", ",").split(",") if t.strip()]
        return set(tokens)
    else:
        return set([t.strip() for t in arg.split(",") if t.strip()])

def main():
    ap = argparse.ArgumentParser(description="Galileo pseudo‑R rotation (no scaling); matched-only LS; real/imag rotated separately; supports free concepts.")
    ap.add_argument("--src", required=True, help="Source CSV: label,r1..,i1..")
    ap.add_argument("--tgt", required=True, help="Target CSV: label,r1..,i1..")
    ap.add_argument("--map", default=None, help="Optional mapping file: src_label,tgt_label (no header)")
    ap.add_argument("--free", default=None, help="Comma list or file of labels to EXCLUDE from LS fit (still transformed).")
    args = ap.parse_args()

    src_df = pd.read_csv(args.src); tgt_df = pd.read_csv(args.tgt)
    s_labels, Rs, Is, s_labcol, s_rcols, s_icols = parse_blocks(src_df)
    t_labels, Rt, It, t_labcol, t_rcols, t_icols = parse_blocks(tgt_df)

    free = load_free_labels(args.free)

    if args.map:
        map_df = pd.read_csv(args.map, header=None)
        pairs = [(str(a), str(b)) for a,b in zip(map_df.iloc[:,0], map_df.iloc[:,1])]
        s_index = {l:i for i,l in enumerate(s_labels)}
        t_index = {l:i for i,l in enumerate(t_labels)}
        matched = [(s_index[a], t_index[b]) for a,b in pairs if a in s_index and b in t_index]
        matched = [(i,j) for (i,j) in matched if s_labels[i] not in free and t_labels[j] not in free]
        if len(matched) < 3:
            raise SystemExit("Need >=3 matched (non‑free) items for stable rotation.")
        s_idx = np.array([i for i,_ in matched], dtype=int)
        t_idx = np.array([j for _,j in matched], dtype=int)
        used_labels = [s_labels[i] for i,_ in matched]
    else:
        common = sorted(set(s_labels) & set(t_labels))
        common = [k for k in common if k not in free]
        if len(common) < 3:
            raise SystemExit("Need >=3 overlapping (non‑free) labels.")
        s_idx = np.array([s_labels.index(k) for k in common], dtype=int)
        t_idx = np.array([t_labels.index(k) for k in common], dtype=int)
        used_labels = common

    Rs_aligned, Is_aligned, Rrot, Irot = align_blocks(Rs, Rt, Is, It, s_idx, t_idx)

    out_df = pd.DataFrame({"label": s_labels})
    if Rs_aligned is not None: out_df = pd.concat([out_df, pd.DataFrame(Rs_aligned, columns=s_rcols)], axis=1)
    if Is_aligned is not None: out_df = pd.concat([out_df, pd.DataFrame(Is_aligned, columns=s_icols)], axis=1)

    # diffs on matched, non‑free only
    diffs = {"label": used_labels}
    if Rs_aligned is not None: Dr = Rs_aligned[s_idx,:] - Rt[t_idx,:]
    else: Dr = None
    if Is_aligned is not None: Di = Is_aligned[s_idx,:] - It[t_idx,:]
    else: Di = None

    if Dr is not None: er2 = (Dr**2).sum(axis=1)
    else: er2 = 0.0
    if Di is not None: ei2 = (Di**2).sum(axis=1)
    else: ei2 = 0.0
    eu = np.sqrt(er2 + ei2)
    J_signed_sq = er2 - ei2
    J_abs_norm = np.sqrt(np.abs(J_signed_sq))

    diffs_df = pd.DataFrame({"label": used_labels, "euclidean_norm": eu, "J_signed_sq": J_signed_sq, "J_abs_norm": J_abs_norm})

    base = Path(args.src).with_suffix("")
    out_df.to_csv(base.with_name(base.name + "_aligned.csv"), index=False)
    diffs_df.to_csv(base.with_name(base.name + "_aligned_diffs.csv"), index=False)

    print(f"Matched (non‑free) points used for LS: {len(used_labels)}")
    if free: print("Free concepts (excluded from LS, still transformed):", sorted(list(free)))
    print("Saved:", base.with_name(base.name + "_aligned.csv"))
    print("Saved:", base.with_name(base.name + "_aligned_diffs.csv"))

if __name__ == "__main__":
    main()
