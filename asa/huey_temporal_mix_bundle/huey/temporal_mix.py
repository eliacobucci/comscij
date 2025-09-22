from __future__ import annotations
import numpy as np

def update_T_freq_damped(T: np.ndarray, i: int, j: int, dt: float,
                         freq: np.ndarray, etaT: float = 1e-3, tau: float = 3.0,
                         stopmask: np.ndarray | None = None):
    if stopmask is not None and (stopmask[i] or stopmask[j]):
        return
    fi = max(freq[i], 1.0); fj = max(freq[j], 1.0)
    w = etaT * np.exp(-dt / max(tau, 1e-8)) / np.sqrt(fi * fj)
    T[i, j] += w

def center_matrix(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    C = J @ A @ J
    np.fill_diagonal(C, 0.0)
    return C

def prune_topk_abs(A: np.ndarray, k: int = 256) -> np.ndarray:
    n = A.shape[0]
    B = np.zeros_like(A)
    for i in range(n):
        row = A[i, :].copy()
        row[i] = 0.0
        if k < n:
            idx = np.argpartition(np.abs(row), -k)[-k:]
            B[i, idx] = row[idx]
        else:
            B[i, :] = row
    np.fill_diagonal(B, 0.0)
    return 0.5 * (B + B.T)

def row_norm_clamp(A: np.ndarray, rho: float | None = None) -> np.ndarray:
    rows = np.linalg.norm(A, axis=1) + 1e-12
    if rho is None:
        rho = np.percentile(rows, 95.0)
        rho = max(rho, 1e-12)
    scale = np.minimum(1.0, rho / rows)
    return (A.T * scale).T

def normalize_fro(A: np.ndarray) -> np.ndarray:
    s = np.linalg.norm(A, ord='fro')
    return A / max(s, 1e-12)

def normalize_median(A: np.ndarray) -> np.ndarray:
    m = np.median(np.abs(A[np.nonzero(A)])) if np.any(A) else 1.0
    return A / max(m, 1e-12)

def build_M(W: np.ndarray, T: np.ndarray, alpha: float = 0.85, norm: str = "fro") -> np.ndarray:
    Ts = 0.5 * (T + T.T)
    if norm == "fro":
        Wn = normalize_fro(W); Tn = normalize_fro(Ts)
    else:
        Wn = normalize_median(W); Tn = normalize_median(Ts)
    return alpha * Wn + (1.0 - alpha) * Tn

def topk_eigs(A: np.ndarray, k: int = 5):
    w, _ = np.linalg.eigh(0.5*(A + A.T))
    w = w[::-1]
    return w[:k]

def spectral_guardrail(M: np.ndarray, alpha: float, hi_ratio: float = 4.0, lo_ratio: float = 1.5, step: float = 0.05) -> float:
    evals = topk_eigs(M, k=2)
    if len(evals) < 2 or abs(evals[1]) < 1e-12:
        return alpha
    r = float(evals[0] / max(evals[1], 1e-12))
    if r > hi_ratio:
        alpha = min(0.98, alpha + step)
    elif r < lo_ratio:
        alpha = max(0.50, alpha - step)
    return alpha
