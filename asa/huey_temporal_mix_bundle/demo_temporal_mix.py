import numpy as np
from huey.temporal_mix import (update_T_freq_damped, center_matrix, prune_topk_abs,
                               row_norm_clamp, build_M, spectral_guardrail, topk_eigs)

np.random.seed(7)
n = 50
W = np.random.randn(n, n) * 0.05
W = 0.5*(W + W.T); np.fill_diagonal(W, 0.0)

T = np.zeros((n, n), dtype=float)
freq = np.random.randint(10, 500, size=n).astype(float)

for _ in range(5000):
    i = np.random.randint(0, n-1)
    j = (i + np.random.randint(1, 6)) % n
    dt = np.random.randint(1, 4)
    update_T_freq_damped(T, i, j, dt, freq, etaT=5e-3, tau=3.0)

T = center_matrix(T)
T = prune_topk_abs(T, k=16)
T = row_norm_clamp(T, rho=None)

alpha = 0.85
M = build_M(W, T, alpha=alpha)
alpha_new = spectral_guardrail(M, alpha, hi_ratio=4.0, lo_ratio=1.5, step=0.05)

print("alpha:", alpha, "->", alpha_new)
print("Top-5 eigenvalues:", np.array2string(topk_eigs(M, 5), precision=3))
