
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Optional
import json
import numpy as np

def _symmetric_increment(W: Dict[int, Dict[int, float]], i: int, j: int, delta: float):
    if i == j: return
    row_i = W.setdefault(i, {}); row_j = W.setdefault(j, {})
    row_i[j] = row_i.get(j, 0.0) + delta
    row_j[i] = row_j.get(i, 0.0) + delta
    if abs(row_i[j]) < 1e-12:
        del row_i[j]
        if not row_i: W.pop(i, None)
        if j in row_j:
            del row_j[i]
            if not row_j: W.pop(j, None)

def _prune_row_topk(row: Dict[int, float], k: int) -> Dict[int, float]:
    if len(row) <= k: return row
    items = sorted(row.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return dict(items)

@dataclass
class CovHebbLearner:
    n: int
    eta: float = 1e-3
    beta: float = 1e-2
    gamma: float = 1e-4
    mu: np.ndarray = field(init=False)
    W: Dict[int, Dict[int, float]] = field(default_factory=dict)

    def __post_init__(self):
        self.mu = np.zeros(self.n, dtype=float)

    def update_from_window(self, active: Iterable[int], values: Optional[Iterable[float]] = None):
        active = list(active)
        if values is None:
            values = [1.0] * len(active)
        else:
            values = list(values)
        for i, xi in zip(active, values):
            self.mu[i] = (1.0 - self.beta) * self.mu[i] + self.beta * float(xi)
        y = [(i, float(xi) - self.mu[i]) for i, xi in zip(active, values)]
        for idx_a, (a, ya) in enumerate(y):
            for b, yb in y[idx_a+1:]:
                # apply forgetting on touched edge
                if a in self.W and b in self.W[a]:
                    self.W[a][b] *= (1.0 - self.gamma)
                if b in self.W and a in self.W[b]:
                    self.W[b][a] *= (1.0 - self.gamma)
                _symmetric_increment(self.W, a, b, self.eta * (ya * yb))

    def prune_topk(self, k: int = 256):
        newW = {i: _prune_row_topk(row, k) for i, row in self.W.items()}
        symW: Dict[int, Dict[int, float]] = {}
        for i, row in newW.items():
            for j, v in row.items():
                if i == j: continue
                vi = v; vj = newW.get(j, {}).get(i, v)
                vavg = 0.5 * (vi + vj)
                symW.setdefault(i, {})[j] = vavg
                symW.setdefault(j, {})[i] = vavg
        self.W = symW

    def to_dense_block(self, nodes):
        import numpy as np
        m = len(nodes); idx = {node: k for k, node in enumerate(nodes)}
        M = np.zeros((m, m), dtype=float)
        for i in nodes:
            for j, v in self.W.get(i, {}).items():
                if j in idx:
                    a = idx[i]; b = idx[j]
                    M[a, b] = v; M[b, a] = v
        np.fill_diagonal(M, 0.0)
        return M

    def eigenspectrum(self, nodes, k: int = 10):
        M = self.to_dense_block(nodes)
        w, _ = np.linalg.eigh(M)
        w = w[::-1]
        return w[:k], w

    def save(self, path: str):
        data = {
            "n": self.n,
            "eta": self.eta,
            "beta": self.beta,
            "gamma": self.gamma,
            "mu": self.mu.tolist(),
            "W": {str(i): {str(j): v for j, v in row.items()} for i, row in self.W.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def load(path: str) -> "CovHebbLearner":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = CovHebbLearner(n=data["n"], eta=data["eta"], beta=data["beta"], gamma=data["gamma"])
        model.mu = np.array(data["mu"], dtype=float)
        W = {}
        for si, row in data["W"].items():
            i = int(si); W[i] = {int(sj): float(v) for sj, v in row.items()}
        model.W = W
        return model
