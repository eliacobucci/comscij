#!/usr/bin/env python3
"""
HueyTime — a drop‑in, time‑dependent Hebbian learner for Huey

Purpose
-------
Replace Huey's windowed co‑occurrence updates with order‑sensitive, time‑dependent
Hebbian learning that yields a *directed*, asymmetric weight matrix W. Use W for
causality/lead‑lag analyses; derive a symmetric matrix S *only* for embedding/
plotting in Galileo‑style spaces.

API Overview
------------
- HueyTime(config: HueyTimeConfig)
    .update_doc(tokens, boundaries=None)
    .update_stream(token, boundary=False)
    .export_W(copy=True)
    .export_S(mode="avg")
    .row_normalize(in_place=True)
    .save(path)
    .load(path)

- HueyTimeConfig dataclass with sensible defaults. See below.

Two learning modes (choose one or mix in separate passes):
- method="lagged": sequence‑aware Hebbian using lagged outer products with exp decay
- method="context": working‑memory (decaying context vector) Hebbian

Notes
-----
• Keep W directed during learning. Do not overwrite it with any symmetrized form.
• Sentence/turn boundaries down‑weight (not remove) cross‑boundary updates.
• Optional small reverse (feedback) learning rate eta_fb captures re‑afference.
• Suitable for streaming transcripts: call update_stream per token.

"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple
import numpy as np
from math import exp
import json

# ---------------------------
# Configuration and Defaults
# ---------------------------

@dataclass
class HueyTimeConfig:
    # vocab
    vocab: Dict[str, int]
    V: Optional[int] = None

    # learning mode: 'lagged' or 'context'
    method: str = "lagged"

    # Lagged method params
    max_lag: int = 8
    tau: float = 3.0               # exp distance decay
    eta_fwd: float = 1e-2          # forward learning rate
    eta_fb: float = 2e-3           # reverse/feedback learning rate (usually smaller)

    # Context method params
    alpha: float = 0.9             # context decay (closer to 1 = longer memory)

    # Regularization / hygiene
    l2_decay: float = 1e-4         # multiplicative forgetting per doc/epoch
    boundary_penalty: float = 0.25 # multiplier across sentence/turn boundaries
    allow_self: bool = False       # whether to allow i->i updates

    # Numerical guards
    max_row_sum: Optional[float] = None  # clip any row sum above this (None = off)

    def validate(self) -> None:
        assert self.method in ("lagged", "context"), "method must be 'lagged' or 'context'"
        if self.V is None:
            self.V = len(self.vocab)
        assert self.V == len(self.vocab), "V must equal len(vocab)"
        assert self.max_lag >= 1, "max_lag must be >= 1"
        assert 0.0 < self.alpha <= 1.0, "alpha in (0,1]"
        assert 0.0 <= self.boundary_penalty <= 1.0, "boundary_penalty in [0,1]"

# ---------------------------
# Core Learner
# ---------------------------

class HueyTime:
    def __init__(self, config: HueyTimeConfig):
        self.cfg = config
        self.cfg.validate()
        V = self.cfg.V
        self.W = np.zeros((V, V), dtype=np.float64)  # directed, asymmetric
        # context vector for streaming in context mode
        self._c = np.zeros(V, dtype=np.float64)
        # token frequency (optional TF‑like damping if desired)
        self.freq = np.zeros(V, dtype=np.int64)

    # ---------- Utilities ----------
    @staticmethod
    def _indices(tokens: Iterable[str], vocab: Dict[str, int]) -> List[int]:
        return [vocab[t] for t in tokens if t in vocab]

    @staticmethod
    def _penalty(boundaries: Optional[List[int]], t: int, u: int, boundary_penalty: float) -> float:
        if not boundaries:
            return 1.0
        for b in boundaries:
            if t < b <= u:
                return boundary_penalty
        return 1.0

    def _apply_row_clips(self) -> None:
        if self.cfg.max_row_sum is None:
            return
        rs = self.W.sum(axis=1)
        big = rs > self.cfg.max_row_sum
        if np.any(big):
            scale = (self.cfg.max_row_sum / np.maximum(rs[big], 1e-12))
            self.W[big, :] *= scale[:, None]

    # ---------- Public API ----------
    def update_doc(self, tokens: List[str], boundaries: Optional[List[int]] = None) -> None:
        """
        tokens: list of token strings for one document/turn.
        boundaries: optional sorted positions (0‑based, AFTER which a boundary occurs),
                    e.g., sentence ends or speaker changes.
        """
        idx = self._indices(tokens, self.cfg.vocab)
        for i in idx:
            self.freq[i] += 1

        if self.cfg.method == "lagged":
            n = len(idx)
            for t in range(n):
                i = idx[t]
                for lag in range(1, self.cfg.max_lag + 1):
                    u = t + lag
                    if u >= n:
                        break
                    j = idx[u]
                    if (not self.cfg.allow_self) and (i == j):
                        continue
                    k = exp(-lag / self.cfg.tau)
                    k *= self._penalty(boundaries, t, u, self.cfg.boundary_penalty)
                    self.W[i, j] += self.cfg.eta_fwd * k
                    if self.cfg.eta_fb > 0.0:
                        self.W[j, i] += self.cfg.eta_fb * k
        else:  # context method
            V = self.cfg.V
            c = np.zeros(V, dtype=np.float64)
            for pos, j in enumerate(idx):
                if c.any():
                    # strengthen i->j for all context i
                    self.W[:, j] += self.cfg.eta_fwd * c
                    if self.cfg.eta_fb > 0.0:
                        self.W[j, :] += self.cfg.eta_fb * c
                # decay + insert current token
                c *= self.cfg.alpha
                c[j] += (1.0 - self.cfg.alpha)
            self._c = c  # persist last context if you want to continue across docs

        # multiplicative forgetting per document
        if self.cfg.l2_decay > 0.0:
            self.W *= (1.0 - self.cfg.l2_decay)
        self._apply_row_clips()

    def update_stream(self, token: str, boundary: bool = False) -> None:
        """Online update, one token at a time (recommended for live transcripts)."""
        if token not in self.cfg.vocab:
            return
        j = self.cfg.vocab[token]
        self.freq[j] += 1

        if self.cfg.method == "lagged":
            raise RuntimeError("update_stream currently supports method='context' best; \
use update_doc for 'lagged' or batch tokens between boundaries.")
        else:
            # context method online
            if self._c.any():
                self.W[:, j] += self.cfg.eta_fwd * self._c
                if self.cfg.eta_fb > 0.0:
                    self.W[j, :] += self.cfg.eta_fb * self._c
            # boundary down‑weighting handled by decaying context harder at boundaries
            if boundary:
                self._c *= (self.cfg.alpha * self.cfg.boundary_penalty)
            else:
                self._c *= self.cfg.alpha
            self._c[j] += (1.0 - self.cfg.alpha)
            if self.cfg.l2_decay > 0.0:
                self.W *= (1.0 - self.cfg.l2_decay)
            self._apply_row_clips()

    def export_W(self, copy: bool = True) -> np.ndarray:
        return self.W.copy() if copy else self.W

    def export_S(self, mode: str = "avg") -> np.ndarray:
        if mode == "avg":
            return 0.5 * (self.W + self.W.T)
        elif mode == "gram":
            return self.W @ self.W.T
        else:
            raise ValueError("mode must be 'avg' or 'gram'")

    def row_normalize(self, in_place: bool = True, eps: float = 1e-12) -> np.ndarray:
        rs = self.W.sum(axis=1, keepdims=True)
        R = np.divide(self.W, np.maximum(rs, eps), out=np.zeros_like(self.W), where=rs>eps)
        if in_place:
            self.W[:] = R
        return R

    def save(self, path: str) -> None:
        meta = asdict(self.cfg)
        np.savez_compressed(
            path,
            W=self.W,
            freq=self.freq,
            meta=json.dumps(meta).encode("utf-8"),
        )

    @classmethod
    def load(cls, path: str) -> "HueyTime":
        blob = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(blob["meta"]).decode("utf-8"))
        cfg = HueyTimeConfig(**meta)
        obj = cls(cfg)
        obj.W = blob["W"]
        obj.freq = blob["freq"]
        return obj

# ---------------------------
# Helper: quick vocab builder
# ---------------------------

def build_vocab(tokens: Iterable[str]) -> Dict[str, int]:
    uniq = sorted(set(tokens))
    return {t: i for i, t in enumerate(uniq)}

# ---------------------------
# Example usage / smoke test
# ---------------------------
if __name__ == "__main__":
    # Tiny toy corpus with boundaries (sentence ends at positions 3 and 7)
    tokens = ["not", "this", "but", "that", 
              "if", "x", "then", "y"]
    boundaries = [3, 7]  # AFTER indices 3 and 7

    vocab = build_vocab(tokens)
    cfg = HueyTimeConfig(vocab=vocab, method="lagged", max_lag=4, tau=2.5,
                         eta_fwd=1e-2, eta_fb=2e-3, boundary_penalty=0.25,
                         l2_decay=0.0, allow_self=False)
    ht = HueyTime(cfg)
    ht.update_doc(tokens, boundaries=boundaries)
    W = ht.export_W()
    S = ht.export_S("avg")
    print("Vocab:", vocab)
    print("Directed index (||W - W^T||1 / ||W||1):", np.sum(np.abs(W - W.T)) / max(np.sum(np.abs(W)), 1e-12))
    print("Top forward edges (i->j, weight):")
    nz = np.argwhere(W > 0)
    edges = [(list(vocab.keys())[i], list(vocab.keys())[j], W[i, j]) for i, j in nz]
    edges.sort(key=lambda x: -x[2])
    for e in edges[:10]:
        print(e)