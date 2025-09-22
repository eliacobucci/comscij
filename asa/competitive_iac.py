#!/usr/bin/env python3
"""
True Interactive Activation & Competition with negative competitive connections
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Optional, Set
import json
import numpy as np
from collections import defaultdict

@dataclass
class CompetitiveIAC:
    """
    True IAC with competitive inhibition between mutually exclusive concepts.
    Creates both positive (cooperative) and negative (competitive) connections.
    """
    n: int
    eta_pos: float = 5e-3  # Learning rate for positive connections
    eta_neg: float = 2e-3  # Learning rate for negative connections  
    beta: float = 1e-2     # EMA rate for means
    gamma: float = 1e-4    # Forgetting factor
    competition_threshold: int = 2  # Min co-occurrences before creating competition
    
    mu: np.ndarray = field(init=False)
    W: Dict[int, Dict[int, float]] = field(default_factory=dict)
    co_occurrence_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    context_sets: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    
    def __post_init__(self):
        self.mu = np.zeros(self.n, dtype=float)
    
    def update_from_window(self, active: Iterable[int], values: Optional[Iterable[float]] = None):
        """Update with both positive and competitive learning."""
        active = list(set(active))  # Remove duplicates
        if len(active) < 2:
            return
            
        if values is None:
            values = [1.0] * len(active)
        else:
            values = list(values)
        
        # Update means
        for i, xi in zip(active, values):
            self.mu[i] = (1.0 - self.beta) * self.mu[i] + self.beta * float(xi)
        
        # Record co-occurrences and contexts for each concept
        for i in active:
            self.context_sets[i].update(active)
            
        # Update co-occurrence counts
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                if a != b:
                    key = tuple(sorted([a, b]))
                    self.co_occurrence_counts[key] = self.co_occurrence_counts.get(key, 0) + 1
        
        # POSITIVE LEARNING: Standard covariance for co-occurring concepts
        y = [(i, float(xi) - self.mu[i]) for i, xi in zip(active, values)]
        
        for idx_a, (a, ya) in enumerate(y):
            for b, yb in y[idx_a+1:]:
                if a != b:
                    # Apply forgetting
                    self._apply_forgetting(a, b)
                    # Positive connection for co-occurrence
                    self._symmetric_increment(a, b, self.eta_pos * (ya * yb))
        
        # COMPETITIVE LEARNING: Create negative connections between competing concepts
        self._update_competitive_connections(active)
    
    def _update_competitive_connections(self, current_active: List[int]):
        """Create competitive inhibition between concepts that don't co-occur."""
        
        # Find concepts that have sufficient co-occurrence data
        established_concepts = set()
        for (a, b), count in self.co_occurrence_counts.items():
            if count >= self.competition_threshold:
                established_concepts.add(a)
                established_concepts.add(b)
        
        if len(established_concepts) < 2:
            return
        
        # For each pair of established concepts, check if they compete
        established_list = list(established_concepts)
        for i in range(len(established_list)):
            for j in range(i + 1, len(established_list)):
                a, b = established_list[i], established_list[j]
                
                # Check if these concepts compete (don't co-occur much)
                key = tuple(sorted([a, b]))
                co_count = self.co_occurrence_counts.get(key, 0)
                
                # Get individual occurrence counts (approximation)
                a_contexts = len(self.context_sets[a])
                b_contexts = len(self.context_sets[b])
                
                # If they rarely co-occur but both are well-established, they compete
                expected_co_occurrence = (a_contexts * b_contexts) / (self.n * 10)  # Heuristic
                
                if (co_count < max(1, expected_co_occurrence * 0.3) and 
                    a_contexts >= self.competition_threshold and 
                    b_contexts >= self.competition_threshold):
                    
                    # Create competitive inhibition
                    competition_strength = -self.eta_neg * np.sqrt(a_contexts * b_contexts) / self.n
                    self._apply_forgetting(a, b)
                    self._symmetric_increment(a, b, competition_strength)
    
    def _symmetric_increment(self, i: int, j: int, delta: float):
        """Add symmetric weight update."""
        if i == j:
            return
            
        row_i = self.W.setdefault(i, {})
        row_j = self.W.setdefault(j, {})
        
        row_i[j] = row_i.get(j, 0.0) + delta
        row_j[i] = row_j.get(i, 0.0) + delta
        
        # Clean up very small weights
        if abs(row_i[j]) < 1e-12:
            row_i.pop(j, None)
            if not row_i:
                self.W.pop(i, None)
        if abs(row_j.get(i, 0)) < 1e-12:
            row_j.pop(i, None)
            if not row_j:
                self.W.pop(j, None)
    
    def _apply_forgetting(self, a: int, b: int):
        """Apply forgetting factor to existing connection."""
        if a in self.W and b in self.W[a]:
            self.W[a][b] *= (1.0 - self.gamma)
        if b in self.W and a in self.W[b]:
            self.W[b][a] *= (1.0 - self.gamma)
    
    def prune_topk(self, k: int = 256):
        """Keep only the strongest k connections per node."""
        for i, row in list(self.W.items()):
            if len(row) > k:
                # Sort by absolute strength and keep top k
                sorted_items = sorted(row.items(), key=lambda x: abs(x[1]), reverse=True)
                self.W[i] = dict(sorted_items[:k])
    
    def to_dense_block(self, nodes):
        """Convert to dense matrix for analysis."""
        m = len(nodes)
        idx = {node: k for k, node in enumerate(nodes)}
        M = np.zeros((m, m), dtype=float)
        
        for i in nodes:
            for j, v in self.W.get(i, {}).items():
                if j in idx:
                    a, b = idx[i], idx[j]
                    M[a, b] = v
        
        np.fill_diagonal(M, 0.0)
        return M
    
    def get_connection_stats(self):
        """Get statistics about positive and negative connections."""
        total_connections = sum(len(row) for row in self.W.values()) // 2  # Symmetric
        positive_connections = 0
        negative_connections = 0
        
        for i, row in self.W.items():
            for j, weight in row.items():
                if i < j:  # Count each pair once
                    if weight > 1e-8:
                        positive_connections += 1
                    elif weight < -1e-8:
                        negative_connections += 1
        
        return {
            'total': total_connections,
            'positive': positive_connections, 
            'negative': negative_connections,
            'neutral': total_connections - positive_connections - negative_connections
        }
    
    def save(self, path: str):
        """Save the model."""
        data = {
            "n": self.n,
            "eta_pos": self.eta_pos,
            "eta_neg": self.eta_neg, 
            "beta": self.beta,
            "gamma": self.gamma,
            "competition_threshold": self.competition_threshold,
            "mu": self.mu.tolist(),
            "W": {str(i): {str(j): v for j, v in row.items()} for i, row in self.W.items()},
            "co_occurrence_counts": {f"{k[0]},{k[1]}": v for k, v in self.co_occurrence_counts.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    
    @staticmethod
    def load(path: str) -> "CompetitiveIAC":
        """Load a saved model."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        model = CompetitiveIAC(
            n=data["n"], 
            eta_pos=data.get("eta_pos", 5e-3),
            eta_neg=data.get("eta_neg", 2e-3),
            beta=data["beta"], 
            gamma=data["gamma"],
            competition_threshold=data.get("competition_threshold", 2)
        )
        model.mu = np.array(data["mu"], dtype=float)
        
        # Restore W matrix
        W = {}
        for si, row in data["W"].items():
            i = int(si)
            W[i] = {int(sj): float(v) for sj, v in row.items()}
        model.W = W
        
        # Restore co-occurrence counts
        co_counts = {}
        for key_str, count in data.get("co_occurrence_counts", {}).items():
            a, b = map(int, key_str.split(","))
            co_counts[(a, b)] = count
        model.co_occurrence_counts = co_counts
        
        return model