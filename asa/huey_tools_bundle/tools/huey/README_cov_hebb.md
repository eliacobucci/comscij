
# Covariance Hebbian (signed) updater for Huey

Switch Huey from raw co-occurrence to **centered (covariance) Hebbian updates** so **negative links** emerge from anti-correlation (e.g., *dog–meows*). Streaming, sparse, and fast.

## How it works
- Maintain EMA means `mu[i]`.
- For each window with activations `x`: compute `y = x - mu` on the active indices.
- Update `W += eta * (y y^T)` (zero diagonal implicitly). Include a tiny forgetting factor `gamma` on touched edges.
- Periodically `prune_topk(K)` to keep only the largest |weights| per node.

## API
```python
from huey.learning.cov_hebb import CovHebbLearner

m = CovHebbLearner(n=V, eta=1e-3, beta=1e-2, gamma=1e-4)
m.update_from_window(active_indices, values=None)  # values default to 1.0
m.prune_topk(256)
M = m.to_dense_block(nodes)       # for diagnostics/eigs
w_top, w_all = m.eigenspectrum(nodes, k=10)
m.save("model.json"); m2 = CovHebbLearner.load("model.json")
```

## Demo
```
python -m huey.examples.cov_hebb_demo
```

## Notes
- Keep your pristine measurement matrix separate if needed; use this signed `W` for dynamics/eigenspectrum.
- For large runs, prune with `K in [128..512]` every few hundred windows.
- Works well for *self-concept* terms (“me/my/myself/our/us”) because they anti-correlate with specific role nouns, creating selective inhibition without manual rules.
