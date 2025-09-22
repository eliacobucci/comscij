
from huey.learning.cov_hebb import CovHebbLearner
import numpy as np

vocab = ["dog","cat","barks","meows","howls","me","my","myself","our","us"]
V = {w:i for i,w in enumerate(vocab)}
model = CovHebbLearner(n=len(vocab), eta=5e-3, beta=1e-2, gamma=1e-4)

windows = [
    ["dog","barks"], ["dog","howls"], ["cat","meows"], ["dog","barks"],
    ["me","myself"], ["my","myself"], ["our","us"], ["me","our"],
    ["cat","meows"], ["dog","barks"], ["cat","meows"], ["dog","barks"],
]

for t, words in enumerate(windows):
    idx = [V[w] for w in words]
    vals = [1.0] * len(idx)
    model.update_from_window(idx, vals)
    if (t+1) % 4 == 0:
        model.prune_topk(4)

nodes = [V["dog"], V["cat"], V["barks"], V["meows"], V["howls"]]
M = model.to_dense_block(nodes)
print("Block (dog/cat/barks/meows/howls):")
print(np.array_str(M, precision=3, suppress_small=True))

vals_top, _ = model.eigenspectrum(nodes, k=5)
print("Top eigenvalues:", np.array_str(vals_top, precision=3))

model.save("/mnt/data/cov_hebb_demo_model.json")
print("Saved model to /mnt/data/cov_hebb_demo_model.json")
