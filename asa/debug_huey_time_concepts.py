#!/usr/bin/env python3
"""
Debug script to test HueyTime concept extraction
"""
import re
from huey_time import HueyTime, HueyTimeConfig, build_vocab

# Test with simple content first
test_content = "The cat sat on the mat. The dog ran in the park. The bird flew through the sky."

print("🧪 DEBUGGING HUEY TIME CONCEPT EXTRACTION")
print("=" * 60)

# Tokenize
words = re.findall(r'\b\w+\b', test_content.lower())
print(f"📝 Words found: {len(words)}")
print(f"🔤 Words: {words[:20]}...")

# Build vocabulary  
vocab = build_vocab(words)
print(f"📚 Vocabulary size: {len(vocab)}")
print(f"🗂️ Vocab sample: {dict(list(vocab.items())[:10])}")

# Configure HueyTime
config = HueyTimeConfig(
    vocab=vocab,
    method="lagged",
    max_lag=8,
    tau=3.0,
    eta_fwd=1e-2,
    eta_fb=2e-3,
    boundary_penalty=0.25,
    l2_decay=1e-4,
    allow_self=False
)

# Initialize and process
huey_time = HueyTime(config)
huey_time.update_doc(words, boundaries=None)

# Get results
W_directed = huey_time.export_W()
S_symmetric = huey_time.export_S("avg")

# Count concepts and connections
concept_count = len(vocab)
connection_count = (W_directed > 0).sum()

print(f"\n📊 RESULTS:")
print(f"   Concept count: {concept_count}")
print(f"   Connection count: {connection_count}")
print(f"   Max weight: {W_directed.max():.6f}")
print(f"   Non-zero weights: {(W_directed > 0).sum()}")

# Show some actual connections
print(f"\n🔗 TOP CONNECTIONS:")
idx_to_word = {i: word for word, i in vocab.items()}
for i in range(len(vocab)):
    for j in range(len(vocab)):
        if W_directed[i, j] > 0:
            print(f"   {idx_to_word[i]} -> {idx_to_word[j]}: {W_directed[i, j]:.6f}")