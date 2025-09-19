#!/usr/bin/env python3
"""
RESET TEST - Minimal HueyTime test with tiny content
"""
from huey_time import HueyTime, HueyTimeConfig, build_vocab
import re

# Tiny test content
content = "The cat sat on the mat. The dog ran in the park. The bird flew away."

print("🔄 RESET TEST")
print("="*40)
print(f"Content: {content}")

# Basic tokenization
words = re.findall(r'\b\w+\b', content.lower())
print(f"Words: {words}")
print(f"Word count: {len(words)}")

# Build vocab
vocab = build_vocab(words)
print(f"Vocab: {vocab}")
print(f"Vocab size: {len(vocab)}")

# Configure HueyTime
config = HueyTimeConfig(vocab=vocab, method="lagged")
huey_time = HueyTime(config)

# Process with NO BOUNDARIES
print("Processing with boundaries=None...")
huey_time.update_doc(words, boundaries=None)

# Get results
W = huey_time.export_W()
concepts = len(vocab)
connections = (W > 0).sum()

print(f"Concepts: {concepts}")
print(f"Connections: {connections}")
print(f"Expected concepts: {len(vocab)} (one per unique word)")

if concepts == len(vocab):
    print("✅ SUCCESS: Correct concept count")
else:
    print("❌ FAILED: Wrong concept count")