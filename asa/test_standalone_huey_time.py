#!/usr/bin/env python3
"""
Test standalone HueyTime with Feynman file to verify it still works
"""
import re
from collections import Counter
from huey_time import HueyTime, HueyTimeConfig, build_vocab

# Test with actual Feynman content
try:
    with open('Richard_Feynman.pdf', 'r', encoding='utf-8') as f:
        content = f.read()
    print("📁 Using Richard_Feynman.pdf")
except:
    try:
        with open('feynman total.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        print("📁 Using feynman total.txt")
    except:
        print("❌ No Feynman file found")
        exit(1)

print("🧪 TESTING STANDALONE HUEY TIME")
print("=" * 60)
print(f"📝 Content length: {len(content)} characters")

# Tokenize the content
words = re.findall(r'\b\w+\b', content.lower())
print(f"🔤 Total words: {len(words)}")

if not words:
    print("❌ No words found!")
    exit(1)

# Apply same vocabulary filtering as the web interface
word_counts = Counter(words)
max_vocab_size = 2000

print(f"📚 Unique words before filtering: {len(word_counts)}")

if len(word_counts) > max_vocab_size:
    print(f"🎯 Filtering to top {max_vocab_size} most frequent words")
    most_frequent_words = [word for word, count in word_counts.most_common(max_vocab_size)]
    words = [word for word in words if word in most_frequent_words]
    
# Build vocabulary
vocab = build_vocab(words)
print(f"📚 Final vocabulary size: {len(vocab)}")

# Configure HueyTime exactly like the web interface
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
print("🚀 Starting HueyTime processing...")
huey_time = HueyTime(config)

# Find sentence boundaries
sentences = re.split(r'[.!?]+', content)
current_pos = 0
boundaries = []

for sentence in sentences[:-1]:
    current_pos += len(re.findall(r'\b\w+\b', sentence.lower()))
    if current_pos < len(words):
        boundaries.append(current_pos - 1)

print(f"🎯 Sentence boundaries: {len(boundaries)}")

# Learn from the document
huey_time.update_doc(words, boundaries=boundaries)

# Get results
W_directed = huey_time.export_W()
S_symmetric = huey_time.export_S("avg")

# Calculate metrics exactly like the web interface should
concept_count = len(vocab)
connection_count = (W_directed > 0).sum()

print(f"\n📊 STANDALONE HUEY TIME RESULTS:")
print(f"   Concept count: {concept_count}")
print(f"   Connection count: {connection_count}")
print(f"   Max weight: {W_directed.max():.6f}")

# This should match what the web interface claims to create
concept_neurons = {i: word for word, i in vocab.items()}
print(f"   concept_neurons length: {len(concept_neurons)}")

if concept_count >= 2000:
    print("✅ SUCCESS: HueyTime creates expected number of concepts")
elif concept_count < 100:
    print("❌ FAILURE: Too few concepts - something is broken")
else:
    print("⚠️ WARNING: Concepts less than expected but not broken")