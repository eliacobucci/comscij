#!/usr/bin/env python3
"""
Test HueyTime with actual Feynman content sample
"""
import re
from huey_time import HueyTime, HueyTimeConfig, build_vocab

# Get a sample from the actual Feynman file
try:
    with open('Richard_Feynman.pdf', 'r', encoding='utf-8') as f:
        content = f.read()
except:
    # If PDF doesn't work, use a longer sample
    content = """
    I was born in Far Rockaway, a suburb of New York, in 1918. I don't know what's the matter with 
    people: they don't learn by understanding; they learn by some other way - by rote or something. 
    Their knowledge is so fragile! I would teach the students about the wave equation for a vibrating 
    string. You pluck the string, and it vibrates. The amplitude of oscillation changes from place 
    to place along the string, and that's described by a mathematical function called the wave equation.
    
    I thought this was very exciting. But when the students took the exam, I would give them a 
    problem about sound. They couldn't figure it out. I said, 'Look, a sound wave is exactly the 
    same mathematically as the wave on a string.' They said, 'Yeah, but that was about strings; 
    this is about sound.' They couldn't see that the mathematics was the same.
    """

print("🧪 TESTING HUEY TIME WITH FEYNMAN SAMPLE")
print("=" * 60)
print(f"📝 Content length: {len(content)} characters")

# Tokenize the content
words = re.findall(r'\b\w+\b', content.lower())
print(f"🔤 Words found: {len(words)}")

if not words:
    print("❌ No words found!")
    exit(1)

# Build vocabulary
vocab = build_vocab(words)
print(f"📚 Vocabulary size: {len(vocab)}")

# Show some vocab samples
vocab_sample = dict(list(vocab.items())[:20])
print(f"🗂️ Vocab sample: {vocab_sample}")

# Configure HueyTime with same parameters as interface
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

# Find sentence boundaries like in the interface
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

# Calculate metrics exactly like the interface
concept_count = len(vocab)
connection_count = (W_directed > 0).sum()

print(f"\n📊 RESULTS:")
print(f"   Concept count: {concept_count}")
print(f"   Connection count: {connection_count}")
print(f"   Max weight: {W_directed.max():.6f}")
print(f"   Avg weight: {W_directed[W_directed > 0].mean():.6f}")

# Test the concept_neurons mapping like the interface
concept_neurons = {i: word for word, i in vocab.items()}
print(f"   concept_neurons length: {len(concept_neurons)}")
print(f"   Sample concepts: {dict(list(concept_neurons.items())[:10])}")

# Test if this would pass the 3D visualization requirement
if concept_count >= 3:
    print("✅ SUFFICIENT CONCEPTS for 3D visualization")
else:
    print("❌ INSUFFICIENT CONCEPTS for 3D visualization")