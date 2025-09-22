#!/usr/bin/env python3
"""
Simple IAC test using ChatGPT-5's covariance Hebbian learning
"""

from cov_hebb import CovHebbLearner
import numpy as np

# Test the dog/cat/bark/meow/howl example
print("🧠 Testing Interactive Activation & Competition (IAC)")
print("=" * 50)

# Create vocabulary - add missing words
vocab = {
    "dog": 0, "cat": 1, "barks": 2, "meows": 3, "howls": 4,
    "me": 5, "my": 6, "myself": 7, "our": 8, "us": 9,
    "animal": 10, "pet": 11
}

# Initialize IAC learner
iac = CovHebbLearner(n=len(vocab), eta=5e-3, beta=1e-2, gamma=1e-4)

# Training windows - need anti-correlation for competitive dynamics
# Mix positive and negative evidence to create centered activations
windows = [
    # Dog contexts (barks present, meows absent)
    ["dog", "barks"], ["dog", "barks"], ["dog", "barks"], 
    ["dog", "howls"], ["dog", "howls"],
    
    # Cat contexts (meows present, barks absent)  
    ["cat", "meows"], ["cat", "meows"], ["cat", "meows"],
    ["cat", "howls"], 
    
    # Mixed contexts to create anti-correlation
    ["barks", "dog"], ["meows", "cat"], 
    ["barks", "dog"], ["meows", "cat"],
    
    # Self-concept training
    ["me", "myself"], ["my", "myself"], ["our", "us"], ["me", "our"],
    
    # More anti-correlation training
    ["dog", "barks", "animal"], ["cat", "meows", "animal"],
    ["dog", "barks", "pet"], ["cat", "meows", "pet"]
]

print(f"📝 Processing {len(windows)} training windows...")

# Train the IAC network
for i, words in enumerate(windows):
    indices = [vocab[word] for word in words if word in vocab]
    iac.update_from_window(indices)
    
    # Prune periodically
    if (i + 1) % 6 == 0:
        iac.prune_topk(8)

# Final pruning
iac.prune_topk(8)

# Get the learned matrix
test_concepts = ["dog", "cat", "barks", "meows", "howls"]
test_indices = [vocab[word] for word in test_concepts]
W_iac = iac.to_dense_block(test_indices)

print(f"\n🎯 Learned IAC Matrix ({' '.join(test_concepts)}):")
print("=" * 50)

# Format the matrix nicely
for i, word_i in enumerate(test_concepts):
    row_str = f"{word_i:>6}: "
    for j, word_j in enumerate(test_concepts):
        val = W_iac[i, j]
        if abs(val) < 1e-6:
            row_str += "   .   "
        else:
            row_str += f"{val:6.3f} "
    print(row_str)

print(f"\n📊 Matrix Analysis:")
print(f"  - Positive connections: {np.sum(W_iac > 1e-6)}")
print(f"  - Negative connections: {np.sum(W_iac < -1e-6)}")
print(f"  - Max value: {np.max(W_iac):.6f}")
print(f"  - Min value: {np.min(W_iac):.6f}")

# Check specific relationships
dog_idx = test_concepts.index("dog")
cat_idx = test_concepts.index("cat") 
bark_idx = test_concepts.index("barks")
meow_idx = test_concepts.index("meows")
howl_idx = test_concepts.index("howls")

print(f"\n🔍 Key Relationships:")
print(f"  - dog ↔ barks:  {W_iac[dog_idx, bark_idx]:6.3f}")
print(f"  - cat ↔ meows:  {W_iac[cat_idx, meow_idx]:6.3f}")
print(f"  - dog ↔ meows:  {W_iac[dog_idx, meow_idx]:6.3f}")
print(f"  - cat ↔ barks:  {W_iac[cat_idx, bark_idx]:6.3f}")
print(f"  - dog ↔ howls:  {W_iac[dog_idx, howl_idx]:6.3f}")
print(f"  - cat ↔ howls:  {W_iac[cat_idx, howl_idx]:6.3f}")

# Check for competitive dynamics
if W_iac[dog_idx, meow_idx] < 0:
    print("\n✅ SUCCESS: dog-meows shows negative connection (competitive inhibition)")
else:
    print(f"\n⚠️  dog-meows connection is {W_iac[dog_idx, meow_idx]:.3f} (expected negative)")

if W_iac[cat_idx, bark_idx] < 0:
    print("✅ SUCCESS: cat-barks shows negative connection (competitive inhibition)")
else:
    print(f"⚠️  cat-barks connection is {W_iac[cat_idx, bark_idx]:.3f} (expected negative)")

print(f"\n🎉 IAC Test Complete!")