#!/usr/bin/env python3
"""
Test the new Competitive IAC with true negative connections
"""

from competitive_iac import CompetitiveIAC
import numpy as np

print("🧠 Testing Competitive IAC with True Negative Connections")
print("=" * 60)

# Create vocabulary  
vocab = {
    "dog": 0, "cat": 1, "barks": 2, "meows": 3, "howls": 4,
    "eats": 5, "food": 6, "treats": 7, "pet": 8, "animal": 9
}

# Initialize competitive IAC
iac = CompetitiveIAC(n=len(vocab), eta_pos=8e-3, eta_neg=4e-3, beta=2e-2, competition_threshold=3)

# Training data with clear competitive patterns
training_windows = [
    # DOG contexts (barks, never meows)
    ["dog", "barks"], ["dog", "barks"], ["dog", "barks"], ["dog", "barks"],
    ["dog", "howls"], ["dog", "howls"], 
    ["dog", "eats", "food"], ["dog", "pet", "animal"],
    
    # CAT contexts (meows, never barks) 
    ["cat", "meows"], ["cat", "meows"], ["cat", "meows"], ["cat", "meows"],
    ["cat", "howls"],  # cats can howl sometimes
    ["cat", "eats", "food"], ["cat", "pet", "animal"],
    
    # More dog contexts
    ["dog", "barks", "loudly"], ["dog", "barks", "outside"],
    ["dog", "treats"], ["dog", "food"],
    
    # More cat contexts  
    ["cat", "meows", "softly"], ["cat", "meows", "inside"],
    ["cat", "treats"], ["cat", "food"],
    
    # General animal contexts
    ["pet", "animal"], ["animal", "food"], ["treats", "food"],
    
    # Reinforce the competing patterns
    ["dog", "barks"], ["cat", "meows"], ["dog", "barks"], ["cat", "meows"],
    ["dog", "barks"], ["cat", "meows"], ["dog", "barks"], ["cat", "meows"],
]

print(f"📝 Training with {len(training_windows)} windows...")

# Train the competitive IAC
for i, words in enumerate(training_windows):
    # Convert words to indices
    indices = [vocab[word] for word in words if word in vocab]
    if len(indices) >= 2:
        iac.update_from_window(indices)
    
    # Prune periodically  
    if (i + 1) % 10 == 0:
        iac.prune_topk(20)

# Final pruning
iac.prune_topk(20)

# Test on key concepts
test_concepts = ["dog", "cat", "barks", "meows", "howls"]
test_indices = [vocab[word] for word in test_concepts]
W_competitive = iac.to_dense_block(test_indices)

print(f"\n🎯 Competitive IAC Matrix ({' '.join(test_concepts)}):")
print("=" * 60)

# Display the matrix nicely
for i, word_i in enumerate(test_concepts):
    row_str = f"{word_i:>6}: "
    for j, word_j in enumerate(test_concepts):
        val = W_competitive[i, j]
        if abs(val) < 1e-6:
            row_str += "   .   "
        elif val > 0:
            row_str += f" +{val:.3f}"
        else:
            row_str += f" {val:.3f}"
    print(row_str)

# Get connection statistics
stats = iac.get_connection_stats()
print(f"\n📊 Connection Statistics:")
print(f"  - Total connections: {stats['total']}")
print(f"  - Positive (cooperative): {stats['positive']}")
print(f"  - Negative (competitive): {stats['negative']}")  
print(f"  - Neutral: {stats['neutral']}")

# Analyze key relationships
dog_idx = test_concepts.index("dog")
cat_idx = test_concepts.index("cat")
bark_idx = test_concepts.index("barks") 
meow_idx = test_concepts.index("meows")
howl_idx = test_concepts.index("howls")

print(f"\n🔍 Key Competitive Relationships:")
print(f"  - dog ↔ barks:  {W_competitive[dog_idx, bark_idx]:7.4f} (should be positive)")
print(f"  - cat ↔ meows:  {W_competitive[cat_idx, meow_idx]:7.4f} (should be positive)")
print(f"  - dog ↔ meows:  {W_competitive[dog_idx, meow_idx]:7.4f} (should be NEGATIVE)")
print(f"  - cat ↔ barks:  {W_competitive[cat_idx, bark_idx]:7.4f} (should be NEGATIVE)")
print(f"  - dog ↔ howls:  {W_competitive[dog_idx, howl_idx]:7.4f} (neutral/positive)")
print(f"  - cat ↔ howls:  {W_competitive[cat_idx, howl_idx]:7.4f} (neutral/positive)")

# Check if competitive dynamics worked
competitive_success = 0
total_tests = 0

if W_competitive[dog_idx, meow_idx] < -1e-4:
    print("\n✅ SUCCESS: dog-meows shows negative competitive connection!")
    competitive_success += 1
else:
    print(f"\n⚠️  dog-meows connection is {W_competitive[dog_idx, meow_idx]:.6f} (expected negative)")
total_tests += 1

if W_competitive[cat_idx, bark_idx] < -1e-4:
    print("✅ SUCCESS: cat-barks shows negative competitive connection!")
    competitive_success += 1  
else:
    print(f"⚠️  cat-barks connection is {W_competitive[cat_idx, bark_idx]:.6f} (expected negative)")
total_tests += 1

if W_competitive[dog_idx, bark_idx] > 1e-4:
    print("✅ SUCCESS: dog-barks shows positive cooperative connection!")
    competitive_success += 1
else:
    print(f"⚠️  dog-barks connection is {W_competitive[dog_idx, bark_idx]:.6f} (expected positive)")
total_tests += 1

if W_competitive[cat_idx, meow_idx] > 1e-4:
    print("✅ SUCCESS: cat-meows shows positive cooperative connection!")
    competitive_success += 1
else:
    print(f"⚠️  cat-meows connection is {W_competitive[cat_idx, meow_idx]:.6f} (expected positive)")
total_tests += 1

print(f"\n🎯 Competitive IAC Performance: {competitive_success}/{total_tests} tests passed")

if competitive_success >= 3:
    print("🎉 Competitive IAC is working! True negative connections created!")
else:
    print("🔧 Competitive IAC needs tuning - adjust eta_neg and competition_threshold")

print(f"\n💡 This demonstrates true Interactive Activation & Competition:")
print(f"   - Positive weights for co-occurring concepts (dog-barks)")  
print(f"   - Negative weights for competing concepts (dog-meows)")
print(f"   - Competitive disambiguation in action!")