#!/usr/bin/env python3
"""
Simple command-line IAC demo - no Streamlit, just results
"""
import numpy as np
import re

def simple_covariance_matrix(words, vocab):
    """Build covariance matrix from word list"""
    n = len(vocab)
    co_matrix = np.zeros((n, n))
    
    # Sliding window processing
    window_size = 3
    windows_processed = 0
    
    for i in range(len(words) - window_size + 1):
        window_words = words[i:i + window_size]
        indices = [vocab[word] for word in window_words if word in vocab]
        
        # Update co-occurrence matrix
        for j in range(len(indices)):
            for k in range(j + 1, len(indices)):
                a, b = indices[j], indices[k]
                if a != b:
                    co_matrix[a, b] += 1.0
                    co_matrix[b, a] += 1.0
        
        windows_processed += 1
    
    # Normalize
    if windows_processed > 0:
        co_matrix = co_matrix / windows_processed
    
    return co_matrix

def test_iac_with_baseline(text, inhibition_baseline):
    """Test IAC with given inhibition baseline"""
    print(f"\n{'='*60}")
    print(f"Testing IAC with inhibition baseline: {inhibition_baseline:.4f}")
    print(f"{'='*60}")
    
    # Tokenize
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = list(set(words))
    vocab = {word: i for i, word in enumerate(unique_words)}
    
    print(f"Words: {len(words)}, Unique: {len(unique_words)}")
    print(f"Vocabulary: {', '.join(unique_words[:10])}")
    
    # Build covariance matrix
    W_covariance = simple_covariance_matrix(words, vocab)
    
    # Apply IAC transformation
    W_iac = W_covariance - inhibition_baseline
    np.fill_diagonal(W_iac, 0.0)  # "Wherever you go, there you are"
    
    # Results
    pos_connections = np.sum(W_iac > 1e-6)
    neg_connections = np.sum(W_iac < -1e-6)
    
    print(f"\nResults:")
    print(f"  Positive connections: {pos_connections}")
    print(f"  Negative connections: {neg_connections}")
    print(f"  Max value: {np.max(W_iac):.6f}")
    print(f"  Min value: {np.min(W_iac):.6f}")
    
    # Show strongest connections
    if np.max(W_iac) > 1e-6:
        pos_idx = np.unravel_index(np.argmax(W_iac), W_iac.shape)
        word1, word2 = unique_words[pos_idx[0]], unique_words[pos_idx[1]]
        print(f"  Strongest positive: {word1} ↔ {word2} ({np.max(W_iac):.4f})")
    
    if np.min(W_iac) < -1e-6:
        neg_idx = np.unravel_index(np.argmin(W_iac), W_iac.shape)
        word1, word2 = unique_words[neg_idx[0]], unique_words[neg_idx[1]]
        print(f"  Strongest negative: {word1} ↔ {word2} ({np.min(W_iac):.4f})")
    
    # Show sample matrix
    if len(unique_words) >= 4:
        print(f"\nSample connections matrix:")
        sample_words = unique_words[:min(6, len(unique_words))]
        print(f"       ", end="")
        for word in sample_words:
            print(f"{word:>8}", end="")
        print()
        
        for i, word_i in enumerate(sample_words):
            print(f"{word_i:>6}:", end="")
            for j, word_j in enumerate(sample_words):
                val = W_iac[vocab[word_i], vocab[word_j]]
                if abs(val) < 1e-5:
                    print("   .   ", end="")
                elif val > 0:
                    print(f" +{val:.3f}", end="")
                else:
                    print(f" {val:.3f}", end="")
            print()

# Test text
test_text = """
The dog barks loudly at strangers. Dogs are loyal pets.
The cat meows softly when hungry. Cats purr when content.
Both animals howl sometimes. Pets need food and water.
Dogs bark to communicate. Cats meow for attention.
"""

if __name__ == "__main__":
    print("🧠 Simple IAC Demo - Command Line")
    
    # Test different baselines
    for baseline in [0.0, 0.001, 0.002, 0.005, 0.008]:
        test_iac_with_baseline(test_text, baseline)
    
    print(f"\n{'='*60}")
    print("✅ IAC Demo Complete!")
    print("Notice how inhibition baseline affects positive vs negative connections.")
    print("Higher baseline = more competition, lower baseline = more cooperation.")