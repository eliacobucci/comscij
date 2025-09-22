#!/usr/bin/env python3
"""
Stable IAC test interface with error handling
"""
import streamlit as st
import numpy as np
import re

st.set_page_config(page_title="🧠 Stable IAC Test", layout="wide")
st.title("🧠 Stable IAC Test Interface")

# Simple text input
text_input = st.text_area("Enter some text to test IAC:", 
                         value="The dog barks loudly. The cat meows softly. Dogs howl at night. Cats purr when happy.",
                         height=100)

# IAC controls
col1, col2 = st.columns([2, 1])
with col1:
    inhibition_baseline = st.slider(
        "🎛️ Inhibition Baseline (Experiment!)",
        min_value=0.0,
        max_value=0.01,
        value=0.002,
        step=0.0005,
        format="%.4f",
        help="Subtract this from all connections. 0 = pure covariance, higher = more competition"
    )
with col2:
    st.write("")
    st.write(f"**Current:** {inhibition_baseline:.4f}")
    if inhibition_baseline == 0.0:
        st.info("Pure covariance")
    elif inhibition_baseline < 0.003:
        st.info("Light competition")
    elif inhibition_baseline < 0.007:
        st.warning("Strong competition") 
    else:
        st.error("Heavy competition")

if st.button("🧠 Test IAC", key="test_button"):
    if text_input.strip():
        try:
            with st.spinner("Processing..."):
                
                # Simple tokenization
                words = re.findall(r'\b\w+\b', text_input.lower())
                if len(words) < 2:
                    st.error("Need at least 2 words to analyze")
                    st.stop()
                
                unique_words = list(set(words))
                if len(unique_words) > 50:
                    st.warning(f"Too many unique words ({len(unique_words)}). Using first 50.")
                    unique_words = unique_words[:50]
                
                vocab = {word: i for i, word in enumerate(unique_words)}
                n = len(vocab)
                
                st.write(f"**Words found:** {len(words)}")
                st.write(f"**Unique words:** {n}")
                
                # Simple covariance computation (no external dependencies)
                co_matrix = np.zeros((n, n))
                
                # Sliding window processing
                window_size = 3
                windows_processed = 0
                
                for i in range(len(words) - window_size + 1):
                    window_words = words[i:i + window_size]
                    # Convert to indices
                    indices = []
                    for word in window_words:
                        if word in vocab:
                            indices.append(vocab[word])
                    
                    # Update co-occurrence matrix
                    for j in range(len(indices)):
                        for k in range(j + 1, len(indices)):
                            a, b = indices[j], indices[k]
                            if a != b:
                                co_matrix[a, b] += 1.0
                                co_matrix[b, a] += 1.0
                    
                    windows_processed += 1
                    
                    # Safety check
                    if windows_processed > 1000:
                        st.warning("Stopped at 1000 windows to prevent freezing")
                        break
                
                # Normalize by number of windows
                if windows_processed > 0:
                    W_covariance = co_matrix / windows_processed
                else:
                    W_covariance = co_matrix
                
                # Apply inhibition baseline for competition
                W_iac = W_covariance - inhibition_baseline
                # Keep diagonal at zero - "wherever you go, there you are"
                np.fill_diagonal(W_iac, 0.0)
                
                st.success(f"✅ Processed {windows_processed} windows with baseline {inhibition_baseline:.4f}")
                
                # Show results safely
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 IAC Matrix Stats")
                    st.write(f"- Matrix size: {W_iac.shape}")
                    
                    pos_connections = np.sum(W_iac > 1e-6)
                    neg_connections = np.sum(W_iac < -1e-6)
                    zero_connections = np.sum(np.abs(W_iac) <= 1e-6)
                    
                    st.write(f"- **Positive** (cooperative): {pos_connections}")
                    st.write(f"- **Negative** (competitive): {neg_connections}")
                    st.write(f"- **Near-zero** (neutral): {zero_connections}")
                    
                    if np.max(W_iac) != np.min(W_iac):  # Safety check
                        st.write(f"- Max value: {np.max(W_iac):.6f}")
                        st.write(f"- Min value: {np.min(W_iac):.6f}")
                    
                    # Show the transformation effect
                    st.write("---")
                    st.write("**Inhibition Effect:**")
                    orig_pos = np.sum(W_covariance > 1e-6)
                    st.write(f"- Before: {orig_pos} positive, 0 negative")
                    st.write(f"- After: {pos_connections} positive, {neg_connections} negative")
                
                with col2:
                    st.subheader("🔗 Strongest Connections")
                    
                    # Find strongest positive connections safely
                    if np.max(W_iac) > 1e-6:
                        pos_idx = np.unravel_index(np.argmax(W_iac), W_iac.shape)
                        if pos_idx[0] < len(unique_words) and pos_idx[1] < len(unique_words):
                            word1 = unique_words[pos_idx[0]]
                            word2 = unique_words[pos_idx[1]]
                            st.write(f"**Strongest positive:** {word1} ↔ {word2} ({np.max(W_iac):.4f})")
                    
                    # Find strongest negative connections safely
                    if np.min(W_iac) < -1e-6:
                        neg_idx = np.unravel_index(np.argmin(W_iac), W_iac.shape)
                        if neg_idx[0] < len(unique_words) and neg_idx[1] < len(unique_words):
                            word1 = unique_words[neg_idx[0]]
                            word2 = unique_words[neg_idx[1]]
                            st.write(f"**Strongest negative:** {word1} ↔ {word2} ({np.min(W_iac):.4f})")
                
                # Show sample connections safely
                if len(unique_words) >= 4:
                    st.subheader("🎯 Sample Connections")
                    sample_size = min(6, len(unique_words))
                    sample_words = unique_words[:sample_size]
                    sample_indices = [vocab[word] for word in sample_words]
                    
                    # Create sample matrix safely
                    sample_matrix = np.zeros((sample_size, sample_size))
                    for i, idx_i in enumerate(sample_indices):
                        for j, idx_j in enumerate(sample_indices):
                            if idx_i < W_iac.shape[0] and idx_j < W_iac.shape[1]:
                                sample_matrix[i, j] = W_iac[idx_i, idx_j]
                    
                    # Display as simple table
                    st.write("**Connection Matrix (sample):**")
                    for i, word_i in enumerate(sample_words):
                        row_str = f"{word_i:>8}: "
                        for j, word_j in enumerate(sample_words):
                            val = sample_matrix[i, j]
                            if abs(val) < 1e-5:
                                row_str += "   .   "
                            elif val > 0:
                                row_str += f" +{val:.3f}"
                            else:
                                row_str += f" {val:.3f}"
                        st.code(row_str)
        
        except Exception as e:
            st.error(f"Error processing: {str(e)}")
            st.write("Try with shorter text or different baseline value")
    
    else:
        st.warning("Please enter some text to analyze")

st.write("---")
st.write("🧠 **About IAC**: Subtracts inhibition baseline from covariance matrix. Co-occurring words stay positive, non-co-occurring become negative (competitive).")