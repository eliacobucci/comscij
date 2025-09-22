#!/usr/bin/env python3
"""
Simple IAC test interface - no complex visualization, just show the results
"""
import streamlit as st
import numpy as np
from cov_hebb import CovHebbLearner
import re

st.set_page_config(page_title="🧠 Simple IAC Test", layout="wide")
st.title("🧠 Simple IAC Test Interface")

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

if st.button("🧠 Test IAC"):
    if text_input.strip():
        with st.spinner("Processing with IAC..."):
            
            # Simple tokenization
            words = re.findall(r'\b\w+\b', text_input.lower())
            unique_words = list(set(words))
            vocab = {word: i for i, word in enumerate(unique_words)}
            
            st.write(f"**Words found:** {len(words)}")
            st.write(f"**Unique words:** {len(unique_words)}")
            st.write(f"**Vocabulary:** {', '.join(unique_words[:10])}")
            
            # IAC learning
            iac = CovHebbLearner(n=len(vocab), eta=5e-3, beta=1e-2, gamma=1e-4)
            
            # Process with sliding windows
            window_size = 3
            windows_processed = 0
            
            for i in range(len(words) - window_size + 1):
                window_words = words[i:i + window_size]
                window_indices = [vocab[word] for word in window_words]
                iac.update_from_window(window_indices)
                windows_processed += 1
            
            # Get results
            all_indices = list(range(len(vocab)))
            W_covariance = iac.to_dense_block(all_indices)
            
            # Apply inhibition baseline for competition
            W_iac = W_covariance - inhibition_baseline
            # Keep diagonal at zero - "wherever you go, there you are"
            np.fill_diagonal(W_iac, 0.0)
            
            st.success(f"✅ Processed {windows_processed} windows with baseline {inhibition_baseline:.4f}")
            
            # Show results
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
                
                # Find strongest positive connections
                max_pos = np.max(W_iac)
                if max_pos > 1e-6:
                    pos_idx = np.unravel_index(np.argmax(W_iac), W_iac.shape)
                    word1 = unique_words[pos_idx[0]]
                    word2 = unique_words[pos_idx[1]]
                    st.write(f"**Strongest positive:** {word1} ↔ {word2} ({max_pos:.4f})")
                
                # Find strongest negative connections  
                min_neg = np.min(W_iac)
                if min_neg < -1e-6:
                    neg_idx = np.unravel_index(np.argmin(W_iac), W_iac.shape)
                    word1 = unique_words[neg_idx[0]]
                    word2 = unique_words[neg_idx[1]]
                    st.write(f"**Strongest negative:** {word1} ↔ {word2} ({min_neg:.4f})")
            
            # Show sample of matrix for key words
            if len(unique_words) >= 4:
                st.subheader("🎯 Sample Connections")
                sample_words = unique_words[:min(6, len(unique_words))]
                sample_indices = [vocab[word] for word in sample_words]
                sample_matrix = W_iac[np.ix_(sample_indices, sample_indices)]
                
                # Create a simple table
                import pandas as pd
                df = pd.DataFrame(sample_matrix, 
                                index=sample_words, 
                                columns=sample_words)
                st.dataframe(df.round(4))
    
    else:
        st.warning("Please enter some text to analyze")

st.write("---")
st.write("🧠 **About IAC**: Interactive Activation & Competition uses covariance-based learning to create both positive (co-occurring) and negative (competing) word connections.")