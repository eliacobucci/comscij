#!/usr/bin/env python3
"""
Simple HueyTime Interface - Direct Processing
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import re
from collections import Counter
from huey_time import HueyTime, HueyTimeConfig, build_vocab

st.set_page_config(page_title="🕐 HueyTime Direct", layout="wide")

st.title("🕐 HueyTime Direct Interface")
st.write("Simple interface that directly uses HueyTime temporal learning")

# File upload
uploaded_file = st.file_uploader("Upload text file", type=['txt', 'pdf'])

if uploaded_file:
    # Read content
    if uploaded_file.name.endswith('.pdf'):
        st.error("PDF support removed for simplicity - use TXT files")
    else:
        content = uploaded_file.getvalue().decode('utf-8')
    
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Length:** {len(content)} characters")
    
    if st.button("🚀 Process with HueyTime"):
        with st.spinner("Processing with HueyTime..."):
            # Tokenize
            words = re.findall(r'\b\w+\b', content.lower())
            st.write(f"📝 **Total words:** {len(words)}")
            
            # Build vocabulary (no filtering initially)
            vocab = build_vocab(words)
            st.write(f"📚 **Unique words:** {len(vocab)}")
            
            if len(vocab) > 1000:
                st.warning(f"⚠️ Large vocabulary ({len(vocab)} words) - this may take time")
            
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
            
            # Initialize HueyTime
            huey_time = HueyTime(config)
            
            # NO SENTENCE BOUNDARIES - let HueyTime see continuous temporal flow
            st.write("🌊 **Processing as continuous temporal sequence** (no boundaries)")
            
            # Process document
            try:
                huey_time.update_doc(words, boundaries=None)
                
                # Get results
                W_directed = huey_time.export_W()
                S_symmetric = huey_time.export_S("avg")
                
                # Calculate results
                concept_count = len(vocab)
                connection_count = (W_directed > 0).sum()
                max_weight = W_directed.max()
                avg_weight = W_directed[W_directed > 0].mean() if connection_count > 0 else 0
                
                # Display results
                st.success("✅ **HueyTime Processing Complete!**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Concepts", concept_count)
                with col2:
                    st.metric("Connections", connection_count)
                with col3:
                    st.metric("Max Weight", f"{max_weight:.4f}")
                with col4:
                    st.metric("Avg Weight", f"{avg_weight:.4f}")
                
                # Show top concepts by frequency
                st.subheader("📊 Top Concepts by Frequency")
                word_counts = Counter(words)
                top_concepts = word_counts.most_common(20)
                df = pd.DataFrame(top_concepts, columns=['Concept', 'Frequency'])
                st.dataframe(df)
                
                # Show strongest connections
                st.subheader("🔗 Strongest Connections")
                idx_to_word = {i: word for word, i in vocab.items()}
                connections = []
                
                for i in range(len(vocab)):
                    for j in range(len(vocab)):
                        if W_directed[i, j] > 0:
                            connections.append({
                                'From': idx_to_word[i],
                                'To': idx_to_word[j],
                                'Weight': W_directed[i, j]
                            })
                
                if connections:
                    conn_df = pd.DataFrame(connections)
                    conn_df = conn_df.sort_values('Weight', ascending=False).head(20)
                    st.dataframe(conn_df)
                
                # Simple 3D visualization if enough concepts
                if concept_count >= 3:
                    st.subheader("🎯 3D Concept Visualization")
                    
                    # Use symmetric matrix for visualization
                    try:
                        eigenvals, eigenvecs = np.linalg.eigh(S_symmetric)
                        # Take top 3 eigenvectors for 3D coords
                        if len(eigenvals) >= 3:
                            coords = eigenvecs[:, -3:]  # Top 3 eigenvectors
                            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
                        else:
                            # Fallback to random positions
                            n = concept_count
                            x, y, z = np.random.randn(n), np.random.randn(n), np.random.randn(n)
                        
                        # Create 3D plot
                        fig = go.Figure()
                        
                        # Get concept names and frequencies
                        concept_names = [idx_to_word[i] for i in range(concept_count)]
                        concept_freqs = [word_counts[name] for name in concept_names]
                        
                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='markers+text',
                            marker=dict(
                                size=[freq/max(concept_freqs)*20+5 for freq in concept_freqs],
                                color=concept_freqs,
                                colorscale='Viridis',
                                showscale=True
                            ),
                            text=concept_names,
                            textposition="middle center",
                            hovertemplate="<b>%{text}</b><br>Frequency: %{marker.color}<extra></extra>"
                        ))
                        
                        fig.update_layout(
                            title=f"3D Concept Space ({concept_count} concepts)",
                            scene=dict(
                                xaxis_title="Dimension 1",
                                yaxis_title="Dimension 2", 
                                zaxis_title="Dimension 3"
                            ),
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Visualization error: {e}")
                        st.info("Using random coordinates for visualization")
                        
                # Store results in session state for potential tool additions
                st.session_state.huey_results = {
                    'W_directed': W_directed,
                    'S_symmetric': S_symmetric,
                    'vocab': vocab,
                    'concept_count': concept_count,
                    'connection_count': connection_count,
                    'word_frequencies': dict(word_counts)
                }
                
            except Exception as e:
                st.error(f"❌ HueyTime processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())