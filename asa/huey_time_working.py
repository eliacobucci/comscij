#!/usr/bin/env python3
"""
WORKING HueyTime Interface - Dead Simple
Uses only the proven HueyTime core that works correctly.
"""
import streamlit as st
import re
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from huey_time import HueyTime, HueyTimeConfig, build_vocab
import os
from huey_activation_cascade import HueyActivationCascade, create_cascade_interface

# Language detection (ChatGPT-5's code)
def detect_language(text):
    """Simple language detection based on character ranges."""
    # Japanese (Hiragana, Katakana, Kanji)
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
        return 'ja'
    
    # Korean (Hangul)
    if re.search(r'[\uAC00-\uD7AF]', text):
        return 'ko'
    
    # Chinese (CJK Unified Ideographs)
    if re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh'
    
    # Default to English
    return 'en'

def load_kill_words(language_code):
    """Load appropriate kill words file for detected language."""
    kill_file = f"huey_kill.{language_code}.txt"
    
    if not os.path.exists(kill_file):
        # Fallback to English if specific language not found
        kill_file = "huey_kill.en.txt"
        
    if not os.path.exists(kill_file):
        # Final fallback to universal
        kill_file = "huey_kill.universal.txt"
        
    if not os.path.exists(kill_file):
        st.warning("⚠️ No kill words file found - proceeding without filtering")
        return set()
    
    try:
        with open(kill_file, 'r', encoding='utf-8') as f:
            kill_words = set(line.strip().lower() for line in f if line.strip())
        st.info(f"📚 Loaded {len(kill_words)} kill words from {kill_file}")
        return kill_words
    except Exception as e:
        st.warning(f"⚠️ Error loading kill words: {e}")
        return set()

# Import JAX for GPU operations with CPU fallback for unsupported ops
try:
    import jax
    import jax.numpy as jnp
    from jax import device_put, jit
    from functools import partial
    jax_available = True
    st.write("🚀 JAX available - GPU for supported ops, CPU for eigendecomposition")
    
    # ChatGPT-5's solution: CPU decorator for operations Metal doesn't support
    @partial(jit, backend="cpu")
    def cpu_eigendecomposition(matrix):
        """Force eigendecomposition to run on CPU where it's supported"""
        return jnp.linalg.eigh(matrix)
        
except ImportError:
    jax_available = False

st.set_page_config(page_title="🕐 HueyTime WORKING", layout="wide")

st.title("🕐 HueyTime - WORKING Interface")
st.write("Dead simple interface using only proven HueyTime core")

# Reset button
if st.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Show current status
if 'huey_results' in st.session_state:
    st.info("📊 Previous results available - upload new file to process or use tools below")
else:
    st.info("📁 Ready to process new file")

# File upload
uploaded_file = st.file_uploader("Upload text file", type=['txt'])

if uploaded_file:
    content = uploaded_file.getvalue().decode('utf-8')
    
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Length:** {len(content)} characters")
    
    if st.button("🚀 Process with HueyTime"):
        with st.spinner("Processing..."):
            
            # Detect language and load appropriate kill words
            detected_language = detect_language(content)
            st.success(f"🌍 **Detected language:** {detected_language}")
            
            kill_words = load_kill_words(detected_language)
            
            # EXACT same code as reset_test.py that works
            words = re.findall(r'\b\w+\b', content.lower())
            st.write(f"**Words before filtering:** {len(words)}")
            
            # Filter out kill words if available
            if kill_words:
                filtered_words = [word for word in words if word not in kill_words]
                st.write(f"**Words after kill word filtering:** {len(filtered_words)} (removed {len(words) - len(filtered_words)})")
                words = filtered_words
            else:
                st.write(f"**Words:** {len(words)} (no kill word filtering)")
            
            vocab = build_vocab(words)
            st.write(f"**Unique words:** {len(vocab)}")
            
            # Configure HueyTime exactly like working test
            config = HueyTimeConfig(vocab=vocab, method="lagged")
            st.write(f"🔧 HueyTime config: method={config.method}, max_lag={config.max_lag}, tau={config.tau}")
            st.write(f"🔧 Learning rates: eta_fwd={config.eta_fwd}, eta_fb={config.eta_fb}")
            huey_time = HueyTime(config)
            
            # Process with NO BOUNDARIES (exactly like working test)
            st.write("🌊 Processing as continuous sequence...")
            huey_time.update_doc(words, boundaries=None)
            
            # Get results exactly like working test
            W = huey_time.export_W()
            S = huey_time.export_S("avg")
            concepts = len(vocab)
            connections = (W > 0).sum()
            
            # Debug the matrices
            st.write(f"🔍 **Matrix Analysis:**")
            st.write(f"  - W shape: {W.shape}")
            st.write(f"  - W max: {np.max(W):.6f}, min: {np.min(W):.6f}, mean: {np.mean(W):.6f}")
            st.write(f"  - W non-zero: {np.count_nonzero(W)}")
            st.write(f"  - S max: {np.max(S):.6f}, min: {np.min(S):.6f}, mean: {np.mean(S):.6f}")
            st.write(f"  - S non-zero: {np.count_nonzero(S)}")
            
            # Store results in session state for tools
            st.session_state.huey_results = {
                'huey_time': huey_time,
                'W': W,
                'S': S,
                'vocab': vocab,
                'words': words,
                'concepts': concepts,
                'connections': connections,
                'detected_language': detected_language,
                'kill_words_used': len(kill_words) if kill_words else 0
            }
            
            # Display results
            st.success("✅ HueyTime Processing Complete!")
            
            # Language and filtering info
            language_names = {
                'en': 'English', 'ja': 'Japanese', 'ko': 'Korean', 
                'zh': 'Chinese', 'es': 'Spanish', 'fr': 'French',
                'de': 'German', 'it': 'Italian', 'is': 'Icelandic', 'ta': 'Tamil'
            }
            lang_name = language_names.get(detected_language, detected_language.upper())
            
            st.info(f"🌍 **Language:** {lang_name} | 📚 **Kill words filtered:** {len(kill_words) if kill_words else 0}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Concepts", concepts)
            with col2:
                st.metric("Connections", connections)
            
            # Verify it worked correctly
            if concepts == len(vocab):
                st.success("✅ Correct: One concept per unique word")
            else:
                st.error(f"❌ Wrong: Expected {len(vocab)} concepts, got {concepts}")
            
            # Show some vocabulary
            st.subheader("📚 Vocabulary Sample")
            vocab_items = list(vocab.items())[:20]  # First 20 words
            for word, idx in vocab_items:
                st.write(f"- {word} (index {idx})")
            
            if len(vocab) > 20:
                st.write(f"... and {len(vocab) - 20} more words")
            
# 3D Visualization (outside processing button, uses session state)
if 'huey_results' in st.session_state:
    st.subheader("🎯 3D Visualization Controls")
    
    # User controls for visualization
    vocab_size = len(st.session_state.huey_results['vocab'])
    max_concepts = st.number_input(
        "📊 Total concepts to show (all will be labeled)",
        min_value=1,
        max_value=vocab_size,
        value=min(200, vocab_size),
        help=f"Enter any number from 1 to {vocab_size} concepts to visualize"
    )
    
    if st.button("🎯 Create 3D Visualization"):
        st.write("DEBUG: Button clicked!")
        st.subheader("🎯 3D Concept Visualization")
        
        with st.spinner("Creating 3D plot..."):
            st.write("DEBUG: Starting visualization...")
            try:
                # Get data from session state
                results = st.session_state.huey_results
                S = results['S']
                vocab = results['vocab']
                words = results['words']
                
                st.write(f"DEBUG: Matrix shape: {S.shape}")
                
                # Proper Galileo Torgerson double-centering for pseudo-Riemannian space
                st.write("🌌 Using Torgerson double-centering for proper Galileo plot...")
                
                # Use the similarity matrix S (not weight matrix W) for proper semantic clustering
                similarity_matrix = S  # S contains the learned similarities, not raw weights
                
                # Torgerson transformation
                n = similarity_matrix.shape[0]
                
                # S is already symmetric similarity matrix from HueyTime
                similarity = similarity_matrix
                
                # Convert similarities to pseudo-distances
                max_sim = np.max(np.abs(similarity))
                if max_sim > 1e-10:
                    pseudo_distances = np.sign(similarity) * (max_sim - np.abs(similarity)) / max_sim
                else:
                    pseudo_distances = np.zeros_like(similarity)
                
                # Square the pseudo-distances 
                distances_squared = np.sign(pseudo_distances) * (pseudo_distances ** 2)
                
                # Double centering to get Gram matrix
                ones = np.ones((n, n))
                centering_matrix = np.eye(n) - (1.0 / n) * ones
                gram_matrix = -0.5 * centering_matrix @ distances_squared @ centering_matrix
                
                # Eigendecomposition of Gram matrix (can have negative eigenvalues)
                if jax_available:
                    st.write("🚀 Using JAX CPU for Gram matrix eigendecomposition...")
                    gram_jax = jnp.array(gram_matrix)
                    eigenvals, eigenvecs = cpu_eigendecomposition(gram_jax)
                    eigenvals = np.array(eigenvals)
                    eigenvecs = np.array(eigenvecs)
                else:
                    eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
                
                # Sort by eigenvalue magnitude (Galileo style)
                idx = np.argsort(np.abs(eigenvals))[::-1]
                sorted_eigenvals = eigenvals[idx]
                sorted_eigenvecs = eigenvecs[:, idx]
                
                st.write(f"🌌 Torgerson transform complete: {len(eigenvals)} eigenvalues")
                st.write(f"📊 Eigenvalue signature: {np.sum(eigenvals > 0)} positive, {np.sum(eigenvals < 0)} negative")
                
                # Use top 3 eigenvectors for 3D coordinates (by magnitude)
                if len(sorted_eigenvals) >= 3:
                    # Scale by square root of absolute eigenvalue (Galileo scaling)
                    coords = np.zeros((n, 3))
                    for i in range(3):
                        eigenval = sorted_eigenvals[i]
                        eigenvec = sorted_eigenvecs[:, i]
                        if abs(eigenval) > 1e-10:
                            coords[:, i] = eigenvec * np.sqrt(abs(eigenval)) * np.sign(eigenval)
                        else:
                            coords[:, i] = eigenvec * 1e-3
                    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
                else:
                    # Fallback for small vocabularies
                    x, y, z = np.random.randn(n), np.random.randn(n), np.random.randn(n)
                
                # Get word frequencies and filter to user-selected number of concepts
                word_counts = Counter(words)
                
                # Keep only user-selected number of most frequent concepts for visualization
                top_concepts = word_counts.most_common(max_concepts)
                top_words = [word for word, freq in top_concepts]
                
                # Get indices of top concepts in vocab
                top_indices = [vocab[word] for word in top_words if word in vocab]
                
                # Filter coordinates and data to top concepts only
                x_top = x[top_indices]
                y_top = y[top_indices] 
                z_top = z[top_indices]
                
                concept_names = top_words
                concept_freqs = [freq for word, freq in top_concepts]
                max_freq = max(concept_freqs)
                
                # Store 3D coordinates for cascade visualization
                concept_positions = {}
                for i, concept_name in enumerate(concept_names):
                    concept_id = vocab[concept_name]
                    concept_positions[concept_id] = (x_top[i], y_top[i], z_top[i])
                st.session_state['visualization_coords'] = concept_positions
                
                st.write(f"DEBUG: Showing top {len(concept_names)} most frequent concepts (from {len(vocab)} total)")
                
                # Create 3D plot
                fig = go.Figure()
                
                # Single scatter plot with ALL concepts labeled
                fig.add_trace(go.Scatter3d(
                    x=x_top, y=y_top, z=z_top,
                    mode='markers+text',
                    marker=dict(
                        size=[freq/max_freq*15+5 for freq in concept_freqs],
                        color=concept_freqs,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Word Frequency"),
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    text=concept_names,
                    textposition="top center",
                    textfont=dict(
                        size=12, 
                        color='black', 
                        family='Arial, sans-serif'
                    ),
                    name=f"All {len(concept_names)} Concepts (labeled)",
                    hovertemplate="<b>%{text}</b><br>Frequency: %{marker.color}<extra></extra>"
                ))
                
                fig.update_layout(
                    title=f"3D Concept Space (top {len(concept_names)} of {len(vocab)} concepts)",
                    scene=dict(
                        xaxis_title="Dimension 1",
                        yaxis_title="Dimension 2", 
                        zaxis_title="Dimension 3"
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ 3D visualization created successfully!")
                
            except Exception as e:
                st.error(f"❌ Visualization error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ChatGPT-5's Activation Cascade (outside processing button, uses session state)
if 'huey_results' in st.session_state:
    st.subheader("🌊 Activation Cascade Analysis")
    st.write("Watch how activation flows through the network from input concepts to target concepts")
    
    # Create mock network object for cascade interface
    class MockNetwork:
        def __init__(self, huey_results):
            self.vocab = huey_results['vocab']
            self.W = huey_results['W']
            self.S = huey_results['S']
            
            # Create mappings for cascade interface
            self.neuron_to_word = {idx: word for word, idx in self.vocab.items()}
            self.word_to_neuron = self.vocab
            
            # Use W matrix connections for cascade
            self.connections = {}
            for i in range(self.W.shape[0]):
                for j in range(self.W.shape[1]):
                    if self.W[i, j] > 0:
                        self.connections[(i, j)] = self.W[i, j]
    
    try:
        # Create mock network and cascade interface
        mock_network = MockNetwork(st.session_state.huey_results)
        cascade = create_cascade_interface(mock_network)
        
        # Get available concepts
        available_concepts = cascade.get_available_concepts()
        
        col1, col2 = st.columns(2)
        with col1:
            input_concepts = st.multiselect(
                "🎯 Select input concepts (stimuli)",
                available_concepts,
                help="Choose concepts to initially activate"
            )
        
        with col2:
            target_concept = st.selectbox(
                "🏁 Select target concept",
                available_concepts,
                help="Choose the concept you want to activate"
            )
        
        # Cascade controls
        col3, col4 = st.columns(2)
        with col3:
            max_steps = st.number_input(
                "⏰ Maximum cascade steps",
                min_value=10,
                max_value=200,
                value=50,
                help="Maximum number of propagation steps"
            )
        
        with col4:
            input_strength = st.slider(
                "⚡ Input strength",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Initial activation strength"
            )
        
        # Run cascade button
        if st.button("🚀 Run Activation Cascade"):
            if input_concepts and target_concept:
                with st.spinner("Running activation cascade..."):
                    # Run the cascade
                    cascade_steps = cascade.run_cascade(
                        input_concepts=input_concepts,
                        target_concept=target_concept,
                        max_steps=max_steps,
                        input_strength=input_strength
                    )
                    
                    # Display results
                    st.success(f"✅ Cascade completed in {len(cascade_steps)} steps")
                    
                    # Show step-by-step progression
                    st.subheader("📊 Cascade Progression")
                    for i, step in enumerate(cascade_steps[:10]):  # Show first 10 steps
                        with st.expander(f"Step {step.step_number}: {step.description}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Total Energy:** {step.total_energy:.4f}")
                                st.write(f"**Newly Activated:** {len(step.newly_activated)}")
                            with col2:
                                if step.newly_activated:
                                    st.write("**New Activations:**")
                                    for concept in step.newly_activated[:5]:
                                        activation = step.activations.get(concept, 0)
                                        st.write(f"- {concept}: {activation:.4f}")
                    
                    # Show final target activation
                    final_step = cascade_steps[-1]
                    target_activation = final_step.activations.get(target_concept, 0)
                    
                    if target_activation >= cascade.activation_threshold:
                        st.success(f"🎯 **Target '{target_concept}' ACTIVATED!** Final activation: {target_activation:.4f}")
                    else:
                        st.warning(f"🎯 Target '{target_concept}' not activated. Final activation: {target_activation:.4f}")
                    
                    # Create animated visualization if we have 3D coordinates
                    if 'visualization_coords' in st.session_state:
                        st.subheader("🎬 Animated Cascade Visualization")
                        concept_positions = st.session_state['visualization_coords']
                        
                        try:
                            cascade_fig = cascade.create_cascade_visualization(
                                cascade_steps, concept_positions
                            )
                            st.plotly_chart(cascade_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Visualization error: {e}")
                    else:
                        st.info("💡 Create a 3D visualization first to see animated cascade")
            else:
                st.error("Please select both input concepts and a target concept")
        
        # Show suggestions for target
        if target_concept:
            st.subheader(f"💡 Suggested inputs for '{target_concept}'")
            suggestions = cascade.suggest_inputs_for_target(target_concept, top_k=5)
            if suggestions:
                for concept, strength in suggestions:
                    st.write(f"- **{concept}** (connection strength: {strength:.4f})")
            else:
                st.write("No strong connections found to this target")
                
    except Exception as e:
        st.error(f"❌ Cascade interface error: {e}")
        import traceback
        st.code(traceback.format_exc())