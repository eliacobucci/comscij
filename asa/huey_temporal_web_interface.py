#!/usr/bin/env python3
"""
Huey🧪 Temporal Learning Web Interface
Experimental web interface for time-delay Hebbian learning.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import base64
from huey_temporal_experiment import HueyTemporalExperiment

# Page configuration
st.set_page_config(
    page_title="Huey🧪 Temporal Learning",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'temporal_huey' not in st.session_state:
        st.session_state.temporal_huey = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def create_temporal_huey(use_temporal=True, tau=3.0, eta_fwd=0.01, eta_fb=0.002, boundary_penalty=0.5):
    """Create a new temporal Huey instance."""
    return HueyTemporalExperiment(
        max_neurons=500,
        window_size=10,
        learning_rate=0.15,
        use_gpu_acceleration=True,
        use_temporal_learning=use_temporal,
        tau=tau,
        eta_fwd=eta_fwd,
        eta_fb=eta_fb,
        boundary_penalty=boundary_penalty
    )

def process_uploaded_file(uploaded_file, huey_instance):
    """Process an uploaded file with Huey."""
    if uploaded_file is None:
        return None
        
    # Save uploaded file temporarily
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Process with Huey
        result = huey_instance.process_file_with_mode(temp_path, conversation_mode=True)
        return result
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def display_comparison_metrics(huey_temporal, huey_windowed=None):
    """Display comparison metrics between temporal and windowed methods."""
    debug_temporal = huey_temporal.get_debug_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Learning Method",
            debug_temporal['learning_method'],
            help="Temporal uses time-decay weighting, Windowed uses fixed windows"
        )
    
    with col2:
        st.metric(
            "Concepts", 
            debug_temporal['network_stats']['concepts'],
            help="Total number of concept neurons created"
        )
    
    with col3:
        st.metric(
            "Connections",
            debug_temporal['updates']['total_connections'],
            help="Total connections between concepts"
        )
    
    with col4:
        avg_strength = debug_temporal['network_stats']['avg_connection_strength']
        st.metric(
            "Avg Strength",
            f"{avg_strength:.6f}",
            help="Average connection strength"
        )

def display_temporal_parameters(debug_info):
    """Display temporal learning parameters."""
    if debug_info['learning_method'] == 'TEMPORAL':
        params = debug_info['temporal_params']
        st.markdown("### 🧪 Temporal Learning Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("τ (tau)", f"{params['tau']:.1f}", help="Exponential decay constant")
        with col2:
            st.metric("η_fwd", f"{params['eta_fwd']:.4f}", help="Forward learning rate")
        with col3:
            st.metric("η_fb", f"{params['eta_fb']:.5f}", help="Feedback learning rate")
        with col4:
            st.metric("Boundary Penalty", f"{params['boundary_penalty']:.2f}", help="Cross-sentence penalty")
        
        # Show sample updates
        if debug_info.get('sample_updates'):
            st.markdown("### 📊 Sample Learning Updates")
            sample_data = []
            for update in debug_info['sample_updates'][:10]:
                sample_data.append({
                    'Connection': update['tokens'],
                    'Lag': update['lag'],
                    'Decay Weight': f"{update['decay_weight']:.6f}",
                    'Final Weight': f"{update['final_weight']:.6f}",
                    'Update Value': f"{update['update']:.6f}"
                })
            
            st.dataframe(pd.DataFrame(sample_data))

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown("""
    # 🧪 Huey Temporal Learning Experiment
    
    **Experimental time-delay Hebbian learning vs traditional windowed approach**
    
    This interface lets you test the new temporal learning method that uses exponential decay 
    to weight connections based on word distance, capturing more nuanced temporal relationships.
    """)
    
    # Sidebar for method selection and parameters
    with st.sidebar:
        st.header("🔬 Experimental Settings")
        
        # Method selection
        use_temporal = st.radio(
            "Learning Method",
            options=[True, False],
            format_func=lambda x: "🕒 Temporal Learning" if x else "📊 Windowed Learning",
            help="Choose between experimental temporal learning or standard windowed method"
        )
        
        if use_temporal:
            st.markdown("### Temporal Parameters")
            
            tau = st.slider(
                "τ (tau) - Decay Constant", 
                min_value=0.5, max_value=10.0, value=3.0, step=0.5,
                help="Controls how quickly distant words are forgotten. Higher = longer memory."
            )
            
            eta_fwd = st.slider(
                "η_fwd - Forward Learning Rate",
                min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f",
                help="Learning rate for A→B connections"
            )
            
            eta_fb = st.slider(
                "η_fb - Feedback Learning Rate",
                min_value=0.0001, max_value=0.01, value=0.002, step=0.0001, format="%.4f",
                help="Learning rate for B→A feedback connections"
            )
            
            boundary_penalty = st.slider(
                "Boundary Penalty",
                min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                help="Reduction factor for learning across sentence boundaries"
            )
        else:
            st.info("Standard windowed learning uses fixed window size and uniform weighting.")
    
    # File upload
    st.header("📁 Upload File for Analysis")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'log', 'dat'],
        help="Upload a conversation file or document for analysis"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process File", type="primary"):
            with st.spinner("Processing file with experimental Huey..."):
                # Create Huey instance
                if use_temporal:
                    huey = create_temporal_huey(True, tau, eta_fwd, eta_fb, boundary_penalty)
                else:
                    huey = create_temporal_huey(False)
                
                # Process file
                result = process_uploaded_file(uploaded_file, huey)
                
                if result and result.get('success'):
                    st.session_state.temporal_huey = huey
                    st.session_state.analysis_results = result
                    st.session_state.processing_complete = True
                    
                    st.success(f"✅ File processed successfully!")
                    st.info(f"Speakers: {result['speakers_registered']}, Exchanges: {result['exchanges_processed']}")
                else:
                    st.error("Failed to process file")
    
    # Analysis results
    if st.session_state.processing_complete and st.session_state.temporal_huey:
        huey = st.session_state.temporal_huey
        
        st.header("📊 Analysis Results")
        
        # Display metrics
        display_comparison_metrics(huey)
        
        # Analysis tabs - Full feature set like original
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Overview",
            "🔍 Concepts",
            "🗺️ 3D Plot", 
            "⚖️ Mass",
            "🧮 Eigenvals",
            "📊 Stats",
            "🧭 Flow",
            "🕐 W/S",
            "🌊 Cascade"
        ])
        
        with tab1:
            st.subheader("Network Overview")
            
            debug = huey.get_debug_summary()
            display_temporal_parameters(debug)
            
            # Network statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Network Structure")
                stats = debug['network_stats']
                st.write(f"**Total Concepts:** {stats['concepts']}")
                st.write(f"**Total Connections:** {debug['updates']['total_connections']}")
                st.write(f"**Average Strength:** {stats['avg_connection_strength']:.6f}")
                st.write(f"**Max Strength:** {stats['max_connection_strength']:.6f}")
            
            with col2:
                st.markdown("### Learning Statistics")
                updates = debug['updates']
                st.write(f"**Nonzero Updates:** {updates['nonzero_updates']}")
                st.write(f"**Zero Updates:** {updates['zero_updates']}")
                if updates['nonzero_updates'] > 0:
                    success_rate = updates['nonzero_updates'] / (updates['nonzero_updates'] + updates['zero_updates']) * 100
                    st.write(f"**Success Rate:** {success_rate:.1f}%")
        
        with tab2:
            st.subheader("3D Concept Visualization")
            
            try:
                coords, eigenvals, labels, eigenvecs = huey.get_3d_coordinates()
                
                if len(coords) > 0:
                    # Create 3D plot
                    fig = go.Figure(data=go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1], 
                        z=coords[:, 2],
                        mode='markers+text',
                        text=labels,
                        textposition="middle center",
                        marker=dict(
                            size=8,
                            color=coords[:, 0],  # Color by first dimension
                            colorscale='Viridis',
                            showscale=True
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                      'X: %{x:.3f}<br>' +
                                      'Y: %{y:.3f}<br>' +
                                      'Z: %{z:.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="3D Concept Space",
                        scene=dict(
                            xaxis_title="Dimension 1",
                            yaxis_title="Dimension 2", 
                            zaxis_title="Dimension 3"
                        ),
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Eigenvalue analysis
                    if len(eigenvals) >= 3:
                        st.markdown("### Eigenvalue Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("λ₁", f"{eigenvals[0]:.6f}")
                        with col2:
                            st.metric("λ₂", f"{eigenvals[1]:.6f}")
                        with col3:
                            st.metric("λ₃", f"{eigenvals[2]:.6f}")
                else:
                    st.warning("Not enough concepts for 3D visualization")
            except Exception as e:
                st.error(f"Visualization error: {e}")
        
        with tab3:
            st.subheader("Learning Process Details")
            
            debug = huey.get_debug_summary()
            
            if debug['learning_method'] == 'TEMPORAL':
                # Show temporal decay curve
                st.markdown("### Temporal Decay Function")
                lags = np.arange(1, 11)
                weights = np.exp(-lags / debug['temporal_params']['tau'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=lags, y=weights,
                    mode='lines+markers',
                    name='exp(-lag/τ)',
                    line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title=f"Exponential Decay (τ={debug['temporal_params']['tau']})",
                    xaxis_title="Word Distance (lag)",
                    yaxis_title="Weight Multiplier",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Learning effectiveness
                st.markdown("### Learning Effectiveness")
                updates = debug['updates']
                total_attempts = updates['nonzero_updates'] + updates['zero_updates']
                
                if total_attempts > 0:
                    success_rate = updates['nonzero_updates'] / total_attempts
                    
                    fig = go.Figure(data=[
                        go.Bar(x=['Successful', 'Failed'], 
                               y=[updates['nonzero_updates'], updates['zero_updates']],
                               marker_color=['green', 'red'])
                    ])
                    
                    fig.update_layout(
                        title=f"Learning Success Rate: {success_rate:.1%}",
                        yaxis_title="Number of Updates"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Concept Analysis")
            
            # Top concepts by connection strength
            if hasattr(huey, 'concept_neurons') and hasattr(huey, 'connections'):
                concept_strengths = {}
                
                for concept, neuron_id in huey.concept_neurons.items():
                    total_strength = 0
                    for conn_key, strength in huey.connections.items():
                        if neuron_id in conn_key:
                            total_strength += strength
                    concept_strengths[concept] = total_strength
                
                # Sort by strength
                sorted_concepts = sorted(concept_strengths.items(), key=lambda x: x[1], reverse=True)
                
                st.markdown("### Top Concepts by Connection Strength")
                concept_data = []
                for i, (concept, strength) in enumerate(sorted_concepts[:20]):
                    concept_data.append({
                        'Rank': i+1,
                        'Concept': concept,
                        'Total Strength': f"{strength:.6f}",
                        'Neuron ID': huey.concept_neurons[concept]
                    })
                
                st.dataframe(pd.DataFrame(concept_data), use_container_width=True)

if __name__ == "__main__":
    main()