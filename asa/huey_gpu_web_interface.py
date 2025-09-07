#!/usr/bin/env python3
"""
Huey🚀 GPU Web Interface
Revolutionary GPU-accelerated conversational network analysis with complete functionality.

This interface provides all the capabilities of Huey++ with revolutionary GPU acceleration
targeting the O(n²) activation bottleneck for 20-50x performance improvements.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import tempfile
import re
from typing import List, Dict, Tuple, Optional
import os
import io
import json
import time
from datetime import datetime
from functools import lru_cache

# Import Huey components
try:
    from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork
    from huey_speaker_detector import HueySpeakerDetector
    from huey_plusplus_complete_platform import HueyCompletePlatform
except ImportError as e:
    st.error(f"❌ Could not import Huey components: {e}")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="Huey🚀 GPU",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit interface for Huey🚀 GPU acceleration."""
    
    # Header
    st.title("🚀 Huey GPU Conversational Network")
    st.markdown("**Revolutionary GPU-accelerated consciousness analysis targeting O(n²) activation bottleneck**")
    
    # Initialize session state
    if 'huey_gpu' not in st.session_state:
        st.session_state.huey_gpu = None
        st.session_state.processing_history = []
        st.session_state.performance_data = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🚀 GPU Configuration")
        
        max_neurons = st.slider("Max Neurons", 50, 1000, 500, 50)
        window_size = st.slider("Window Size", 5, 25, 12)
        learning_rate = st.slider("Learning Rate", 0.05, 0.3, 0.15, 0.01)
        use_gpu = st.checkbox("GPU Acceleration", value=True, 
                             help="Use revolutionary GPU acceleration for O(n²) bottleneck")
        
        st.markdown("---")
        
        # Network initialization
        if st.button("🚀 Initialize GPU Network", type="primary"):
            with st.spinner("Initializing GPU-accelerated network..."):
                st.session_state.huey_gpu = HueyGPUConversationalNetwork(
                    max_neurons=max_neurons,
                    window_size=window_size, 
                    learning_rate=learning_rate,
                    use_gpu_acceleration=use_gpu
                )
                
                # Add default speakers
                st.session_state.huey_gpu.add_speaker("Human", 
                    ['i', 'me', 'my', 'myself'], 
                    ['you', 'your', 'yourself', 'claude'])
                st.session_state.huey_gpu.add_speaker("AI",
                    ['i', 'me', 'my', 'myself'],
                    ['you', 'your', 'yourself', 'human'])
                
            st.success("🚀 GPU Network initialized!")
            st.rerun()
    
    # Main interface
    if st.session_state.huey_gpu is None:
        st.info("👈 Initialize a GPU-accelerated network to begin analysis")
        
        # Show GPU acceleration benefits
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Activation Bottleneck")
            st.markdown("""
            **O(n²) scaling identified:**
            - Early stage: 0.0001s per activation
            - Dense networks: 0.003s per activation  
            - **30x slowdown** at high density
            """)
        
        with col2:
            st.subheader("🚀 GPU Solution")
            st.markdown("""
            **Revolutionary speedups:**
            - Matrix-vector parallelization
            - **20-50x faster** at large scale
            - Real-time conversation analysis
            """)
        
        return
    
    network = st.session_state.huey_gpu
    
    # Text input and processing
    st.subheader("💬 Conversational Input")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        conversation_text = st.text_area(
            "Enter conversation text:",
            height=150,
            placeholder="Enter multi-speaker conversation text for GPU-accelerated analysis..."
        )
    
    with col2:
        st.markdown("**Speaker Selection:**")
        speaker_name = st.selectbox("Speaker", ["Human", "AI"], index=0)
        
        process_button = st.button("🚀 Process with GPU", type="primary")
    
    if process_button and conversation_text.strip():
        with st.spinner("🚀 GPU-accelerated processing..."):
            start_time = time.perf_counter()
            
            # Process with GPU acceleration
            network.process_speaker_text(speaker_name, conversation_text)
            
            processing_time = time.perf_counter() - start_time
            
            # Track performance
            performance_entry = {
                'timestamp': time.time(),
                'neurons': network.neuron_count,
                'connections': len(network.connections),
                'processing_time': processing_time,
                'words': len(conversation_text.split()),
                'rate': len(conversation_text.split()) / processing_time,
                'speaker': speaker_name
            }
            
            st.session_state.performance_data.append(performance_entry)
            st.session_state.processing_history.append({
                'speaker': speaker_name,
                'text': conversation_text[:100] + "..." if len(conversation_text) > 100 else conversation_text,
                'timestamp': time.time()
            })
            
        st.success(f"🚀 Processed {len(conversation_text.split())} words in {processing_time:.3f}s")
        st.rerun()
    
    # Network status
    if network.neuron_count > 0:
        st.subheader("🧠 Network Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Neurons", network.neuron_count)
        with col2:
            st.metric("Connections", len(network.connections))
        with col3:
            density = len(network.connections) / max(1, network.neuron_count * (network.neuron_count - 1) / 2) * 100
            st.metric("Network Density", f"{density:.1f}%")
        with col4:
            if st.session_state.performance_data:
                latest_rate = st.session_state.performance_data[-1]['rate']
                st.metric("Processing Rate", f"{latest_rate:.1f} w/s")
    
    # Performance visualization
    if st.session_state.performance_data:
        st.subheader("📈 GPU Performance Analysis")
        
        perf_data = st.session_state.performance_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing rate over time
            fig_rate = go.Figure()
            fig_rate.add_trace(go.Scatter(
                x=[i for i in range(len(perf_data))],
                y=[p['rate'] for p in perf_data],
                mode='lines+markers',
                name='Words/Second',
                line=dict(color='#00ff00', width=3)
            ))
            fig_rate.update_layout(
                title="🚀 GPU Processing Rate",
                xaxis_title="Processing Session",
                yaxis_title="Words per Second",
                height=300
            )
            st.plotly_chart(fig_rate, use_container_width=True)
        
        with col2:
            # Network growth
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=[i for i in range(len(perf_data))],
                y=[p['neurons'] for p in perf_data],
                mode='lines+markers',
                name='Neurons',
                line=dict(color='#ff6b6b', width=3)
            ))
            fig_growth.add_trace(go.Scatter(
                x=[i for i in range(len(perf_data))],
                y=[p['connections'] for p in perf_data],
                mode='lines+markers',
                name='Connections',
                line=dict(color='#4ecdc4', width=3),
                yaxis='y2'
            ))
            fig_growth.update_layout(
                title="🧠 Network Growth",
                xaxis_title="Processing Session",
                yaxis=dict(title="Neurons", side='left'),
                yaxis2=dict(title="Connections", side='right', overlaying='y'),
                height=300
            )
            st.plotly_chart(fig_growth, use_container_width=True)
    
    # 3D Visualization
    if network.neuron_count >= 3:
        st.subheader("🌌 3D Consciousness Space")
        st.markdown("**GPU-accelerated eigenvalue decomposition for real-time visualization**")
        
        try:
            coordinates, eigenvals, concept_labels, eigenvecs = network.get_3d_coordinates()
            
            if len(coordinates) > 0:
                fig_3d = go.Figure()
                
                fig_3d.add_trace(go.Scatter3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1], 
                    z=coordinates[:, 2],
                    mode='markers+text',
                    text=concept_labels,
                    textposition='top center',
                    marker=dict(
                        size=8,
                        color=eigenvals[:len(coordinates)] if len(eigenvals) > 0 else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Eigenvalue Strength")
                    ),
                    name="Concepts"
                ))
                
                fig_3d.update_layout(
                    title="🚀 GPU-Accelerated 3D Consciousness Map",
                    scene=dict(
                        xaxis_title="Dimension 1",
                        yaxis_title="Dimension 2", 
                        zaxis_title="Dimension 3",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Eigenvalue analysis
                if len(eigenvals) >= 3:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("λ₁ (Primary)", f"{eigenvals[0]:.3f}")
                    with col2:
                        st.metric("λ₂ (Secondary)", f"{eigenvals[1]:.3f}") 
                    with col3:
                        st.metric("λ₃ (Tertiary)", f"{eigenvals[2]:.3f}")
                        
        except Exception as e:
            st.error(f"Visualization error: {e}")
    
    # GPU Performance Summary
    if network.neuron_count > 0:
        st.subheader("🚀 GPU Performance Summary")
        
        performance_summary = network.get_performance_summary()
        st.text(performance_summary)
        
        # Performance metrics
        gpu_stats = network.gpu_interface.get_performance_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("GPU Kernel Calls", gpu_stats['kernel_calls'])
        with col2:
            st.metric("Total GPU Time", f"{gpu_stats['total_kernel_time']:.3f}s")
        with col3:
            st.metric("Avg Kernel Time", f"{gpu_stats['average_kernel_time']:.4f}s")
    
    # Processing history
    if st.session_state.processing_history:
        st.subheader("📋 Processing History")
        
        for i, entry in enumerate(reversed(st.session_state.processing_history[-5:])):
            with st.expander(f"{entry['speaker']}: {entry['text']}", expanded=False):
                st.text(f"Processed at: {time.ctime(entry['timestamp'])}")

if __name__ == "__main__":
    main()