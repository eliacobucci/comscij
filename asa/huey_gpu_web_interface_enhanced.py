#!/usr/bin/env python3
"""
Huey🚀 GPU Web Interface - Enhanced with File Upload
Revolutionary GPU-accelerated conversational network analysis with file upload capability.
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
    page_title="Huey🚀 GPU Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Enhanced Streamlit interface for Huey🚀 GPU acceleration with file upload."""
    
    # Header
    st.title("🚀 Huey GPU Enhanced - File Upload Ready")
    st.markdown("**Revolutionary GPU-accelerated consciousness analysis with file processing capabilities**")
    
    # Initialize session state
    if 'huey_gpu' not in st.session_state:
        st.session_state.huey_gpu = None
        st.session_state.processing_history = []
        st.session_state.performance_data = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🚀 GPU Configuration")
        
        max_neurons = st.slider("Max Neurons", 50, 2000, 1000, 50)
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
                st.session_state.huey_gpu.add_speaker("Feynman",
                    ['i', 'me', 'my', 'myself'],
                    ['you', 'your', 'yourself'])
                st.session_state.huey_gpu.add_speaker("Interviewer",
                    ['i', 'me', 'my', 'myself'],
                    ['you', 'your', 'yourself'])
                
            st.success("🚀 GPU Network initialized!")
            st.rerun()
        
        # Performance stats
        if st.session_state.huey_gpu is not None:
            st.markdown("---")
            st.subheader("⚡ GPU Stats")
            gpu_stats = st.session_state.huey_gpu.gpu_interface.get_performance_stats()
            st.metric("GPU Calls", gpu_stats['kernel_calls'])
            st.metric("GPU Time", f"{gpu_stats['total_kernel_time']:.3f}s")
            st.metric("GPU Status", "🚀 ACTIVE" if gpu_stats['gpu_enabled'] else "💻 CPU")
    
    # Main interface
    if st.session_state.huey_gpu is None:
        st.info("👈 Initialize a GPU-accelerated network to begin analysis")
        
        # Show GPU acceleration benefits
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 O(n²) Bottleneck Solved")
            st.markdown("""
            **Previous issues:**
            - JAX Metal compatibility broken
            - CPU-only processing was 20-50x slower
            - Large files took minutes instead of seconds
            """)
        
        with col2:
            st.subheader("🚀 PyTorch MPS Solution")
            st.markdown("""
            **Now working:**
            - Native Apple Silicon acceleration
            - 20-50x speedup restored
            - Real-time conversation analysis
            """)
        
        return
    
    network = st.session_state.huey_gpu
    
    # Enhanced Input Section
    st.subheader("📁 Input Methods")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Text", "📁 File Upload", "🎯 Quick Tests"])
    
    with tab1:
        st.markdown("**Type or paste conversation text:**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            conversation_text = st.text_area(
                "Enter conversation text:",
                height=200,
                placeholder="Enter multi-speaker conversation text for GPU-accelerated analysis...",
                key="manual_text"
            )
        
        with col2:
            st.markdown("**Speaker:**")
            speaker_name = st.selectbox("Speaker", ["Human", "AI", "Feynman", "Interviewer"], index=0, key="manual_speaker")
            
            if st.button("🚀 Process Manual Text", type="primary", key="process_manual"):
                if conversation_text.strip():
                    process_text(conversation_text, speaker_name, network)
    
    with tab2:
        st.markdown("**Upload text files (Feynman conversations, transcripts, etc.):**")
        
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'md', 'csv'],
            help="Upload .txt, .md, or .csv files containing conversation data"
        )
        
        if uploaded_file is not None:
            # Read the uploaded file
            try:
                file_content = uploaded_file.read().decode('utf-8')
                st.success(f"✅ Loaded: {uploaded_file.name} ({len(file_content.split())} words)")
                
                # Show preview
                with st.expander("📄 File Preview (first 500 characters)"):
                    st.text(file_content[:500] + ("..." if len(file_content) > 500 else ""))
                
                # Processing options
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    file_speaker = st.selectbox("Assign to speaker:", 
                                              ["Human", "AI", "Feynman", "Interviewer"], 
                                              index=2, key="file_speaker")  # Default to Feynman
                
                with col2:
                    if st.button("🚀 Process File", type="primary", key="process_file"):
                        process_text(file_content, file_speaker, network)
                        
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab3:
        st.markdown("**Quick performance tests:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Small Test (Physics)", key="test_small"):
                test_text = """
                Physics is the fundamental science that seeks to understand the universe. 
                It studies matter, energy, and their interactions. From quantum mechanics 
                to relativity, physics provides the foundation for all other sciences.
                """
                process_text(test_text, "Feynman", network)
        
        with col2:
            if st.button("🚀 Large Test (Feynman Sample)", key="test_large"):
                test_text = """
                Richard Feynman was one of the most influential physicists of the 20th century. 
                He made fundamental contributions to quantum mechanics, quantum electrodynamics, 
                and particle physics. His approach to physics was characterized by intuitive 
                understanding rather than formal mathematical proofs. He believed that if you 
                can't explain something simply, you don't understand it well enough. This 
                philosophy shaped his teaching and research methods throughout his career.
                
                Feynman's work on quantum electrodynamics earned him the Nobel Prize in Physics 
                in 1965, which he shared with Julian Schwinger and Sin-Itiro Tomonaga. His 
                path integral formulation of quantum mechanics provided a new way to understand 
                quantum phenomena. The Feynman diagrams he developed became an essential tool 
                for theoretical physicists to visualize and calculate particle interactions.
                
                Beyond his scientific achievements, Feynman was known for his curiosity about 
                the world and his ability to explain complex concepts in simple terms. He was 
                an excellent teacher and communicator, making physics accessible to general 
                audiences through his lectures and books.
                """
                process_text(test_text, "Feynman", network)
    
    # Network Status and Performance
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
    
    # Processing history
    if st.session_state.processing_history:
        st.subheader("📋 Processing History")
        
        for i, entry in enumerate(reversed(st.session_state.processing_history[-5:])):
            with st.expander(f"{entry['speaker']}: {entry['text'][:50]}...", expanded=False):
                st.text(f"Processed at: {time.ctime(entry['timestamp'])}")
                if i < len(st.session_state.performance_data):
                    perf = st.session_state.performance_data[-(i+1)]
                    st.text(f"Processing time: {perf['processing_time']:.3f}s")
                    st.text(f"Words processed: {perf['words']}")
                    st.text(f"Rate: {perf['rate']:.1f} words/second")

def process_text(text: str, speaker: str, network):
    """Process text with GPU acceleration and update interface."""
    if not text.strip():
        st.warning("Please provide some text to process.")
        return
    
    with st.spinner(f"🚀 GPU-accelerated processing ({len(text.split())} words)..."):
        start_time = time.perf_counter()
        
        # Process with GPU acceleration
        try:
            network.process_speaker_text(speaker, text)
            processing_time = time.perf_counter() - start_time
            
            # Track performance
            performance_entry = {
                'timestamp': time.time(),
                'neurons': network.neuron_count,
                'connections': len(network.connections),
                'processing_time': processing_time,
                'words': len(text.split()),
                'rate': len(text.split()) / processing_time,
                'speaker': speaker
            }
            
            st.session_state.performance_data.append(performance_entry)
            st.session_state.processing_history.append({
                'speaker': speaker,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'timestamp': time.time()
            })
            
            # Success message with performance info
            st.success(f"🚀 Processed {len(text.split())} words in {processing_time:.3f}s ({performance_entry['rate']:.1f} w/s)")
            
            # Get GPU stats
            gpu_stats = network.gpu_interface.get_performance_stats()
            if gpu_stats['gpu_enabled']:
                st.info(f"⚡ GPU: {gpu_stats['kernel_calls']} calls, {gpu_stats['total_kernel_time']:.3f}s total")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()