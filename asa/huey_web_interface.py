#!/usr/bin/env python3
"""
🧠 Huey Web Interface
=====================

A user-friendly web interface for the Huey Hebbian Self-Concept Analysis Platform.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.

Run with: streamlit run huey_web_interface.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import tempfile
import os
import io
import json
from datetime import datetime

# Import Huey components
try:
    from huey_complete_platform import HueyCompletePlatform
    from huey_speaker_detector import HueySpeakerDetector
except ImportError as e:
    st.error(f"❌ Could not import Huey components: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🧠 Huey Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin: 1rem 0;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3498db;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'huey' not in st.session_state:
    st.session_state.huey = None
if 'conversation_processed' not in st.session_state:
    st.session_state.conversation_processed = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def initialize_huey(max_neurons, window_size, learning_rate):
    """Initialize Huey platform with given parameters"""
    session_name = f"web_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    huey = HueyCompletePlatform(
        session_name=session_name,
        max_neurons=max_neurons,
        window_size=window_size,
        learning_rate=learning_rate
    )
    return huey

def load_previous_session(filename, huey):
    """Load a previously exported session file"""
    try:
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        # Extract basic session info
        if 'session_data' in session_data:
            session_info = session_data['session_data']
            st.session_state.analysis_results = {
                'success': True,
                'speakers_info': session_info.get('speakers_info', []),
                'conversation_data': session_info.get('conversation_data', []),
                'analysis_results': session_info
            }
            
            # Update huey with loaded session name
            if 'session_name' in session_info:
                huey.session_name = session_info['session_name']
            
            return True
        else:
            st.error("Invalid session file format")
            return False
            
    except Exception as e:
        st.error(f"Error loading session: {e}")
        return False

def process_uploaded_file(uploaded_file, huey):
    """Process uploaded conversation file (TXT or PDF)"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            # Handle PDF files
            try:
                import PyPDF2
                
                # Save PDF to temporary location
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text from PDF
                content = ""
                with open(tmp_file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                
                # Clean up PDF temp file
                os.unlink(tmp_file_path)
                
                # Save extracted text to new temp file for processing
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as txt_file:
                    txt_file.write(content)
                    tmp_file_path = txt_file.name
                    
            except ImportError:
                return {'error': "PyPDF2 is required for PDF processing. Please install with: pip install PyPDF2"}
            except Exception as e:
                return {'error': f"Error processing PDF: {str(e)}"}
        else:
            # Handle TXT files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                content = uploaded_file.getvalue().decode('utf-8')
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

        # Process with speaker detector
        detector = HueySpeakerDetector()
        result = detector.process_conversation_file(tmp_file_path)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        if 'error' in result:
            return {'error': result['error']}
        
        # Register speakers and process conversation
        huey.register_speakers(result['speakers_info'])
        analysis_results = huey.process_conversation(result['conversation_data'])
        
        return {
            'success': True,
            'speakers_info': result['speakers_info'],
            'conversation_data': result['conversation_data'],
            'analysis_results': analysis_results
        }
        
    except Exception as e:
        return {'error': str(e)}

def create_3d_concept_plot(huey, num_concepts, min_mass):
    """Create 3D concept visualization using Plotly"""
    try:
        # Get concepts above threshold
        concepts_data = []
        for neuron_id, word in huey.network.neuron_to_word.items():
            # Calculate concept mass from inertial_mass
            total_mass = 0.0
            if hasattr(huey.network, 'inertial_mass'):
                for (i, j), mass in huey.network.inertial_mass.items():
                    if i == neuron_id or j == neuron_id:
                        total_mass += mass
            
            if total_mass >= min_mass and len(concepts_data) < num_concepts:
                concepts_data.append({
                    'name': word,
                    'mass': total_mass,
                    'id': neuron_id
                })
        
        concepts_data.sort(key=lambda x: x['mass'], reverse=True)
        concepts_data = concepts_data[:num_concepts]
        
        if len(concepts_data) < 3:
            return None
        
        # Create 3D positions (simplified layout)
        np.random.seed(42)
        positions = np.random.randn(len(concepts_data), 3)
        
        # Extract data for plotting
        names = [c['name'] for c in concepts_data]
        masses = [c['mass'] for c in concepts_data]
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=[m/max(masses)*20 + 5 for m in masses],
                color=masses,
                colorscale='Viridis',
                colorbar=dict(title="Concept Mass"),
                opacity=0.8
            ),
            text=names,
            textposition="middle right",
            hovertemplate='<b>%{text}</b><br>Mass: %{marker.color:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Concept Space ({len(concepts_data)} concepts)',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=800,
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 3D plot: {e}")
        return None

def create_mass_comparison_plot(huey, concepts_to_compare):
    """Create mass comparison bar chart"""
    try:
        concept_masses = []
        
        for concept in concepts_to_compare:
            found = False
            for neuron_id, word in huey.network.neuron_to_word.items():
                if word.lower() == concept.lower():
                    # Calculate concept mass from inertial_mass
                    total_mass = 0.0
                    if hasattr(huey.network, 'inertial_mass'):
                        for (i, j), mass in huey.network.inertial_mass.items():
                            if i == neuron_id or j == neuron_id:
                                total_mass += mass
                    
                    concept_masses.append({
                        'concept': concept,
                        'mass': total_mass,
                        'found': True
                    })
                    found = True
                    break
            
            if not found:
                concept_masses.append({
                    'concept': concept,
                    'mass': 0.0,
                    'found': False
                })
        
        # Filter found concepts
        found_concepts = [c for c in concept_masses if c['found']]
        
        if not found_concepts:
            return None, concept_masses
        
        # Create bar chart
        df = pd.DataFrame(found_concepts)
        fig = px.bar(df, x='concept', y='mass', 
                    title='Concept Mass Comparison',
                    labels={'mass': 'Mass', 'concept': 'Concept'})
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig, concept_masses
        
    except Exception as e:
        st.error(f"Error creating mass comparison: {e}")
        return None, []

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Huey: Hebbian Self-Concept Analysis Platform</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Welcome to Huey!</strong> Upload a conversation file and explore how self-concepts emerge 
    through Hebbian learning. No coding required - just upload, configure, and analyze.
    </div>
    """, unsafe_allow_html=True)
    
    # Copyright notice
    st.markdown("""
    <div style="text-align: center; font-size: 0.8em; color: #666; margin: 1rem 0; border-top: 1px solid #ddd; padding-top: 0.5rem;">
    © 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Network parameters
        st.subheader("Network Parameters")
        max_neurons = st.number_input("Max Neurons", min_value=100, max_value=1000, value=500, step=50,
                                     help="Maximum number of concepts the network can learn")
        window_size = st.number_input("Window Size", min_value=3, max_value=15, value=7, step=1,
                                     help="Sliding window size for Hebbian learning")
        learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.15, step=0.01, format="%.2f",
                                       help="How quickly associations form")
        
        # Initialize button
        if st.button("🚀 Initialize Huey", type="primary"):
            with st.spinner("Initializing Huey platform..."):
                st.session_state.huey = initialize_huey(max_neurons, window_size, learning_rate)
                st.session_state.conversation_processed = False
            st.success("✅ Huey initialized!")
            st.rerun()

    # Main content area
    if st.session_state.huey is None:
        st.warning("👈 Please initialize Huey in the sidebar first.")
        return

    # File upload section
    st.markdown('<h2 class="section-header">📁 Load Data</h2>', unsafe_allow_html=True)
    
    # Tab for choosing between new conversation or previous session
    upload_tab, session_tab = st.tabs(["📄 New Conversation", "💾 Previous Session"])
    
    with upload_tab:
        st.subheader("Upload New Conversation File")
        uploaded_file = st.file_uploader(
            "Choose a conversation file",
            type=['txt', 'pdf'],
            help="Upload a .txt or .pdf file containing your conversation. The system will automatically detect speakers."
        )
        
        if uploaded_file is not None and not st.session_state.conversation_processed:
            if st.button("🔍 Process Conversation", type="primary"):
                with st.spinner("Processing conversation file..."):
                    result = process_uploaded_file(uploaded_file, st.session_state.huey)
                    
                    if 'error' in result:
                        st.error(f"❌ Error processing file: {result['error']}")
                    else:
                        st.session_state.analysis_results = result
                        st.session_state.conversation_processed = True
                        st.success("✅ Conversation processed successfully!")
                        st.rerun()
    
    with session_tab:
        st.subheader("Load Previous Analysis Session")
        
        # Find existing session files
        import glob
        session_files = glob.glob("*_complete_data.json")
        
        if session_files:
            selected_file = st.selectbox(
                "Choose a previous session:",
                options=session_files,
                help="Select a previously exported session file to reload"
            )
            
            if st.button("📂 Load Session Data", type="primary"):
                if load_previous_session(selected_file, st.session_state.huey):
                    st.session_state.conversation_processed = True
                    st.success(f"✅ Loaded session: {selected_file}")
                    st.rerun()
                else:
                    st.error("❌ Failed to load session data")
        else:
            st.info("No previous session files found. Export a session first to see it here.")

    # Analysis section (only show if conversation is processed)
    if st.session_state.conversation_processed and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        huey = st.session_state.huey
        
        # Display basic stats
        st.markdown('<h2 class="section-header">📊 Analysis Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Speakers", len(results['speakers_info']))
        
        with col2:
            st.metric("Exchanges", len(results['conversation_data']))
        
        with col3:
            # Get network stats
            stats = huey.query_concepts("network_statistics")
            if 'neuron_stats' in stats:
                st.metric("Total Concepts", stats['neuron_stats']['total_neurons'])
        
        with col4:
            if 'neuron_stats' in stats:
                st.metric("Total Mass", f"{stats['neuron_stats']['total_mass']:.1f}")

        # Tabbed interface for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Concept Associations", 
            "🗺️ 3D Visualization", 
            "⚖️ Mass Comparison", 
            "🧮 Eigenvalue Analysis",
            "📊 Network Stats"
        ])
        
        with tab1:
            st.subheader("Explore Concept Associations")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                concept_name = st.text_input("Enter concept to explore:", value="i")
            with col2:
                top_n = st.number_input("Number of associations:", min_value=5, max_value=50, value=15, step=1)
            
            if st.button("🔍 Analyze Associations"):
                result = huey.query_concepts("strongest_associations", concept=concept_name, top_n=top_n)
                
                if 'associations' in result:
                    st.success(f"Found {len(result['associations'])} associations for '{concept_name}'")
                    
                    # Create dataframe for display
                    df = pd.DataFrame(result['associations'])
                    df['rank'] = range(1, len(df) + 1)
                    df = df[['rank', 'concept', 'strength']]
                    
                    # Display as table
                    st.dataframe(df, use_container_width=True)
                    
                    # Create bar chart
                    fig = px.bar(df.head(10), x='concept', y='strength',
                               title=f"Top 10 Associations for '{concept_name}'")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Concept '{concept_name}' not found in network")
        
        with tab2:
            st.subheader("3D Concept Space Visualization")
            
            col1, col2 = st.columns(2)
            with col1:
                num_concepts_3d = st.number_input("Number of concepts to plot", min_value=10, max_value=200, value=50, step=5)
            with col2:
                min_mass_3d = st.number_input("Minimum mass threshold", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
            
            if st.button("🗺️ Generate 3D Visualization"):
                with st.spinner("Creating 3D visualization..."):
                    fig = create_3d_concept_plot(huey, num_concepts_3d, min_mass_3d)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Not enough concepts with mass ≥ {min_mass_3d} for visualization")
        
        with tab3:
            st.subheader("Concept Mass Comparison")
            
            # Input for concepts to compare
            concepts_input = st.text_area(
                "Enter concepts to compare (one per line):",
                value="i\nme\nmyself\nspeaker_trump\nspeaker_times"
            )
            
            if st.button("⚖️ Compare Masses"):
                concepts_list = [c.strip() for c in concepts_input.split('\n') if c.strip()]
                
                if concepts_list:
                    fig, results_data = create_mass_comparison_plot(huey, concepts_list)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    for data in results_data:
                        status = "✅ Found" if data['found'] else "❌ Not Found"
                        mass_str = f"{data['mass']:.3f}" if data['found'] else "N/A"
                        st.write(f"**{data['concept']}**: {mass_str} {status}")
                else:
                    st.error("Please enter at least one concept to compare")
        
        with tab4:
            st.subheader("Eigenvalue Analysis")
            
            min_mass_eigen = st.number_input("Minimum mass for eigenvalue analysis", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f")
            
            if st.button("🧮 Analyze Eigenvalues"):
                with st.spinner("Computing eigenvalues..."):
                    try:
                        # Get concepts above threshold
                        concepts = []
                        for neuron_id, word in huey.network.neuron_to_word.items():
                            # Calculate concept mass from inertial_mass
                            total_mass = 0.0
                            if hasattr(huey.network, 'inertial_mass'):
                                for (i, j), mass in huey.network.inertial_mass.items():
                                    if i == neuron_id or j == neuron_id:
                                        total_mass += mass
                            
                            if total_mass >= min_mass_eigen:
                                concepts.append({
                                    'id': neuron_id,
                                    'concept': word,
                                    'mass': total_mass
                                })
                        
                        if len(concepts) >= 2:
                            # Build association matrix
                            n = len(concepts)
                            association_matrix = np.zeros((n, n))
                            
                            for i, concept_i in enumerate(concepts):
                                for j, concept_j in enumerate(concepts):
                                    # Get connection strength from inertial_mass
                                    strength = 0.0
                                    neuron_i_id = concept_i['id']
                                    neuron_j_id = concept_j['id']
                                    
                                    if i != j and hasattr(huey.network, 'inertial_mass'):
                                        # Check both directions for off-diagonal elements
                                        if (neuron_i_id, neuron_j_id) in huey.network.inertial_mass:
                                            strength = huey.network.inertial_mass[(neuron_i_id, neuron_j_id)]
                                        elif (neuron_j_id, neuron_i_id) in huey.network.inertial_mass:
                                            strength = huey.network.inertial_mass[(neuron_j_id, neuron_i_id)]
                                    
                                    # Set matrix element (diagonal will remain 0, allowing negative eigenvalues)
                                    association_matrix[i, j] = strength
                            
                            # Compute eigenvalues
                            from scipy.linalg import eigvals
                            eigenvalues = eigvals(association_matrix)
                            
                            # Extract real parts of eigenvalues (even if some are complex numbers)
                            real_eigenvalues = [eig.real if hasattr(eig, 'real') else float(eig) for eig in eigenvalues]
                            
                            # Classify by sign (triangle inequality framework)
                            positive_eigs = [eig for eig in real_eigenvalues if eig > 1e-10]  # Real dimensions
                            negative_eigs = [eig for eig in real_eigenvalues if eig < -1e-10]  # Imaginary dimensions
                            zero_eigs = [eig for eig in real_eigenvalues if abs(eig) <= 1e-10]  # Near-zero dimensions
                            
                            # Display results
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Dimensions", len(real_eigenvalues))
                            with col2:
                                st.metric("Real Dimensions", len(positive_eigs))
                            with col3:
                                st.metric("Imaginary Dimensions", len(negative_eigs))
                            with col4:
                                st.metric("Zero Dimensions", len(zero_eigs))
                            
                            # Eigenvalue statistics and visualization
                            st.subheader("Eigenvalue Statistics")
                            
                            if positive_eigs:
                                st.markdown("**Positive Eigenvalues (Real Dimensions)**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Largest", f"{max(positive_eigs):.3f}")
                                with col2:
                                    st.metric("Smallest", f"{min(positive_eigs):.3f}")
                                with col3:
                                    st.metric("Mean", f"{np.mean(positive_eigs):.3f}")
                                with col4:
                                    st.metric("Std Dev", f"{np.std(positive_eigs):.3f}")
                            
                            if negative_eigs:
                                st.markdown("**Negative Eigenvalues (Imaginary Dimensions)**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Largest", f"{max(negative_eigs):.3f}")
                                with col2:
                                    st.metric("Smallest", f"{min(negative_eigs):.3f}")
                                with col3:
                                    st.metric("Mean", f"{np.mean(negative_eigs):.3f}")
                                with col4:
                                    st.metric("Std Dev", f"{np.std(negative_eigs):.3f}")
                            
                            # Combined eigenvalue plot
                            if real_eigenvalues:
                                fig = px.histogram(x=real_eigenvalues, nbins=min(20, len(real_eigenvalues)),
                                                 title="Eigenvalue Distribution (All Dimensions)")
                                fig.add_vline(x=0, line_dash="dash", line_color="red", 
                                            annotation_text="Zero Line")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Triangle inequality analysis
                            if negative_eigs:
                                st.info(f"🔍 **Triangle Inequality Violation Detected**: Your data contains {len(negative_eigs)} imaginary dimensions, indicating non-Euclidean (pseudo-Riemannian) concept space.")
                            else:
                                st.success("✅ **Euclidean Space**: All eigenvalues are non-negative, indicating your concept space follows Euclidean geometry.")
                        else:
                            st.error(f"Need at least 2 concepts with mass ≥ {min_mass_eigen}")
                            
                    except Exception as e:
                        st.error(f"Error in eigenvalue analysis: {e}")
        
        with tab5:
            st.subheader("Detailed Network Statistics")
            
            if 'neuron_stats' in stats:
                # Network overview
                st.markdown("### Network Overview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Total Neurons:** {stats['neuron_stats']['total_neurons']}")
                    st.write(f"**Active Neurons:** {stats['neuron_stats']['active_neurons']}")
                    st.write(f"**Total Mass:** {stats['neuron_stats']['total_mass']:.3f}")
                    st.write(f"**Average Mass:** {stats['neuron_stats']['average_mass']:.3f}")
                
                with col2:
                    st.write(f"**Total Connections:** {stats['connection_stats']['total_connections']}")
                    st.write(f"**Strong Connections:** {stats['connection_stats']['strong_connections']}")
                    st.write(f"**Average Strength:** {stats['connection_stats']['average_strength']:.3f}")
                    st.write(f"**Max Strength:** {stats['connection_stats']['max_strength']:.3f}")
                
                # Speaker statistics
                if 'speaker_stats' in stats:
                    st.markdown("### Speaker Activity")
                    for speaker, speaker_stats in stats['speaker_stats'].items():
                        blocks = speaker_stats.get('blocks_processed', 0)
                        st.write(f"**{speaker}:** {blocks} blocks processed")
        
        # Export section
        st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Generate Analysis Report"):
                with st.spinner("Generating comprehensive report..."):
                    report = huey.create_analysis_report(include_visualizations=True)
                    st.download_button(
                        label="⬇️ Download Report",
                        data=report,
                        file_name=f"{huey.session_name}_analysis_report.md",
                        mime="text/markdown"
                    )
        
        with col2:
            if st.button("💾 Export Session Data"):
                with st.spinner("Exporting session data..."):
                    filename = huey.export_session_data()
                    with open(filename, 'r') as f:
                        data = f.read()
                    st.download_button(
                        label="⬇️ Download Session Data",
                        data=data,
                        file_name=f"{huey.session_name}_complete_data.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()