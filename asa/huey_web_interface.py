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
import time
from datetime import datetime
from functools import lru_cache

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
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}
if 'last_cache_clear' not in st.session_state:
    st.session_state.last_cache_clear = time.time()

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

def process_uploaded_file(uploaded_file, huey, timeout_hours=2.0, exchange_limit=10000):
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
        
        # Register speakers and process conversation with monitoring
        huey.register_speakers(result['speakers_info'])
        
        # Add progress monitoring
        import time
        from datetime import datetime
        
        total_exchanges = len(result['conversation_data'])
        start_time = time.time()
        st.write(f"🎯 **PROCESSING MONITOR**")
        st.write(f"   📊 Total exchanges to process: {total_exchanges}")
        st.write(f"   🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
        st.write(f"   ⚡ Processing with progress tracking...")
        
        # Create progress tracking containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_estimate = st.empty()
        
        # Process with configurable timeout and stall detection  
        MAX_PROCESSING_TIME = timeout_hours * 3600
        STALL_CHECK_INTERVAL = 300   # Check every 5 minutes
        
        # Limit exchanges if specified
        if exchange_limit > 0:
            result['conversation_data'] = result['conversation_data'][:exchange_limit]
            st.warning(f"⚠️ Limited processing to first {exchange_limit} exchanges")
        
        try:
            # Process conversation with monitoring using manual loop instead of bulk processing
            total_exchanges = len(result['conversation_data'])
            
            # Process each exchange individually with progress updates
            for i, (speaker_id, text) in enumerate(result['conversation_data']):
                # Update progress
                progress = (i + 1) / total_exchanges
                progress_bar.progress(progress)
                
                # Update status with enhanced milestone reporting
                elapsed_time = time.time() - start_time
                exchanges_per_sec = (i + 1) / max(elapsed_time, 1)
                remaining_exchanges = total_exchanges - (i + 1)
                est_remaining_time = remaining_exchanges / max(exchanges_per_sec, 0.1)
                
                # Enhanced milestone notifications
                progress_pct = int(progress * 100)
                if (i + 1) % max(1, total_exchanges // 20) == 0:  # Every 5% or major milestones
                    if progress_pct in [10, 25, 50, 75, 90]:
                        time_estimate.success(f"🎯 **{progress_pct}% Milestone** - Excellent progress!")
                
                # Simple progress display without warp factor
                warp_display = ""
                
                status_text.text(f"Processing exchange {i+1:,}/{total_exchanges:,} ({progress_pct}%) | "
                               f"Rate: {exchanges_per_sec:.1f}/sec | "
                               f"{warp_display} | "
                               f"ETA: {est_remaining_time/60:.1f}min")
                
                time_estimate.text(f"⏱️ Elapsed: {elapsed_time/60:.1f}min | "
                                 f"Estimated remaining: {est_remaining_time/60:.1f}min")
                
                # Check timeout
                if elapsed_time > MAX_PROCESSING_TIME:
                    st.warning(f"⚠️ TIMEOUT: Processing exceeded {timeout_hours:.1f} hours at exchange {i+1}")
                    st.info(f"💾 Saving progress: {i+1} exchanges processed successfully")
                    # Generate analysis from partial results and return success with partial data
                    analysis_results = huey._generate_comprehensive_analysis()
                    return {
                        'success': True,
                        'partial_results': True,
                        'exchanges_processed': i+1,
                        'total_exchanges': total_exchanges,
                        'timeout_reason': f'Exceeded {timeout_hours:.1f} hour limit',
                        'analysis': analysis_results,
                        'huey': huey
                    }
                
                # Check exchange limit
                if i >= exchange_limit:
                    st.warning(f"⚠️ STOPPED: Reached exchange limit of {exchange_limit}")
                    break
                
                # Process this exchange
                try:
                    huey.network.process_speaker_text(speaker_id, text)
                except Exception as e:
                    st.error(f"❌ Error processing exchange {i+1}: {str(e)}")
                    return {'error': f'Processing failed at exchange {i+1}: {str(e)}'}
                
                # Update display every 10 exchanges to avoid too much UI updating
                if (i + 1) % 10 == 0:
                    time.sleep(0.01)  # Brief pause to allow UI updates
            
            # Generate final analysis
            analysis_results = huey._generate_comprehensive_analysis()
            
            # Check if we exceeded timeout during processing
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_PROCESSING_TIME:
                st.error(f"⚠️ TIMEOUT: Processing exceeded {timeout_hours:.1f} hours.")
                return {'error': f'Processing timeout after {timeout_hours:.1f} hours'}
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            import traceback
            error_details = traceback.format_exc()
            st.error(f"❌ Processing failed after {elapsed_time/60:.1f} minutes: {str(e)}")
            st.code(error_details, language="python")  # Show full error for debugging
            return {'error': f'Processing failed: {str(e)}'}
        
        # Update final status
        elapsed_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.success(f"✅ Processing complete! Processed {total_exchanges} exchanges in {elapsed_time/60:.1f} minutes")
        time_estimate.info(f"⏱️ Average rate: {total_exchanges/(elapsed_time/60):.1f} exchanges/minute")
        
        return {
            'success': True,
            'speakers_info': result['speakers_info'],
            'conversation_data': result['conversation_data'],
            'analysis_results': analysis_results
        }
        
    except Exception as e:
        return {'error': str(e)}

def analyze_concept_flow(network, concept, threshold=0.1):
    """
    Analyze directional flow patterns for a specific concept using left/right eigenvectors.
    
    Args:
        network: Huey network instance
        concept: The concept to analyze
        threshold: Minimum connection strength threshold
        
    Returns:
        Dictionary with outgoing/incoming flows and asymmetry metrics
    """
    try:
        if concept not in network.word_to_neuron:
            return None
            
        concept_id = network.word_to_neuron[concept]
        
        # Build asymmetric connection matrix
        num_neurons = len(network.neuron_to_word)
        outgoing_matrix = np.zeros((num_neurons, num_neurons))
        incoming_matrix = np.zeros((num_neurons, num_neurons))
        
        # Create index mapping
        neuron_ids = list(network.neuron_to_word.keys())
        id_to_index = {nid: i for i, nid in enumerate(neuron_ids)}
        
        if concept_id not in id_to_index:
            return None
            
        concept_idx = id_to_index[concept_id]
        
        # Fill matrices with directional connections
        for (i, j), strength in network.connections.items():
            if strength >= threshold and i in id_to_index and j in id_to_index:
                idx_i, idx_j = id_to_index[i], id_to_index[j]
                outgoing_matrix[idx_i][idx_j] = strength  # i → j
                incoming_matrix[idx_j][idx_i] = strength  # j ← i
        
        # Analyze outgoing influences (right eigenvector)
        outgoing_influences = []
        for j in range(num_neurons):
            if j != concept_idx:
                strength = outgoing_matrix[concept_idx][j]
                if strength >= threshold:
                    target_id = neuron_ids[j]
                    target_word = network.neuron_to_word[target_id]
                    outgoing_influences.append((target_word, strength))
        
        # Analyze incoming influences (left eigenvector)
        incoming_influences = []
        for i in range(num_neurons):
            if i != concept_idx:
                strength = incoming_matrix[concept_idx][i]
                if strength >= threshold:
                    source_id = neuron_ids[i]
                    source_word = network.neuron_to_word[source_id]
                    incoming_influences.append((source_word, strength))
        
        # Sort by strength
        outgoing_influences.sort(key=lambda x: x[1], reverse=True)
        incoming_influences.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate asymmetry score
        total_outgoing = sum(strength for _, strength in outgoing_influences)
        total_incoming = sum(strength for _, strength in incoming_influences)
        total_flow = total_outgoing + total_incoming
        
        asymmetry_score = 0
        if total_flow > 0:
            asymmetry_score = abs(total_outgoing - total_incoming) / total_flow
        
        return {
            'concept': concept,
            'outgoing': outgoing_influences,
            'incoming': incoming_influences,
            'asymmetry_score': asymmetry_score,
            'total_outgoing': total_outgoing,
            'total_incoming': total_incoming
        }
        
    except Exception as e:
        print(f"Error in analyze_concept_flow: {e}")
        return None

def analyze_network_asymmetry(network, threshold=0.1):
    """
    Analyze network-wide asymmetry patterns and flow efficiency.
    
    Args:
        network: Huey network instance
        threshold: Minimum connection strength threshold
        
    Returns:
        Dictionary with network asymmetry metrics
    """
    try:
        # Build directional connection matrix
        neuron_ids = list(network.neuron_to_word.keys())
        num_neurons = len(neuron_ids)
        id_to_index = {nid: i for i, nid in enumerate(neuron_ids)}
        
        forward_matrix = np.zeros((num_neurons, num_neurons))
        backward_matrix = np.zeros((num_neurons, num_neurons))
        
        # Fill matrices
        strong_connections = 0
        total_connections = 0
        asymmetric_connections = 0
        
        for (i, j), strength in network.connections.items():
            if i in id_to_index and j in id_to_index:
                idx_i, idx_j = id_to_index[i], id_to_index[j]
                forward_matrix[idx_i][idx_j] = strength
                backward_matrix[idx_j][idx_i] = strength
                
                if strength >= threshold:
                    strong_connections += 1
                    # Check for asymmetry
                    reverse_key = (j, i)
                    reverse_strength = network.connections.get(reverse_key, 0)
                    if abs(strength - reverse_strength) > 0.1:
                        asymmetric_connections += 1
                
                total_connections += 1
        
        # Calculate overall asymmetry
        asymmetry_matrix = np.abs(forward_matrix - backward_matrix)
        total_strength = np.sum(forward_matrix + backward_matrix)
        overall_asymmetry = np.sum(asymmetry_matrix) / max(total_strength, 1e-6)
        
        # Calculate flow efficiency (how well information flows)
        # Based on spectral properties of the adjacency matrix
        try:
            eigenvalues = np.linalg.eigvals(forward_matrix)
            real_eigenvals = np.real(eigenvalues)
            flow_efficiency = np.max(real_eigenvals) / max(np.mean(real_eigenvals), 1e-6)
            flow_efficiency = min(flow_efficiency, 10.0)  # Cap at reasonable value
        except:
            flow_efficiency = 0.0
        
        # Find most asymmetric concepts
        concept_asymmetries = []
        for i, neuron_id in enumerate(neuron_ids):
            if neuron_id in network.neuron_to_word:
                word = network.neuron_to_word[neuron_id]
                
                # Calculate outgoing vs incoming asymmetry
                outgoing_strength = np.sum(forward_matrix[i, :])
                incoming_strength = np.sum(backward_matrix[i, :])
                total_strength = outgoing_strength + incoming_strength
                
                if total_strength > threshold:
                    asymmetry = abs(outgoing_strength - incoming_strength) / total_strength
                    flow_type = "outgoing" if outgoing_strength > incoming_strength else "incoming"
                    if asymmetry > 0.1:  # Only include significantly asymmetric concepts
                        concept_asymmetries.append((word, asymmetry, flow_type))
        
        # Sort by asymmetry score
        concept_asymmetries.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'overall_asymmetry': overall_asymmetry,
            'strong_asymmetric_connections': asymmetric_connections,
            'flow_efficiency': flow_efficiency,
            'total_connections': total_connections,
            'strong_connections': strong_connections,
            'top_asymmetric_concepts': concept_asymmetries[:20]
        }
        
    except Exception as e:
        print(f"Error in analyze_network_asymmetry: {e}")
        return None

def create_3d_concept_plot(huey, num_concepts, min_mass, connection_direction="both", eigenvector_type="right"):
    """
    Create 3D concept visualization using directional eigenvector coordinates
    
    Args:
        huey: Huey network instance
        num_concepts: Number of top concepts to visualize
        min_mass: Minimum mass threshold for concepts
        connection_direction: "outgoing", "incoming", "both" (default)
        eigenvector_type: "right", "left", "both" (default: "right")
    """
    st.write(f"🔥 FUNCTION CALLED: connection_direction='{connection_direction}', eigenvector_type='{eigenvector_type}'")
    try:
        # System artifacts to filter out
        system_artifacts = {
            'speaker_speaker_a', 'speaker_speaker_b', 'speaker_speaker_c', 
            'speaker_speaker_d', 'speaker_speaker_e', 'speaker_speaker_f',
            're', 'e', 'g', '4', 'lines'
        }
        
        # Get all concepts and their masses, filtering out system artifacts
        all_concepts = []
        for neuron_id, word in huey.network.neuron_to_word.items():
            # Skip system artifacts
            if word.lower() in system_artifacts:
                continue
                
            # Calculate concept mass from inertial_mass
            total_mass = 0.0
            if hasattr(huey.network, 'inertial_mass'):
                for (i, j), mass in huey.network.inertial_mass.items():
                    if i == neuron_id or j == neuron_id:
                        total_mass += mass
            
            all_concepts.append({
                'name': word,
                'mass': total_mass,
                'id': neuron_id
            })
        
        # Sort by mass and take top concepts
        all_concepts.sort(key=lambda x: x['mass'], reverse=True)
        concepts_data = all_concepts[:num_concepts]
        
        if len(concepts_data) < 3:
            st.error(f"Only found {len(concepts_data)} concepts total")
            return None, None, None
            
        # Build association matrix for selected concepts
        n_concepts = len(concepts_data)
        association_matrix = np.zeros((n_concepts, n_concepts))
        concept_ids = [c['id'] for c in concepts_data]
        id_to_index = {concept_id: i for i, concept_id in enumerate(concept_ids)}
        
        # Fill association matrix with DIRECTIONAL connection strengths
        connection_source = None
        if hasattr(huey.network, 'synaptic_strengths'):
            connection_source = 'synaptic_strengths'
            for (i, j), strength in huey.network.synaptic_strengths.items():
                if i in id_to_index and j in id_to_index:
                    idx_i, idx_j = id_to_index[i], id_to_index[j]
                    
                    # Directional connections: i → j (temporal/causal flow)
                    if connection_direction in ["outgoing", "both"]:
                        association_matrix[idx_i][idx_j] = strength
                    if connection_direction in ["incoming", "both"]:
                        association_matrix[idx_j][idx_i] = strength
                        
        elif hasattr(huey.network, 'inertial_mass'):
            connection_source = 'inertial_mass'
            max_mass = max(huey.network.inertial_mass.values()) if huey.network.inertial_mass else 1.0
            for (i, j), mass in huey.network.inertial_mass.items():
                if i in id_to_index and j in id_to_index:
                    idx_i, idx_j = id_to_index[i], id_to_index[j]
                    strength = mass / max_mass  # Normalize by actual max
                    
                    # Directional connections: i → j (temporal/causal flow from Hebbian learning)
                    if connection_direction in ["outgoing", "both"]:
                        association_matrix[idx_i][idx_j] = strength
                    if connection_direction in ["incoming", "both"]:
                        association_matrix[idx_j][idx_i] = strength
        
        # DIRECTIONAL EIGENVECTOR ANALYSIS
        try:
            n = association_matrix.shape[0]
            
            # For asymmetric matrices, we need different approaches
            if connection_direction == "both":
                # Symmetric case: use traditional double centering (Torgerson)
                max_sim = np.max(association_matrix) if np.max(association_matrix) > 0 else 1.0
                distance_matrix = max_sim - association_matrix
                D_squared = distance_matrix ** 2
                
                ones_matrix = np.ones((n, n)) / n
                H = np.eye(n) - ones_matrix
                B = -0.5 * H @ D_squared @ H
                
                eigenvalues, eigenvectors = np.linalg.eigh(B)
                
                # Sort by eigenvalue (descending)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                eigenvector_analysis = "symmetric"
                
            else:
                # Asymmetric case: Proper pseudo-Riemannian eigenvalue extraction
                if eigenvector_type == "right":
                    print(f"🔥 ENTERING RIGHT EIGENVECTOR CASE")
                    # RIGHT: Build upper triangular directly from asymmetric network data (i -> j)
                    right_matrix = np.zeros((n_concepts, n_concepts))
                    if hasattr(huey.network, 'inertial_mass'):
                        max_mass = max(huey.network.inertial_mass.values()) if huey.network.inertial_mass else 1.0
                        for (i, j), mass in huey.network.inertial_mass.items():
                            if i in id_to_index and j in id_to_index:
                                idx_i, idx_j = id_to_index[i], id_to_index[j]
                                # Only populate upper triangle: i -> j (outgoing connections)
                                if idx_i <= idx_j:  # Upper triangular condition
                                    right_matrix[idx_i][idx_j] = mass / max_mass
                    
                    print(f"DEBUG RIGHT - Direct from network data:")
                    print(f"   Non-zero elements: {np.count_nonzero(right_matrix)}")
                    print(f"   Sum: {np.sum(right_matrix):.3f}")
                    print(f"   Matrix sample (first 3x3): \n{right_matrix[:3,:3]}")
                    
                    # Symmetrize for eigenvalue analysis
                    sym_matrix = right_matrix + right_matrix.T
                    print(f"   After symmetrization - Sum: {np.sum(sym_matrix):.3f}")
                    print(f"   Symmetrized sample (first 3x3): \n{sym_matrix[:3,:3]}")
                    
                    # Apply double-centering (Torgerson procedure) to the symmetrized matrix
                    n = sym_matrix.shape[0]
                    max_sim = np.max(sym_matrix) if np.max(sym_matrix) > 0 else 1.0
                    distance_matrix = max_sim - sym_matrix
                    D_squared = distance_matrix ** 2
                    
                    ones_matrix = np.ones((n, n)) / n
                    H = np.eye(n) - ones_matrix
                    B = -0.5 * H @ D_squared @ H
                    
                    eigenvalues, eigenvectors = np.linalg.eigh(B)
                    print(f"   After double-centering - eigenvalue range: {np.min(eigenvalues):.3f} to {np.max(eigenvalues):.3f}")
                    eigenvector_analysis = f"right_from_network_data"
                    
                elif eigenvector_type == "left":
                    print(f"🔥 ENTERING LEFT EIGENVECTOR CASE")
                    # LEFT: Build lower triangular directly from asymmetric network data (j -> i)
                    left_matrix = np.zeros((n_concepts, n_concepts))
                    if hasattr(huey.network, 'inertial_mass'):
                        max_mass = max(huey.network.inertial_mass.values()) if huey.network.inertial_mass else 1.0
                        for (i, j), mass in huey.network.inertial_mass.items():
                            if i in id_to_index and j in id_to_index:
                                idx_i, idx_j = id_to_index[i], id_to_index[j]
                                # Only populate lower triangle: j -> i (incoming connections)
                                if idx_j <= idx_i:  # Lower triangular condition
                                    left_matrix[idx_j][idx_i] = mass / max_mass
                    
                    print(f"DEBUG LEFT - Direct from network data:")
                    print(f"   Non-zero elements: {np.count_nonzero(left_matrix)}")
                    print(f"   Sum: {np.sum(left_matrix):.3f}")
                    print(f"   Matrix sample (first 3x3): \n{left_matrix[:3,:3]}")
                    
                    # Symmetrize for eigenvalue analysis
                    sym_matrix = left_matrix + left_matrix.T
                    print(f"   After symmetrization - Sum: {np.sum(sym_matrix):.3f}")
                    print(f"   Symmetrized sample (first 3x3): \n{sym_matrix[:3,:3]}")
                    
                    # Apply double-centering (Torgerson procedure) to the symmetrized matrix
                    n = sym_matrix.shape[0]
                    max_sim = np.max(sym_matrix) if np.max(sym_matrix) > 0 else 1.0
                    distance_matrix = max_sim - sym_matrix
                    D_squared = distance_matrix ** 2
                    
                    ones_matrix = np.ones((n, n)) / n
                    H = np.eye(n) - ones_matrix
                    B = -0.5 * H @ D_squared @ H
                    
                    eigenvalues, eigenvectors = np.linalg.eigh(B)
                    print(f"   After double-centering - eigenvalue range: {np.min(eigenvalues):.3f} to {np.max(eigenvalues):.3f}")
                    eigenvector_analysis = f"left_from_network_data"
                    
                else:  # both
                    # For "both", use the full symmetrized matrix
                    sym_matrix = (association_matrix + association_matrix.T) / 2
                    eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)
                    eigenvector_analysis = f"both_{connection_direction}_full_sym"
                
                # Normalize eigenvectors to eigenvalues (not to unit length)
                for i in range(eigenvectors.shape[1]):
                    if abs(eigenvalues[i]) > 1e-10:
                        eigenvectors[:, i] = eigenvectors[:, i] * np.sqrt(abs(eigenvalues[i]))
                
                # eigenvalues are real and can be positive/negative for pseudo-Riemannian space
                
                # Sort by algebraic eigenvalue (descending: most positive first)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
            
            # Calculate warp factor: sum of positive eigenvalues / sum of all eigenvalues
            print(f"🔥 REACHED WARP FACTOR CALCULATION SECTION")
            positive_eigenvals = eigenvalues[eigenvalues > 1e-10]
            negative_eigenvals = eigenvalues[eigenvalues < -1e-10]
            
            print(f"WARP DEBUG ({eigenvector_analysis}): {len(positive_eigenvals)} pos, {len(negative_eigenvals)} neg")
            print(f"WARP DEBUG eigenvalue sample: {eigenvalues[:5]}")
            
            if len(positive_eigenvals) > 0:
                sum_positive = np.sum(positive_eigenvals)
                sum_negative = np.sum(negative_eigenvals)  # This will be negative
                sum_all = sum_positive + sum_negative
                
                print(f"WARP DEBUG sums: pos={sum_positive:.6f}, neg={sum_negative:.6f}, all={sum_all:.6f}")
                print(f"WARP DEBUG all eigenvals: {eigenvalues}")
                print(f"WARP DEBUG eigenval range: min={np.min(eigenvalues):.6f}, max={np.max(eigenvalues):.6f}")
                
                if abs(sum_all) > 1e-8:
                    warp_factor = sum_positive / sum_all  # Use algebraic sum directly (no abs)
                    print(f"WARP DEBUG calculated: {warp_factor:.6f}")
                else:
                    # When positive and negative eigenvalues cancel out exactly, 
                    # the space is perfectly balanced - warp factor approaches infinity
                    warp_factor = float('inf')
                    print(f"WARP DEBUG perfectly balanced space: infinite warp factor")
                
                # Handle infinity and negative warp factors in display
                if warp_factor == float('inf'):
                    st.info(f"🚀 **Warp Factor: ∞** (perfectly balanced) | "
                           f"Positive: {len(positive_eigenvals)} | "
                           f"Negative: {len(negative_eigenvals)} eigenvalues")
                elif warp_factor < 0:
                    st.warning(f"🚀 **Warp Factor: {warp_factor:.3f}** (negative space) | "
                              f"Positive: {len(positive_eigenvals)} | "
                              f"Negative: {len(negative_eigenvals)} eigenvalues")
                else:
                    st.info(f"🚀 **Warp Factor: {warp_factor:.3f}** | "
                           f"Positive: {len(positive_eigenvals)} | "
                           f"Negative: {len(negative_eigenvals)} eigenvalues")
            else:
                warp_factor = 1.0
                print(f"WARP DEBUG: No positive eigenvalues, defaulting to 1.0")
                st.info("🚀 **Warp Factor: 1.000** (no eigenvalues found)")
            
            # Use top 3 dimensions with proper eigenvalue scaling
            if eigenvectors.shape[1] >= 3:
                x = eigenvectors[:, 0] * np.sqrt(np.abs(eigenvalues[0]))
                y = eigenvectors[:, 1] * np.sqrt(np.abs(eigenvalues[1]))
                z = eigenvectors[:, 2] * np.sqrt(np.abs(eigenvalues[2]))
                
                # Calculate percentage of variance explained by first 3 components
                total_variance = np.sum(np.abs(eigenvalues))
                variance_3d = np.sum(np.abs(eigenvalues[:3]))
                variance_percent = (variance_3d / total_variance * 100) if total_variance > 0 else 0
                
            else:
                # Fallback if less than 3 dimensions
                x = eigenvectors[:, 0] if eigenvectors.shape[1] >= 1 else np.zeros(n_concepts)
                y = eigenvectors[:, 1] if eigenvectors.shape[1] >= 2 else np.random.randn(n_concepts) * 0.1
                z = np.random.randn(n_concepts) * 0.1
                variance_percent = 0
                
            st.success(f"✅ Using eigenvector coordinates from {connection_source}")
            st.write(f"Top 3 eigenvalues: {eigenvalues[:3]}")
            st.write(f"Variance explained by 3D plot: {variance_percent:.1f}%")
            
        except np.linalg.LinAlgError:
            st.warning("⚠️ Eigendecomposition failed, using random coordinates")
            np.random.seed(42)
            x = np.random.randn(n_concepts)
            y = np.random.randn(n_concepts)  
            z = np.random.randn(n_concepts)
        
        # Extract data for plotting
        names = [c['name'] for c in concepts_data]
        masses = [c['mass'] for c in concepts_data]
        ids = [c['id'] for c in concepts_data]
        
        # Create mapping from neuron_id to position index
        id_to_index = {neuron_id: i for i, neuron_id in enumerate(ids)}
        
        # Debug: Check what connection attributes exist
        connection_attrs = []
        if hasattr(huey.network, 'synaptic_strengths'):
            connection_attrs.append('synaptic_strengths')
        if hasattr(huey.network, 'inertial_mass'):
            connection_attrs.append('inertial_mass')
        if hasattr(huey.network, 'associations'):
            connection_attrs.append('associations')
        
        # Get connection data - try different sources
        connections = []
        connection_source = None
        
        # Try synaptic_strengths first
        if hasattr(huey.network, 'synaptic_strengths'):
            connection_source = 'synaptic_strengths'
            for (i, j), strength in huey.network.synaptic_strengths.items():
                if i in id_to_index and j in id_to_index and strength > 0.05:  # Lower threshold
                    idx_i, idx_j = id_to_index[i], id_to_index[j]
                    connections.append({
                        'x': [x[idx_i], x[idx_j]], 
                        'y': [y[idx_i], y[idx_j]], 
                        'z': [z[idx_i], z[idx_j]],
                        'strength': strength,
                        'from_id': i,
                        'to_id': j
                    })
        
        # If no synaptic strengths, try inertial_mass connections
        elif hasattr(huey.network, 'inertial_mass'):
            connection_source = 'inertial_mass'
            for (i, j), mass in huey.network.inertial_mass.items():
                if i in id_to_index and j in id_to_index and mass > 0.1:
                    idx_i, idx_j = id_to_index[i], id_to_index[j]
                    connections.append({
                        'x': [x[idx_i], x[idx_j]], 
                        'y': [y[idx_i], y[idx_j]], 
                        'z': [z[idx_i], z[idx_j]],
                        'strength': min(1.0, mass / 5.0),  # Normalize mass to 0-1
                        'from_id': i,
                        'to_id': j
                    })
        
        # Create figure with just the concepts first
        fig = go.Figure()
        
        # Add concept nodes
        fig.add_trace(go.Scatter3d(
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
            hovertemplate='<b>%{text}</b><br>Mass: %{marker.color:.3f}<extra></extra>',
            name="concepts"
        ))
        
        # Create descriptive title based on analysis type
        direction_desc = {
            "outgoing": "Outgoing (Influence) Flow",
            "incoming": "Incoming (Influenced) Flow", 
            "both": "Bidirectional Flow"
        }
        
        eigenvector_desc = {
            "right": "Right Eigenvectors (How concepts influence others)",
            "left": "Left Eigenvectors (How concepts are influenced)",
            "both": "Combined Left/Right Analysis"
        }
        
        if connection_direction == "both":
            title_text = f'3D Concept Space - Traditional Symmetric Analysis ({len(concepts_data)} concepts, {len(connections)} connections)'
        else:
            title_text = f'3D Concept Space - Directional Analysis: {direction_desc[connection_direction]} ({len(concepts_data)} concepts, {len(connections)} connections)'
        
        if 'variance_percent' in locals():
            title_text += f'<br><sub>{eigenvector_desc[eigenvector_type]} from {connection_source} ({variance_percent:.1f}% variance explained)</sub>'
        else:
            title_text += f'<br><sub>{eigenvector_desc[eigenvector_type]} from {connection_source}</sub>'
            
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis_title='1st Eigenvector',
                yaxis_title='2nd Eigenvector', 
                zaxis_title='3rd Eigenvector'
            ),
            width=800,
            height=600
        )
        
        return fig, connections, id_to_index
        
    except Exception as e:
        st.error(f"Error creating 3D plot: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

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
        # Safety Configuration Panel
        st.info("🛡️ **Processing Safety Controls** - Prevents runaway processes")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_neurons = st.number_input(
                "Max Neurons", 
                min_value=100, max_value=500, value=250, step=50,
                help="Reduce for large files to prevent memory issues"
            )
        with col2:
            timeout_hours = st.number_input(
                "Timeout (hours)", 
                min_value=0.5, max_value=12.0, value=2.0, step=0.5,
                help="Auto-stop if processing takes too long"
            )
        with col3:
            exchange_limit = st.number_input(
                "Exchange Limit", 
                min_value=1000, max_value=50000, value=10000, step=1000,
                help="Maximum exchanges to process (0 = no limit)"
            )
        
        # Update Huey network with new neuron limit
        if st.session_state.huey.network.max_neurons != max_neurons:
            st.session_state.huey.network.max_neurons = max_neurons
            st.success(f"✅ Updated neuron limit to {max_neurons}")
        
        uploaded_file = st.file_uploader(
            "Choose a conversation file",
            type=['txt', 'pdf'],
            help="Upload a .txt or .pdf file containing your conversation. The system will automatically detect speakers."
        )
        
        if uploaded_file is not None and not st.session_state.conversation_processed:
            if st.button("🔍 Process Conversation", type="primary"):
                with st.spinner("Processing conversation file..."):
                    # Pass safety parameters to processing function
                    st.session_state.processing_params = {
                        'max_neurons': max_neurons,
                        'timeout_hours': timeout_hours, 
                        'exchange_limit': exchange_limit
                    }
                    result = process_uploaded_file(uploaded_file, st.session_state.huey, 
                                                 timeout_hours, exchange_limit)
                    
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
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Handle different result structures
            if 'speakers_info' in results:
                st.metric("Speakers", len(results['speakers_info']))
            elif 'huey' in results and hasattr(results['huey'], 'network') and hasattr(results['huey'].network, 'speakers'):
                st.metric("Speakers", len(results['huey'].network.speakers))
            else:
                st.metric("Speakers", "Unknown")
        
        with col2:
            if 'conversation_data' in results:
                st.metric("Exchanges", len(results['conversation_data']))
            elif 'exchanges_processed' in results:
                st.metric("Exchanges", f"{results['exchanges_processed']}/{results.get('total_exchanges', '?')}")
            else:
                st.metric("Exchanges", "Unknown")
        
        with col3:
            # Get network stats
            stats = huey.query_concepts("network_statistics")
            if 'neuron_stats' in stats:
                st.metric("Total Concepts", stats['neuron_stats']['total_neurons'])
        
        with col4:
            if 'neuron_stats' in stats:
                st.metric("Total Mass", f"{stats['neuron_stats']['total_mass']:.1f}")
        
        with col5:
            # Display total mass instead of warp factor
            total_mass = sum(huey.network.inertial_mass.values()) if hasattr(huey.network, 'inertial_mass') else 0
            st.metric("Total Mass", f"{total_mass:.1f}", delta="Network Strength")

        # Tabbed interface for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Concept Associations", 
            "🗺️ 3D Visualization", 
            "⚖️ Mass Comparison", 
            "🧮 Eigenvalue Analysis",
            "📊 Network Stats",
            "🧭 Directional Flow Analysis"
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
            
            # Enhanced directional analysis controls
            st.markdown("#### Directional Analysis Settings")
            
            directional_analysis = st.selectbox(
                "🧭 Analysis Type",
                [
                    "both (symmetric analysis)",
                    "outgoing → (right eigenvectors)", 
                    "incoming ← (left eigenvectors)"
                ],
                index=0,
                help="""
                • **Both**: Traditional symmetric analysis (undirected relationships)
                • **Outgoing**: How concepts influence others (right eigenvectors, A·v = λv)
                • **Incoming**: How concepts are influenced (left eigenvectors, u·A = λu)
                """
            )
            
            # Convert selection to parameters
            if "outgoing" in directional_analysis:
                connection_direction = "outgoing"
                eigenvector_type = "right"
            elif "incoming" in directional_analysis:
                connection_direction = "incoming" 
                eigenvector_type = "left"
            else:
                connection_direction = "both"
                eigenvector_type = "right"
            
            # Standard visualization controls
            st.markdown("#### Visualization Parameters")
            col1, col2 = st.columns(2)
            with col1:
                num_concepts_3d = st.number_input("Number of concepts to plot", min_value=10, max_value=200, value=50, step=5)
            with col2:
                min_mass_3d = st.number_input("Minimum mass threshold", min_value=0.0, max_value=1000.0, value=0.0, step=0.1, format="%.1f")
            
            # Initialize session state
            if 'plot_data_3d' not in st.session_state:
                st.session_state.plot_data_3d = None
            
            col_btn1, col_btn2 = st.columns([2, 1])
            
            with col_btn1:
                generate_clicked = st.button("🗺️ Generate 3D Visualization")
            with col_btn2:
                if st.button("🗑️ Clear Plot"):
                    st.session_state.plot_data_3d = None
                    st.rerun()
            
            if generate_clicked:
                with st.spinner("Creating 3D visualization..."):
                    result = create_3d_concept_plot(huey, num_concepts_3d, min_mass_3d, connection_direction, eigenvector_type)
                    
                    if result and result[0] is not None:  # fig is not None
                        fig, connections, id_to_index = result
                        st.session_state.plot_data_3d = {
                            'fig': fig,
                            'connections': connections,
                            'id_to_index': id_to_index,
                            'huey_network': huey.network
                        }
                    else:
                        st.error(f"Not enough concepts with mass ≥ {min_mass_3d} for visualization")
            
            # Display plot and controls if we have plot data
            if st.session_state.plot_data_3d:
                plot_data = st.session_state.plot_data_3d
                fig = plot_data['fig']
                connections = plot_data['connections']
                id_to_index = plot_data['id_to_index']
                huey_network = plot_data['huey_network']
                
                # Always display the base plot first
                st.plotly_chart(fig, use_container_width=True, key="base_plot")
                
                # If we have connections, add the interactive feature
                if connections:
                    # Add concept selector dropdown
                    concept_names = [word for nid, word in huey_network.neuron_to_word.items() 
                                   if nid in id_to_index]
                    concept_names.sort()
                    
                    st.info(f"💡 **Connection Highlighting Available** ({len(connections)} connections found)")
                    
                    # Concept selection dropdown
                    selected_concept = st.selectbox(
                        "🎯 Select concept to highlight connections:",
                        options=['None'] + concept_names,
                        key="concept_selector_3d"
                    )
                    
                    if selected_concept != 'None':
                        # Find the neuron ID for the selected concept
                        selected_id = None
                        for nid, word in huey_network.neuron_to_word.items():
                            if word == selected_concept and nid in id_to_index:
                                selected_id = nid
                                break
                        
                        if selected_id:
                            # Create new figure with glowing connected concepts
                            fig_highlighted = go.Figure()
                            
                            # Build connection info for this concept
                            connected_concepts = {}
                            highlighted_count = 0
                            for conn in connections:
                                if conn['from_id'] == selected_id:
                                    connected_concepts[conn['to_id']] = conn['strength']
                                    highlighted_count += 1
                                elif conn['to_id'] == selected_id:
                                    connected_concepts[conn['from_id']] = conn['strength']
                                    highlighted_count += 1
                            
                            # Rebuild the concept data from what we have available
                            concept_data_list = []
                            for nid, word in huey_network.neuron_to_word.items():
                                if nid in id_to_index:
                                    # Calculate mass
                                    total_mass = 0.0
                                    if hasattr(huey_network, 'inertial_mass'):
                                        for (i, j), mass in huey_network.inertial_mass.items():
                                            if i == nid or j == nid:
                                                total_mass += mass
                                    concept_data_list.append({
                                        'id': nid,
                                        'name': word,
                                        'mass': total_mass,
                                        'index': id_to_index[nid]
                                    })
                            
                            # Sort by index to match the original plot order
                            concept_data_list.sort(key=lambda x: x['index'])
                            
                            # Get max mass for scaling
                            max_mass = max(c['mass'] for c in concept_data_list) if concept_data_list else 1
                            
                            # Separate selected concept from connected ones for different traces
                            selected_x, selected_y, selected_z = [], [], []
                            selected_names, selected_masses = [], []
                            
                            connected_x, connected_y, connected_z = [], [], []
                            connected_names, connected_masses, connected_strengths = [], [], []
                            
                            dimmed_x, dimmed_y, dimmed_z = [], [], []
                            dimmed_names, dimmed_masses = [], []
                            
                            for concept_data in concept_data_list:
                                i = concept_data['index']
                                concept_id = concept_data['id']
                                concept_name = concept_data['name']
                                concept_mass = concept_data['mass']
                                
                                # Get coordinates from the original plot data
                                x_coord = fig.data[0].x[i]
                                y_coord = fig.data[0].y[i]
                                z_coord = fig.data[0].z[i]
                                
                                if concept_id == selected_id:
                                    selected_x.append(x_coord)
                                    selected_y.append(y_coord)
                                    selected_z.append(z_coord)
                                    selected_names.append(concept_name)
                                    selected_masses.append(concept_mass)
                                elif concept_id in connected_concepts:
                                    connected_x.append(x_coord)
                                    connected_y.append(y_coord)
                                    connected_z.append(z_coord)
                                    connected_names.append(concept_name)
                                    connected_masses.append(concept_mass)
                                    connected_strengths.append(connected_concepts[concept_id])
                                else:
                                    dimmed_x.append(x_coord)
                                    dimmed_y.append(y_coord)
                                    dimmed_z.append(z_coord)
                                    dimmed_names.append(concept_name)
                                    dimmed_masses.append(concept_mass)
                            
                            # Add selected concept (yellow glow)
                            if selected_x:
                                fig_highlighted.add_trace(go.Scatter3d(
                                    x=selected_x, y=selected_y, z=selected_z,
                                    mode='markers+text',
                                    marker=dict(
                                        size=[(m/max_mass*30 + 15) for m in selected_masses],
                                        color='yellow',
                                        opacity=1.0,
                                        line=dict(width=4, color='orange')
                                    ),
                                    text=selected_names,
                                    textposition="middle right",
                                    textfont=dict(size=12, color='black'),
                                    hovertemplate='<b>%{text}</b><br>Mass: %{customdata:.3f}<extra></extra>',
                                    customdata=selected_masses,
                                    name="Selected",
                                    showlegend=False
                                ))
                            
                            # Add connected concepts with continuous thermal spectrum
                            if connected_x:
                                # Calculate sizes based on connection strength
                                sizes = [(m/max_mass*20 + s*15 + 8) for m, s in zip(connected_masses, connected_strengths)]
                                line_widths = [max(2, s*6) for s in connected_strengths]
                                
                                fig_highlighted.add_trace(go.Scatter3d(
                                    x=connected_x, y=connected_y, z=connected_z,
                                    mode='markers+text',
                                    marker=dict(
                                        size=sizes,
                                        color=connected_strengths,  # Continuous mapping
                                        colorscale=[[0, 'blue'], [0.5, 'orange'], [1, 'red']],  # Cool to hot spectrum
                                        cmin=0, cmax=1,  # Normalize to 0-1 range
                                        opacity=0.9,
                                        colorbar=dict(title="Connection<br>Strength", x=1.02),
                                        line=dict(width=2, color='black')  # Consistent outline
                                    ),
                                    text=connected_names,
                                    textposition="middle right",
                                    textfont=dict(size=12, color='black'),
                                    hovertemplate='<b>%{text}</b><br>Strength: %{marker.color:.3f}<br>Mass: %{customdata:.3f}<extra></extra>',
                                    customdata=connected_masses,
                                    name="Connected",
                                    showlegend=False
                                ))
                            
                            # Add dimmed unconnected concepts
                            if dimmed_x:
                                fig_highlighted.add_trace(go.Scatter3d(
                                    x=dimmed_x, y=dimmed_y, z=dimmed_z,
                                    mode='markers+text',
                                    marker=dict(
                                        size=[(m/max_mass*15 + 4) for m in dimmed_masses],
                                        color='lightgray',
                                        opacity=0.3,
                                        line=dict(width=1, color='gray')
                                    ),
                                    text=dimmed_names,
                                    textposition="middle right",
                                    textfont=dict(size=10, color='darkgray'),
                                    hovertemplate='<b>%{text}</b><br>Mass: %{customdata:.3f}<extra></extra>',
                                    customdata=dimmed_masses,
                                    name="Unconnected",
                                    showlegend=False
                                ))
                            
                            # Copy layout
                            fig_highlighted.update_layout(fig.layout)
                            fig_highlighted.update_layout(
                                title=f'🎯 Highlighting {highlighted_count} connections for: {selected_concept}'
                            )
                            
                            # Add continuous spectrum legend
                            st.info(
                                "**🌈 Continuous Thermal Spectrum:**\n"
                                "- 🟡 **Yellow**: Selected concept\n"
                                "- 🔴 **Red**: Strongest connections - Hot/Active\n" 
                                "- 🟠 **Orange**: Medium connections - Warm\n"
                                "- 🔵 **Blue**: Weakest connections - Cool/Inactive\n"
                                "- 🔘 **Gray**: Unconnected concepts\n"
                                "- **Color bar**: Shows exact connection strength"
                            )
                            
                            st.plotly_chart(fig_highlighted, use_container_width=True, key="highlighted_plot")
                else:
                    st.warning("No connections found in the network data. The plot shows concept positions only.")
        
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
                
                # Network Health Summary
                st.markdown("### 🏥 Network Health")
                if hasattr(huey.network, 'conversation_history'):
                    st.write(f"**Processing Rate:** {len(huey.network.conversation_history)} conversations processed")
                st.write(f"**System Status:** ✅ Active and Learning")
                
                # Speaker statistics
                if 'speaker_stats' in stats:
                    st.markdown("### Speaker Activity")
                    for speaker, speaker_stats in stats['speaker_stats'].items():
                        blocks = speaker_stats.get('blocks_processed', 0)
                        st.write(f"**{speaker}:** {blocks} blocks processed")
        
        with tab6:
            st.subheader("Advanced Directional Flow Analysis")
            st.markdown("**Analyze temporal and causal concept relationships using left/right eigenvectors**")
            
            # Flow Analysis Section
            st.markdown("### 🌊 Concept Flow Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                flow_concept = st.selectbox(
                    "Select concept for flow analysis",
                    options=[''] + sorted([word for word in huey.network.word_to_neuron.keys()]),
                    key="flow_concept_selector"
                )
            
            with col2:
                flow_threshold = st.number_input(
                    "Flow strength threshold", 
                    min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                    help="Minimum connection strength to show in flow analysis"
                )
            
            if flow_concept and st.button("🔍 Analyze Concept Flow"):
                with st.spinner("Analyzing directional flow patterns..."):
                    # Create flow analysis
                    flow_results = analyze_concept_flow(huey.network, flow_concept, flow_threshold)
                    
                    if flow_results:
                        col_out, col_in = st.columns(2)
                        
                        with col_out:
                            st.markdown(f"#### 📤 **{flow_concept}** → Outgoing Influence")
                            st.caption("Concepts that this concept influences (right eigenvector analysis)")
                            if flow_results['outgoing']:
                                for target, strength in flow_results['outgoing'][:10]:
                                    st.write(f"• **{target}**: {strength:.3f}")
                            else:
                                st.info("No significant outgoing influences found")
                        
                        with col_in:
                            st.markdown(f"#### 📥 **{flow_concept}** ← Incoming Influence")  
                            st.caption("Concepts that influence this concept (left eigenvector analysis)")
                            if flow_results['incoming']:
                                for source, strength in flow_results['incoming'][:10]:
                                    st.write(f"• **{source}**: {strength:.3f}")
                            else:
                                st.info("No significant incoming influences found")
                        
                        # Flow asymmetry analysis
                        if flow_results['asymmetry_score'] > 0:
                            st.markdown("#### ⚖️ Flow Asymmetry Analysis")
                            asymmetry = flow_results['asymmetry_score']
                            if asymmetry > 0.3:
                                st.success(f"🎯 **High Asymmetry** ({asymmetry:.2f}) - Strong directional flow pattern")
                            elif asymmetry > 0.1:
                                st.warning(f"🔄 **Moderate Asymmetry** ({asymmetry:.2f}) - Some directional bias")
                            else:
                                st.info(f"↔️ **Low Asymmetry** ({asymmetry:.2f}) - Mostly bidirectional connections")
            
            # Network-wide asymmetry analysis
            st.markdown("### 🌐 Network Asymmetry Analysis")
            
            if st.button("📊 Analyze Network-wide Flow Patterns"):
                with st.spinner("Computing network asymmetry metrics..."):
                    network_asymmetry = analyze_network_asymmetry(huey.network)
                    
                    if network_asymmetry:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Overall Asymmetry", 
                                f"{network_asymmetry['overall_asymmetry']:.3f}",
                                help="Global measure of directional bias in the network"
                            )
                        
                        with col2:
                            st.metric(
                                "Strong Flows", 
                                network_asymmetry['strong_asymmetric_connections'],
                                help="Number of connections with strong directional bias"
                            )
                        
                        with col3:
                            st.metric(
                                "Flow Efficiency", 
                                f"{network_asymmetry['flow_efficiency']:.3f}",
                                help="How efficiently information flows through the network"
                            )
                        
                        # Top asymmetric concepts
                        st.markdown("#### 🏆 Most Asymmetric Concepts")
                        st.caption("Concepts with strongest directional flow patterns")
                        
                        asymmetric_concepts = network_asymmetry.get('top_asymmetric_concepts', [])
                        if asymmetric_concepts:
                            for i, (concept, score, flow_type) in enumerate(asymmetric_concepts[:10], 1):
                                direction_emoji = "📤" if flow_type == "outgoing" else "📥" if flow_type == "incoming" else "🔄"
                                st.write(f"{i}. {direction_emoji} **{concept}**: {score:.3f} ({flow_type})")
                        else:
                            st.info("No strongly asymmetric concepts found")
        
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