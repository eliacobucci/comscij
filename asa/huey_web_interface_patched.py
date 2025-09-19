"""
Huey Web Interface (patched)
============================
A user-friendly web interface for the Huey Hebbian Self-Concept Analysis Platform.
Patched to use sparse matrices instead of dense np.zeros, with concept cap.
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
from scipy.sparse import csr_matrix

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

# Initialize session state
if 'huey' not in st.session_state:
    st.session_state.huey = None
if 'conversation_processed' not in st.session_state:
    st.session_state.conversation_processed = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Sparse matrix helper ---
def _build_sparse_assoc_matrix(huey, concept_ids):
    id_to_index = {cid: i for i, cid in enumerate(concept_ids)}
    rows, cols, vals = [], [], []

    edge_dict = None
    if hasattr(huey.network, 'inertial_mass') and isinstance(huey.network.inertial_mass, dict):
        edge_dict = huey.network.inertial_mass
    elif hasattr(huey.network, 'synaptic_strengths') and isinstance(huey.network.synaptic_strengths, dict):
        edge_dict = huey.network.synaptic_strengths

    if edge_dict is None:
        n = len(concept_ids)
        return csr_matrix((n, n), dtype=np.float32)

    for (i, j), strength in edge_dict.items():
        if i in id_to_index and j in id_to_index:
            try:
                s = float(strength)
            except Exception:
                continue
            if s == 0.0:
                continue
            ii = id_to_index[i]
            jj = id_to_index[j]
            rows.append(ii); cols.append(jj); vals.append(s)
            if ii != jj:
                rows.append(jj); cols.append(ii); vals.append(s)

    n = len(concept_ids)
    return csr_matrix((np.array(vals, dtype=np.float32),
                       (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                      shape=(n, n))

# --- Main plotting function (patched) ---
def create_3d_concept_plot(huey, num_concepts, min_mass):
    try:
        # Collect concepts and masses
        all_concepts = []
        for neuron_id, word in huey.network.neuron_to_word.items():
            total_mass = 0.0
            if hasattr(huey.network, 'inertial_mass'):
                for (i, j), mass in huey.network.inertial_mass.items():
                    if i == neuron_id or j == neuron_id:
                        total_mass += mass
            all_concepts.append({'name': word, 'mass': total_mass, 'id': neuron_id})

        all_concepts.sort(key=lambda x: x['mass'], reverse=True)
        concepts_data = all_concepts[:num_concepts]
        if len(concepts_data) < 3:
            st.error(f"Only found {len(concepts_data)} concepts total")
            return None, None, None

        # Cap concepts for web mode
        MAX_CONCEPTS = int(os.getenv("HUEY_MAX_CONCEPTS", "1500"))
        concepts_data = concepts_data[:MAX_CONCEPTS]
        concept_ids = [c['id'] for c in concepts_data]

        # Sparse association matrix
        association_matrix = _build_sparse_assoc_matrix(huey, concept_ids)

        # Low-rank eigendecomposition
        from scipy.sparse.linalg import svds
        try:
            k = min(3, association_matrix.shape[0]-1)
            u, s, vt = svds(association_matrix.astype(np.float32), k=k)
            x, y, z = u[:, -1], u[:, -2], u[:, -3] if k >= 3 else (u[:, 0], u[:, 1], np.random.randn(u.shape[0]))
        except Exception as e:
            st.warning(f"⚠️ Sparse SVD failed: {e}, falling back to random coords")
            np.random.seed(42)
            n = association_matrix.shape[0]
            x, y, z = np.random.randn(n), np.random.randn(n), np.random.randn(n)

        # Plot concepts
        names = [c['name'] for c in concepts_data]
        masses = [c['mass'] for c in concepts_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x.astype(np.float32), y=y.astype(np.float32), z=z.astype(np.float32),
            mode='markers+text',
            marker=dict(size=[m/max(masses)*20+5 for m in masses], color=masses, colorscale='Viridis'),
            text=names, textposition="middle right"
        ))
        fig.update_layout(title=f"3D Concept Space ({len(concepts_data)} concepts)")
        return fig, [], {cid: i for i, cid in enumerate(concept_ids)}
    except Exception as e:
        st.error(f"Error creating 3D plot: {e}")
        return None, None, None

# The rest of the file remains as in your original, but replace any np.zeros((n,n)) blocks with calls to _build_sparse_assoc_matrix.
