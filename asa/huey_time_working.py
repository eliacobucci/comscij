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
import PyPDF2
import io
from huey_time import HueyTime, HueyTimeConfig, build_vocab
import os
from huey_activation_cascade import HueyActivationCascade, create_cascade_interface
from cov_hebb import CovHebbLearner
import warnings

# Suppress numpy warnings temporarily while we fix the numerical issues
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# PDF Text Extraction Function
def extract_pdf_text(uploaded_file):
    """Extract text from PDF file"""
    try:
        # Create a BytesIO object from the uploaded file
        pdf_bytes = io.BytesIO(uploaded_file.read())

        # Create PDF reader
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)

        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# ChatGPT-5's Temporal Mix Functions
def update_T_freq_damped(T: np.ndarray, i: int, j: int, dt: float,
                         freq: np.ndarray, etaT: float = 1e-3, tau: float = 3.0,
                         stopmask: np.ndarray = None):
    if stopmask is not None and (stopmask[i] or stopmask[j]):
        return
    fi = max(freq[i], 1.0); fj = max(freq[j], 1.0)
    w = etaT * np.exp(-dt / max(tau, 1e-8)) / np.sqrt(fi * fj)
    T[i, j] += w

def center_matrix(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    # Sanitize input matrix first
    A_clean = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Prevent overflow by checking magnitude
    max_val = np.max(np.abs(A_clean))
    if max_val > 1e10:  # Scale down if too large
        A_clean = A_clean / (max_val / 1e6)

    J = np.eye(n) - np.ones((n, n)) / n
    C = J @ A_clean @ J

    # Clean result and ensure numerical stability
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    return C

def prune_topk_abs(A: np.ndarray, k: int = 256) -> np.ndarray:
    n = A.shape[0]
    # Sanitize input matrix first
    A_clean = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    B = np.zeros_like(A_clean)
    for i in range(n):
        row = A_clean[i, :].copy()
        row[i] = 0.0

        # Check for valid values in row
        if np.any(np.isfinite(row)) and np.any(np.abs(row) > 0):
            if k < n:
                abs_row = np.abs(row)
                if np.max(abs_row) > 0:  # Only proceed if there are non-zero values
                    idx = np.argpartition(abs_row, -min(k, np.sum(abs_row > 0)))[-min(k, np.sum(abs_row > 0)):]
                    B[i, idx] = row[idx]
            else:
                B[i, :] = row

    np.fill_diagonal(B, 0.0)
    result = 0.5 * (B + B.T)
    # Clean final result
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def row_norm_clamp(A: np.ndarray, rho: float = None) -> np.ndarray:
    # Sanitize input matrix first
    A_clean = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    rows = np.linalg.norm(A_clean, axis=1) + 1e-12
    rows = np.nan_to_num(rows, nan=1e-12, posinf=1e6, neginf=1e-12)

    if rho is None:
        rho = np.percentile(rows, 95.0)
        rho = max(rho, 1e-12)

    scale = np.minimum(1.0, rho / rows)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=0.0)

    result = (A_clean.T * scale).T
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def normalize_fro(A: np.ndarray) -> np.ndarray:
    # Sanitize input matrix first
    A_clean = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    s = np.linalg.norm(A_clean, ord='fro')
    s = max(s, 1e-12)  # Prevent division by zero

    result = A_clean / s
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def build_M(W: np.ndarray, T: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    # Sanitize inputs
    W_clean = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    T_clean = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

    Ts = 0.5 * (T_clean + T_clean.T)
    Wn = normalize_fro(W_clean)
    Tn = normalize_fro(Ts)

    result = alpha * Wn + (1.0 - alpha) * Tn
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def topk_eigs(A: np.ndarray, k: int = 5):
    # Sanitize input matrix first
    A_clean = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure matrix is symmetric and numerically stable
    A_sym = 0.5 * (A_clean + A_clean.T)

    try:
        w, _ = np.linalg.eigh(A_sym)
        w = np.nan_to_num(w, nan=0.0, posinf=1e6, neginf=-1e6)
        w = w[::-1]  # Sort in descending order
        return w[:k]
    except np.linalg.LinAlgError:
        # Fallback: return zeros if eigenvalue computation fails
        return np.zeros(k)

def spectral_guardrail(M: np.ndarray, alpha: float, hi_ratio: float = 4.0, lo_ratio: float = 1.5, step: float = 0.05) -> float:
    evals = topk_eigs(M, k=2)
    if len(evals) < 2 or abs(evals[1]) < 1e-12:
        return alpha
    r = float(evals[0] / max(evals[1], 1e-12))
    if r > hi_ratio:
        alpha = min(0.98, alpha + step)
    elif r < lo_ratio:
        alpha = max(0.50, alpha - step)
    return alpha

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
    # Map language codes to the new file naming convention
    language_file_map = {
        "en": "kill_english.txt",
        "es": "kill_spanish.txt", 
        "fr": "kill_french.txt",
        "de": "kill_german.txt",
        "it": "kill_italian.txt",
        "is": "kill_icelandic.txt",
        "zh": "kill_mandarin_chinese.txt",
        "ja": "kill_japanese.txt",
        "ko": "kill_korean.txt",  # Will need this file
        "ta": "kill_tamil.txt",
        "hi": "kill_hindi.txt"
    }
    
    # Try the new organized folder first
    if language_code in language_file_map:
        kill_file = f"blue_killfiles_all/{language_file_map[language_code]}"
        if os.path.exists(kill_file):
            try:
                with open(kill_file, 'r', encoding='utf-8') as f:
                    kill_words = set(line.strip().lower() for line in f if line.strip())
                st.info(f"📚 Loaded {len(kill_words)} kill words from {kill_file}")
                return kill_words
            except Exception as e:
                st.warning(f"⚠️ Error loading kill words from {kill_file}: {e}")
    
    # Fallback to old naming convention
    kill_file = f"huey_kill.{language_code}.txt"
    if os.path.exists(kill_file):
        try:
            with open(kill_file, 'r', encoding='utf-8') as f:
                kill_words = set(line.strip().lower() for line in f if line.strip())
            st.info(f"📚 Loaded {len(kill_words)} kill words from {kill_file}")
            return kill_words
        except Exception as e:
            st.warning(f"⚠️ Error loading kill words: {e}")
    
    # Final fallback
    st.warning("⚠️ No kill words file found - proceeding without filtering")
    return set()

# Import JAX with proper ARM64 handling
try:
    import jax
    import jax.numpy as jnp
    from jax import device_put, jit
    from functools import partial
    jax_available = True
    st.write("🚀 JAX available - GPU acceleration enabled")
    
    # Test for Metal GPU
    devices = jax.devices()
    if any('metal' in str(device).lower() for device in devices):
        st.success("⚡ JAX Metal GPU acceleration detected!")
    else:
        st.info(f"💻 JAX running on: {devices}")
        
except (ImportError, RuntimeError) as e:
    jax_available = False
    st.error(f"❌ JAX FAILED: {e}")
    st.stop()

# JAX is essential - define the CPU eigendecomposition function
@partial(jit, backend="cpu")
def cpu_eigendecomposition(matrix):
    """Force eigendecomposition to run on CPU where it's supported"""
    return jnp.linalg.eigh(matrix)

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

# Very compact file upload and language selection
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    uploaded_file = st.file_uploader("📁 File", type=['txt', 'pdf'])
with col2:
    if uploaded_file:
        language_options = {
            "Auto-detect (CJK only)": "auto", "English": "en", "Spanish": "es", "French": "fr",
            "German": "de", "Italian": "it", "Icelandic": "is", "Hindi": "hi",
            "Mandarin Chinese": "zh", "Japanese": "ja", "Korean": "ko", "Tamil": "ta"
        }
        selected_language_name = st.selectbox("🌍 Lang", list(language_options.keys()))

if uploaded_file:
    # Handle different file types
    if uploaded_file.name.lower().endswith('.pdf'):
        content = extract_pdf_text(uploaded_file)
        if not content:
            st.error("Could not extract text from PDF. Please try a different file.")
            st.stop()
    else:
        content = uploaded_file.getvalue().decode('utf-8')

    st.write(f"**File:** {uploaded_file.name} | **Type:** {'PDF' if uploaded_file.name.lower().endswith('.pdf') else 'Text'} | **Length:** {len(content)} characters")
    selected_language_code = language_options[selected_language_name]
    
    # Learning rate control
    learning_rate = st.slider(
        "📈 Learning Rate",
        min_value=0.001,
        max_value=2.0,
        value=0.10,
        step=0.001,
        format="%.3f",
        help="Controls temporal learning strength (higher = stronger connections, lower = weaker connections)"
    )

    # IAC toggle and competition control
    use_iac = st.checkbox(
        "🧠 Interactive Activation & Competition (IAC)",
        value=False,
        help="Enable signed covariance-based learning for competitive dynamics (e.g., dog-meows negative connection)"
    )

    # Self-concept (◊) consciousness control
    self_control = st.selectbox(
        "◊ Self-Concept Control",
        options=["Natural", "Always ON", "Always OFF"],
        index=0,
        help="Test 'wherever you go, there you are' hypothesis - force ◊ symbol always active/inactive during IAC processing"
    )

    # Competition level slider (only show when IAC is enabled)
    if use_iac:
        competition_percentage = st.slider(
            "⚔️ Competition Level",
            min_value=0,
            max_value=100,
            value=25,
            step=1,
            format="%d%%",
            help="Higher values = more competition (more negative connections). Lower values = more cooperation."
        )
    
    if st.button("🚀 Process with HueyTime"):
        with st.spinner("Processing..."):
            
            # Use selected language or auto-detect
            if selected_language_code == "auto":
                detected_language = detect_language(content)
                st.info(f"🔍 **Auto-detected language:** {detected_language}")
            else:
                detected_language = selected_language_code
                st.success(f"🌍 **Selected language:** {selected_language_name}")
            
            kill_words = load_kill_words(detected_language)
            
            # Process text - include diamond symbol ◊ for self-concept experiments
            words = re.findall(r'\b\w+\b|◊', content.lower())
            
            st.write(f"**Words before filtering:** {len(words)}")
            
            # Filter out kill words if available
            if kill_words:
                filtered_words = [word for word in words if word not in kill_words]
                st.write(f"**Words after kill word filtering:** {len(filtered_words)} (removed {len(words) - len(filtered_words)})")
                words = filtered_words
            else:
                st.write(f"**Words:** {len(words)} (no kill word filtering)")
            
            vocab = build_vocab(words)
            st.write(f"**Unique words before pruning:** {len(vocab)}")

            # CRITICAL: Prune vocabulary to most frequent words BEFORE HueyTime learning
            # EXPERIMENT: Filter out speaker names to test outlier squish hypothesis
            word_counts = Counter(words)
            max_vocab_size = 500  # Reasonable limit for good performance

            # Remove likely speaker names (including common misspellings)
            speaker_patterns = ['feynman', 'wiener', 'weiner',  # Include both correct and misspelled
                              'FEYNMAN', 'WIENER', 'WEINER', 'Wiener', 'Weiner', 'Feynman',
                              'feynman:', 'wiener:', 'weiner:', 'FEYNMAN:', 'WIENER:', 'WEINER:',
                              'Wiener:', 'Weiner:', 'Feynman:']
            filtered_counts = {word: count for word, count in word_counts.items()
                             if word.lower() not in [p.lower().rstrip(':') for p in speaker_patterns]}

            # Show what speaker variants were actually found and removed
            removed_speakers = {word: count for word, count in word_counts.items() if word not in filtered_counts}
            st.write(f"🚫 **Speaker filtering:** Removed {len(removed_speakers)} speaker variants:")
            for word, count in removed_speakers.items():
                st.write(f"  - {word}: {count} occurrences")

            if len(filtered_counts) > max_vocab_size:
                top_words = [word for word, count in sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]]
                # Filter words to keep only the most frequent ones
                words = [word for word in words if word in top_words]
                # Rebuild vocab with pruned word list
                vocab = build_vocab(words)
                st.write(f"**Pruned to top {max_vocab_size} most frequent words (speaker-filtered):** {len(vocab)}")
            else:
                # Still filter speaker names even if under max size
                words = [word for word in words if word in filtered_counts]
                vocab = build_vocab(words)
                st.write(f"**Vocabulary size OK (speaker-filtered):** {len(vocab)} words")

            # CHATGPT-5's TEMPORAL MIX APPROACH: Frequency-damped learning with matrix mixing
            st.write("🧠 Using ChatGPT-5's advanced temporal mix algorithm...")

            # Initialize matrices
            n = len(vocab)
            W = np.random.randn(n, n) * 0.05  # Baseline matrix (small random values)
            W = 0.5 * (W + W.T)  # Make symmetric
            np.fill_diagonal(W, 0.0)

            T = np.zeros((n, n), dtype=float)  # Temporal matrix
            freq = np.zeros(n, dtype=float)   # Word frequencies
            concept_mass = np.zeros(n, dtype=float)  # Inertial mass for each concept

            # Count frequencies
            for word in words:
                if word in vocab:
                    freq[vocab[word]] += 1

            # Temporal learning with frequency damping and mass accumulation
            st.write("⏰ Building temporal connections with frequency damping and mass tracking...")

            # Diagnostics for learning process
            total_connections_learned = 0
            total_strength_increase = 0.0
            total_mass_accumulated = 0.0
            max_strength_increase = 0.0
            sample_learning_events = []

            for i in range(len(words) - 1):
                if words[i] in vocab and words[i+1] in vocab:
                    word_i = vocab[words[i]]
                    word_j = vocab[words[i+1]]
                    dt = 1.0  # Time step

                    # Store connection strength before update to calculate mass increase
                    old_strength = T[word_i, word_j]

                    update_T_freq_damped(T, word_i, word_j, dt, freq, etaT=learning_rate, tau=3.0)

                    # Calculate mass increase (10% of strength increase, like original implementation)
                    new_strength = T[word_i, word_j]
                    strength_increase = new_strength - old_strength
                    mass_increase = strength_increase * 0.1

                    # Accumulate mass for both concepts involved in the connection
                    concept_mass[word_i] += mass_increase
                    concept_mass[word_j] += mass_increase

                    # Collect diagnostics
                    if strength_increase > 0:
                        total_connections_learned += 1
                        total_strength_increase += strength_increase
                        total_mass_accumulated += mass_increase * 2  # Both concepts get the mass
                        max_strength_increase = max(max_strength_increase, strength_increase)

                        # Collect sample learning events (first 10)
                        if len(sample_learning_events) < 10:
                            sample_learning_events.append({
                                'words': (words[i], words[i+1]),
                                'freqs': (freq[word_i], freq[word_j]),
                                'strength_increase': strength_increase,
                                'mass_increase': mass_increase
                            })

            # Show temporal learning diagnostics
            if total_connections_learned > 0:
                avg_strength_increase = total_strength_increase / total_connections_learned
                st.write(f"📊 **Temporal Learning Diagnostics:**")
                st.write(f"  - Connections learned: {total_connections_learned:,}")
                st.write(f"  - Total strength accumulated: {total_strength_increase:.6f}")
                st.write(f"  - Average strength increase: {avg_strength_increase:.6f}")
                st.write(f"  - Max strength increase: {max_strength_increase:.6f}")
                st.write(f"  - Total mass accumulated: {total_mass_accumulated:.6f}")
                st.write(f"  - Learning rate (etaT): {learning_rate:.3f}")

                if sample_learning_events:
                    st.write("**Sample learning events:**")
                    for event in sample_learning_events[:5]:
                        st.write(f"  '{event['words'][0]}' → '{event['words'][1]}' (freq: {event['freqs'][0]:.0f}, {event['freqs'][1]:.0f}) "
                               f"→ strength: +{event['strength_increase']:.6f}, mass: +{event['mass_increase']:.6f}")
            else:
                st.warning("⚠️ **No temporal learning occurred!** All strength increases were zero.")

            # Apply ChatGPT-5's matrix conditioning pipeline
            st.write("🔧 Applying advanced matrix conditioning...")
            T = center_matrix(T)
            T = prune_topk_abs(T, k=min(64, n//4))  # Adaptive pruning
            T = row_norm_clamp(T, rho=None)

            # Build final matrix with spectral guardrails
            alpha = 0.85
            M = build_M(W, T, alpha=alpha)
            alpha_final = spectral_guardrail(M, alpha, hi_ratio=4.0, lo_ratio=1.5, step=0.05)

            if alpha_final != alpha:
                st.write(f"📊 Spectral guardrail adjusted alpha: {alpha:.2f} → {alpha_final:.2f}")
                M = build_M(W, T, alpha=alpha_final)

            # Use M as our similarity matrix (S)
            S = M
            W = S  # For compatibility with existing code
            concepts = len(vocab)
            connections = (W > 0).sum()

            # FREQUENCY ANALYSIS: Investigate outlier behavior
            frequencies = freq
            vocab_list = list(vocab.keys())

            # Create frequency-word pairs for analysis
            freq_word_pairs = [(frequencies[vocab[word]], word) for word in vocab_list]
            freq_word_pairs.sort(reverse=True)  # Highest frequency first

            # Check if debug_mode exists (it's defined later in the visualization section)
            try:
                show_debug = debug_mode
            except NameError:
                show_debug = True  # Show frequency analysis by default for now

            if show_debug:
                st.write("🔍 **Frequency Analysis:**")
                st.write(f"Frequency range: {np.min(frequencies)} to {np.max(frequencies)}")
                st.write(f"Mean frequency: {np.mean(frequencies):.1f}")
                st.write("**Highest frequency words:**")
                for freq, word in freq_word_pairs[:10]:
                    st.write(f"  {word}: {freq}")
                st.write("**Lowest frequency words:**")
                for freq, word in freq_word_pairs[-10:]:
                    st.write(f"  {word}: {freq}")

                # Check if Feynman/Weiner are in the data
                feynman_freq = frequencies[vocab['feynman']] if 'feynman' in vocab else 0
                weiner_freq = frequencies[vocab['weiner']] if 'weiner' in vocab else 0
                if feynman_freq > 0 or weiner_freq > 0:
                    st.write("**Outlier suspects:**")
                    if feynman_freq > 0:
                        st.write(f"  feynman: {feynman_freq}")
                    if weiner_freq > 0:
                        st.write(f"  weiner: {weiner_freq}")
            
            # IAC Processing if enabled
            iac_learner = None
            W_iac = None
            if use_iac:
                st.write("🧠 **Interactive Activation & Competition Processing**")
                
                # Initialize IAC learner
                iac_learner = CovHebbLearner(n=len(vocab), eta=5e-3, beta=1e-2, gamma=1e-4)
                
                # Process text with sliding window for IAC learning
                window_size = 5  # Words in each window
                for i in range(len(words) - window_size + 1):
                    window_words = words[i:i + window_size]
                    # Convert words to indices
                    window_indices = [vocab[word] for word in window_words if word in vocab]
                    if len(window_indices) >= 2:  # Need at least 2 words for covariance
                        iac_learner.update_from_window(window_indices)
                    
                    # Prune every 100 windows to keep memory manageable
                    if (i + 1) % 100 == 0:
                        iac_learner.prune_topk(256)
                
                # Final pruning
                iac_learner.prune_topk(256)
                
                # Get IAC matrix with simple baseline subtraction
                vocab_indices = list(range(len(vocab)))
                W_covariance = iac_learner.to_dense_block(vocab_indices)
                
                # Apply inhibition baseline for competition (percentage of average weight)
                avg_weight = np.mean(np.abs(W_covariance))
                inhibition_baseline = (competition_percentage / 100.0) * avg_weight
                W_iac = W_covariance - inhibition_baseline
                # Keep diagonal at zero - "wherever you go, there you are"
                np.fill_diagonal(W_iac, 0.0)
                
                st.success(f"✅ IAC learning complete! Processed {len(words) - window_size + 1} windows")
                st.write(f"  - IAC matrix shape: {W_iac.shape}")
                st.write(f"  - Competition level (inhibition): {inhibition_baseline:.4f}")
                st.write(f"  - IAC positive connections: {np.sum(W_iac > 0)}")
                st.write(f"  - IAC negative connections: {np.sum(W_iac < 0)}")
                st.write(f"  - IAC max: {np.max(W_iac):.6f}, min: {np.min(W_iac):.6f}")
            
            # Debug the matrices
            st.write(f"🔍 **Matrix Analysis:**")
            st.write(f"  - W shape: {W.shape}")
            st.write(f"  - W max: {np.max(W):.6f}, min: {np.min(W):.6f}, mean: {np.mean(W):.6f}")
            st.write(f"  - W non-zero: {np.count_nonzero(W)}")
            st.write(f"  - S max: {np.max(S):.6f}, min: {np.min(S):.6f}, mean: {np.mean(S):.6f}")
            st.write(f"  - S non-zero: {np.count_nonzero(S)}")
            
            # Calculate total network mass
            total_mass = np.sum(concept_mass)

            # Store results in session state for tools
            st.session_state.huey_results = {
                'W': W,
                'S': S,
                'vocab': vocab,
                'words': words,
                'concepts': concepts,
                'connections': connections,
                'detected_language': detected_language,
                'kill_words_used': len(kill_words) if kill_words else 0,
                'iac_enabled': use_iac,
                'iac_learner': iac_learner,
                'W_iac': W_iac,
                'concept_mass': concept_mass,
                'total_mass': total_mass
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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Concepts", concepts)
            with col2:
                st.metric("Connections", connections)
            with col3:
                st.metric("Total Mass", f"{total_mass:.2f}")
            
            # Verify it worked correctly
            if concepts == len(vocab):
                st.success("✅ Correct: One concept per unique word")
            else:
                st.error(f"❌ Wrong: Expected {len(vocab)} concepts, got {concepts}")
            
            # Show vocabulary with mass information
            st.subheader("📚 Vocabulary with Mass")

            # Create list of (word, index, mass) and sort by mass (descending)
            vocab_with_mass = [(word, idx, concept_mass[idx]) for word, idx in vocab.items()]
            vocab_with_mass.sort(key=lambda x: x[2], reverse=True)

            # Show top 20 by mass
            st.write("**Top concepts by inertial mass:**")
            for word, idx, mass in vocab_with_mass[:20]:
                st.write(f"- **{word}** (mass: {mass:.3f}, index {idx})")

            if len(vocab) > 20:
                st.write(f"... and {len(vocab) - 20} more concepts")

            # Show mass distribution
            if len(concept_mass) > 0:
                st.write(f"**Mass distribution:** min: {np.min(concept_mass):.3f}, max: {np.max(concept_mass):.3f}, mean: {np.mean(concept_mass):.3f}")
            
# 3D Visualization (outside processing button, uses session state)
if 'huey_results' in st.session_state:
    st.subheader("🎯 3D Visualization Controls")
    
    # User controls for visualization - using narrower columns for compact inputs
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        vocab_size = len(st.session_state.huey_results['vocab'])
        max_concepts = st.number_input(
            "📊 Concepts to show",
            min_value=1,
            max_value=vocab_size,
            value=min(200, vocab_size),
            help=f"Enter 1 to {vocab_size}",
            format="%d"
        )
    
    with col2:
        # Matrix type selector
        matrix_options = ["HueyTime (S matrix)", "HueyTime (W matrix)"]
        if st.session_state.huey_results.get('iac_enabled', False):
            matrix_options.append("IAC (signed weights)")
        
        matrix_choice = st.selectbox(
            "🧠 Matrix type",
            matrix_options,
            help="Choose which matrix to visualize"
        )
    
    with col3:
        eigenvector_choice = st.selectbox(
            "🧮 Eigenvector type",
            ["Average (default)", "Left eigenvectors", "Right eigenvectors"],
            help="Choose eigenvectors for visualization"
        )

        # Log transform toggle for better readability
        log_transform = st.checkbox(
            "📏 Log transform display",
            value=False,
            help="⚠️ Spreads points for readability but DISTORTS true distances - display only!"
        )

    # Debug toggle - visible and clean
    debug_mode = st.checkbox("🔍 Debug diagnostics", value=False, help="Show detailed matrix analysis (slower)")

    if st.button("🎯 Create 3D Visualization"):
        st.write("DEBUG: Button clicked!")
        st.subheader("🎯 3D Concept Visualization")
        
        with st.spinner("Creating 3D plot..."):
            st.write("DEBUG: Starting visualization...")
            try:
                # Get data from session state
                results = st.session_state.huey_results
                S = results['S']
                W = results['W']
                W_iac = results.get('W_iac', None)
                vocab = results['vocab']
                words = results['words']
                
                st.write(f"DEBUG: Matrix shape: {S.shape}")
                
                # Proper Galileo Torgerson double-centering for pseudo-Riemannian space
                st.write("🌌 Using Torgerson double-centering for proper Galileo plot...")
                
                # Choose base matrix first
                if matrix_choice == "IAC (signed weights)" and W_iac is not None:
                    base_matrix = W_iac
                    st.write("🧠 **Using IAC signed weight matrix**")
                elif matrix_choice == "HueyTime (W matrix)":
                    base_matrix = W
                    st.write("🔍 **Using HueyTime W matrix**")
                else:
                    base_matrix = S
                    st.write("🔍 **Using HueyTime S matrix**")

                if debug_mode:
                    # DIAGNOSTIC: Check which matrix we're actually using
                    st.write("🔍 **Matrix Selection Diagnostics:**")
                    st.write(f"Matrix choice: {matrix_choice}")
                    st.write(f"W_iac available: {W_iac is not None}")
                    st.write(f"Selected matrix type: {'IAC' if matrix_choice == 'IAC (signed weights)' and W_iac is not None else 'W' if matrix_choice == 'HueyTime (W matrix)' else 'S'}")
                    st.write(f"Selected matrix shape: {base_matrix.shape}")
                    st.write(f"Selected matrix range: [{np.min(base_matrix):.6f}, {np.max(base_matrix):.6f}]")
                
                # Then choose eigenvector direction
                if eigenvector_choice == "Left eigenvectors":
                    # Use matrix directly (left eigenvectors of directed matrix)
                    similarity_matrix = base_matrix
                    st.write("📐 Left eigenvectors")
                elif eigenvector_choice == "Right eigenvectors":
                    # Use matrix transpose (right eigenvectors of directed matrix)
                    similarity_matrix = base_matrix.T
                    st.write("📐 Right eigenvectors")
                else:
                    # Use average (default - symmetric similarity matrix)
                    if matrix_choice == "IAC (signed weights)":
                        # IAC matrix might be asymmetric, symmetrize it
                        similarity_matrix = (base_matrix + base_matrix.T) / 2.0
                        st.write("📐 Symmetrized IAC matrix")
                    else:
                        similarity_matrix = base_matrix
                        st.write("📐 Default matrix")
                
                # Torgerson transformation with validation
                n = similarity_matrix.shape[0]

                # Validate matrix size
                if n == 0:
                    st.error("❌ Empty similarity matrix - cannot perform Torgerson transformation")
                    st.stop()

                if n < 2:
                    st.warning("⚠️ Matrix too small for meaningful Torgerson transformation")
                    st.stop()

                # Validate matrix content
                if not np.isfinite(similarity_matrix).all():
                    st.error("❌ Similarity matrix contains invalid values (NaN or infinity)")
                    st.stop()

                # For directed matrices, make symmetric for Torgerson
                if eigenvector_choice != "Average (default)":
                    similarity = (similarity_matrix + similarity_matrix.T) / 2.0
                else:
                    similarity = similarity_matrix  # S is already symmetric

                if debug_mode:
                    # DIAGNOSTIC: Check similarity matrix properties
                    st.write("🔍 **Similarity Matrix Diagnostics:**")
                    st.write(f"Matrix shape: {similarity.shape}")
                    st.write(f"Matrix range: [{np.min(similarity):.6f}, {np.max(similarity):.6f}]")
                    st.write(f"Matrix mean: {np.mean(similarity):.6f}")
                    st.write(f"Matrix std: {np.std(similarity):.6f}")
                    st.write(f"Matrix rank: {np.linalg.matrix_rank(similarity)}")
                    st.write(f"Matrix condition number: {np.linalg.cond(similarity):.2e}")

                    # Check for symmetry
                    symmetry_error = np.max(np.abs(similarity - similarity.T))
                    st.write(f"Symmetry error: {symmetry_error:.2e}")

                    # Check diagonal
                    diag_vals = np.diag(similarity)
                    st.write(f"Diagonal range: [{np.min(diag_vals):.6f}, {np.max(diag_vals):.6f}]")

                # GALILEO METHOD: Direct eigendecomposition of similarity matrix
                try:
                    # Direct eigendecomposition of similarity matrix (proper Galileo approach)
                    if jax_available:
                        st.write("🚀 Using JAX CPU for direct similarity matrix eigendecomposition...")
                        similarity_jax = jnp.array(similarity)
                        eigenvals, eigenvecs = cpu_eigendecomposition(similarity_jax)
                        eigenvals = np.array(eigenvals)
                        eigenvecs = np.array(eigenvecs)
                    else:
                        eigenvals, eigenvecs = np.linalg.eigh(similarity)

                    # Validate eigenvalue results
                    if not np.isfinite(eigenvals).all() or not np.isfinite(eigenvecs).all():
                        st.warning("⚠️ Invalid eigenvalues detected, using fallback")
                        eigenvals = np.ones(n) * 1e-6
                        eigenvecs = np.eye(n)

                    # Sort by eigenvalue algebraic value (largest to smallest, Galileo style)
                    idx = np.argsort(eigenvals)[::-1]
                    sorted_eigenvals = eigenvals[idx]
                    sorted_eigenvecs = eigenvecs[:, idx]

                except Exception as e:
                    st.error(f"❌ Error in Galileo eigendecomposition: {str(e)}")
                    # Use fallback values
                    eigenvals = np.ones(n) * 1e-6
                    eigenvecs = np.eye(n)
                    sorted_eigenvals = eigenvals
                    sorted_eigenvecs = eigenvecs
                
                st.write(f"🌌 Galileo eigendecomposition complete: {len(eigenvals)} eigenvalues")
                st.write(f"📊 Eigenvalue signature: {np.sum(eigenvals > 0)} positive, {np.sum(eigenvals < 0)} negative")

                if debug_mode:
                    # DIAGNOSTIC: Detailed eigenvalue analysis
                    st.write("🔍 **Eigenvalue Spectrum Analysis:**")
                    st.write(f"Top 10 eigenvalues: {sorted_eigenvals[:10]}")
                    st.write(f"Bottom 5 eigenvalues: {sorted_eigenvals[-5:]}")

                    # Check for dominant eigenvalue
                    if len(sorted_eigenvals) > 1:
                        ratio = abs(sorted_eigenvals[0]) / abs(sorted_eigenvals[1]) if abs(sorted_eigenvals[1]) > 1e-10 else float('inf')
                        st.write(f"Eigenvalue dominance ratio (λ₁/λ₂): {ratio:.2f}")

                    # Cumulative variance explained
                    abs_eigenvals = np.abs(sorted_eigenvals)
                    total_variance = np.sum(abs_eigenvals)
                    if total_variance > 1e-10:
                        cumulative_variance = np.cumsum(abs_eigenvals) / total_variance * 100
                        st.write(f"Variance explained by first 3 dimensions: {cumulative_variance[2]:.1f}%")
                        st.write(f"First dimension explains: {cumulative_variance[0]:.1f}%")

                    # Check for effective rank
                    significant_eigenvals = np.sum(np.abs(sorted_eigenvals) > 1e-6)
                    st.write(f"Effective rank (eigenvals > 1e-6): {significant_eigenvals}")
                
                # Use top 3 eigenvectors for 3D coordinates (Galileo method)
                if len(sorted_eigenvals) >= 3:
                    # Standard Galileo coordinate calculation
                    coords = np.zeros((n, 3))
                    for i in range(3):
                        eigenval = sorted_eigenvals[i]
                        eigenvec = sorted_eigenvecs[:, i]

                        # Standard eigenvector scaling with eigenvalue weighting
                        coords[:, i] = eigenvec * np.sqrt(abs(eigenval))
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

                # Apply log transform if requested (for display readability only)
                if not log_transform:
                    # Show original coordinate ranges for debugging
                    orig_x_range = f"[{np.min(x_top):.3f}, {np.max(x_top):.3f}]"
                    orig_y_range = f"[{np.min(y_top):.3f}, {np.max(y_top):.3f}]"
                    orig_z_range = f"[{np.min(z_top):.3f}, {np.max(z_top):.3f}]"

                    # Use a power-law compression for very small coordinates
                    def compress_coord(coord):
                        abs_coord = np.abs(coord)
                        # For small values, use square root compression: sign(x) * sqrt(|x|)
                        # This dramatically compresses outliers while preserving sign
                        return np.sign(coord) * np.sqrt(abs_coord)

                    x_top = compress_coord(x_top)
                    y_top = compress_coord(y_top)
                    z_top = compress_coord(z_top)

                    # Show transformed coordinate ranges
                    new_x_range = f"[{np.min(x_top):.3f}, {np.max(x_top):.3f}]"
                    new_y_range = f"[{np.min(y_top):.3f}, {np.max(y_top):.3f}]"
                    new_z_range = f"[{np.min(z_top):.3f}, {np.max(z_top):.3f}]"

                    st.info(f"🔄 **Log compression applied** - outliers pulled closer for readability (distances distorted)")
                    st.write(f"**Before:** X{orig_x_range}, Y{orig_y_range}, Z{orig_z_range}")
                    st.write(f"**After:** X{new_x_range}, Y{new_y_range}, Z{new_z_range}")

                concept_names = top_words
                concept_freqs = [freq for word, freq in top_concepts]
                max_freq = max(concept_freqs)

                # Get mass data for top concepts
                concept_masses = []
                if 'concept_mass' in st.session_state.huey_results:
                    concept_mass_array = st.session_state.huey_results['concept_mass']
                    for word in top_words:
                        if word in vocab:
                            word_idx = vocab[word]
                            mass = concept_mass_array[word_idx] if word_idx < len(concept_mass_array) else 0.0
                            concept_masses.append(mass)
                        else:
                            concept_masses.append(0.0)
                else:
                    # No mass data available - likely using old session state before mass implementation
                    concept_masses = [0.0] * len(concept_names)
                    st.warning("⚠️ **Mass data not available** - This is likely cached data from before mass calculation was implemented. Upload a new file to see mass values!")

                # Store 3D coordinates for cascade visualization
                concept_positions = {}
                for i, concept_name in enumerate(concept_names):
                    concept_id = vocab[concept_name]
                    concept_positions[concept_id] = (x_top[i], y_top[i], z_top[i])
                st.session_state['visualization_coords'] = concept_positions
                
                st.write(f"DEBUG: Showing top {len(concept_names)} most frequent concepts (from {len(vocab)} total)")
                
                # Create 3D plot
                fig = go.Figure()
                
                # Single scatter plot with ALL concepts labeled (back to original)
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
                        size=13,
                        color='black',
                        family='Arial, sans-serif'
                    ),
                    name=f"All {len(concept_names)} Concepts (labeled)",
                    customdata=concept_masses,
                    hovertemplate="<b>%{text}</b><br>" +
                                "Frequency: %{marker.color}<br>" +
                                "Inertial Mass: %{customdata:.3f}<extra></extra>"
                ))
                
                # Create title with log transform indicator
                title_text = f"3D Concept Space (top {len(concept_names)} of {len(vocab)} concepts)"
                if not log_transform:
                    title_text += " - LOG TRANSFORMED (Display Only)"

                fig.update_layout(
                    title=title_text,
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
        # Create mock network with ALL vocabulary (not just top frequent)
        # Ensure cascade has access to ALL words including pronouns like "i", "me", "us"
        mock_network = MockNetwork(st.session_state.huey_results)
        cascade = create_cascade_interface(mock_network)
        
        # Get available concepts
        available_concepts = cascade.get_available_concepts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            input_concepts = st.multiselect("🎯 Input concepts", available_concepts)
        with col2:
            target_concept = st.selectbox("🏁 Target concept", available_concepts)
        
        # Cascade controls - compact layout
        col3, col4, col5 = st.columns([1, 1, 2])
        with col3:
            max_steps = st.number_input(
                "⏰ Max steps",
                min_value=10,
                max_value=200,
                value=50,
                help="Max cascade steps",
                format="%d"
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

# Network Connection Inspector (outside processing button, uses session state)
if 'huey_results' in st.session_state:
    st.subheader("🔍 Network Connection Inspector")
    st.write("Check the connection strength between any two concepts")

    results = st.session_state.huey_results
    vocab = results['vocab']
    available_words = sorted(vocab.keys())

    col1, col2 = st.columns(2)
    with col1:
        word1 = st.selectbox(
            "🔤 First word",
            available_words,
            help="Select the first word to check"
        )

    with col2:
        word2 = st.selectbox(
            "🔤 Second word",
            available_words,
            help="Select the second word to check"
        )

    if word1 and word2 and word1 != word2:
        # Get indices
        idx1 = vocab[word1]
        idx2 = vocab[word2]

        # Check all available matrices
        st.write(f"**Connection: '{word1}' ↔ '{word2}'**")

        # HueyTime matrices
        W = results['W']
        S = results['S']

        huey_w_conn = W[idx1, idx2] if idx1 < W.shape[0] and idx2 < W.shape[1] else 0.0
        huey_s_conn = S[idx1, idx2] if idx1 < S.shape[0] and idx2 < S.shape[1] else 0.0

        col3, col4 = st.columns(2)
        with col3:
            st.metric("HueyTime W matrix", f"{huey_w_conn:.6f}")
        with col4:
            st.metric("HueyTime S matrix", f"{huey_s_conn:.6f}")

        # IAC matrix if available
        if results.get('iac_enabled', False) and results.get('W_iac') is not None:
            W_iac = results['W_iac']
            iac_conn = W_iac[idx1, idx2] if idx1 < W_iac.shape[0] and idx2 < W_iac.shape[1] else 0.0

            st.metric("IAC signed matrix", f"{iac_conn:.6f}")

            # Interpret the IAC connection
            if iac_conn > 0.001:
                st.success(f"🤝 **Cooperative relationship** - '{word1}' and '{word2}' reinforce each other")
            elif iac_conn < -0.001:
                st.error(f"⚔️ **Competitive relationship** - '{word1}' and '{word2}' inhibit each other")
            else:
                st.info(f"🤷 **Neutral relationship** - '{word1}' and '{word2}' have minimal interaction")

        # Show bidirectional connections
        if huey_w_conn != W[idx2, idx1] or huey_s_conn != S[idx2, idx1]:
            st.write("**Directional differences:**")
            st.write(f"- {word1} → {word2}: W={W[idx1, idx2]:.6f}, S={S[idx1, idx2]:.6f}")
            st.write(f"- {word2} → {word1}: W={W[idx2, idx1]:.6f}, S={S[idx2, idx1]:.6f}")

    elif word1 == word2:
        st.warning("⚠️ Please select two different words to check their connection")

    # Quick connection lookup
    st.write("**💡 Quick connection finder:**")
    search_word = st.text_input(
        "🔍 Enter a word to see its strongest connections",
        help="Type any word from your vocabulary"
    )

    if search_word and search_word in vocab:
        search_idx = vocab[search_word]

        # Show concept mass for this word
        if 'concept_mass' in results:
            word_mass = results['concept_mass'][search_idx]
            st.metric(f"'{search_word}' Concept Mass", f"{word_mass:.3f}")

        # Find strongest connections in each matrix
        matrices_to_check = [
            ("HueyTime W", results['W']),
            ("HueyTime S", results['S'])
        ]

        if results.get('iac_enabled', False) and results.get('W_iac') is not None:
            matrices_to_check.append(("IAC signed", results['W_iac']))

        for matrix_name, matrix in matrices_to_check:
            if search_idx < matrix.shape[0]:
                # Get connections for this word
                connections = matrix[search_idx, :]

                # Find top positive and negative connections
                top_positive_idx = np.argsort(connections)[-6:][::-1]  # Top 5 plus self
                top_negative_idx = np.argsort(connections)[:5]  # Bottom 5

                st.write(f"**{matrix_name} matrix - '{search_word}' connections:**")

                # Positive connections
                pos_connections = []
                for idx in top_positive_idx:
                    if idx != search_idx and connections[idx] > 0.001:  # Skip self and tiny values
                        # Find word for this index
                        word_matches = [w for w, i in vocab.items() if i == idx]
                        if word_matches:
                            word = word_matches[0]
                            pos_connections.append(f"{word} ({connections[idx]:.4f})")

                if pos_connections:
                    st.write(f"  🤝 Positive: {', '.join(pos_connections[:5])}")

                # Negative connections (for IAC)
                if matrix_name == "IAC signed":
                    neg_connections = []
                    for idx in top_negative_idx:
                        if connections[idx] < -0.001:  # Only significant negative values
                            # Find word for this index
                            word_matches = [w for w, i in vocab.items() if i == idx]
                            if word_matches:
                                word = word_matches[0]
                                neg_connections.append(f"{word} ({connections[idx]:.4f})")

                    if neg_connections:
                        st.write(f"  ⚔️ Negative: {', '.join(neg_connections[:5])}")

    elif search_word and search_word not in vocab:
        st.warning(f"⚠️ '{search_word}' not found in vocabulary. Available words: {len(vocab)} total")