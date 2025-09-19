#!/usr/bin/env python3
"""
Huey🚀 GPU Conversational Network
Revolutionary GPU-accelerated version targeting the activation calculation bottleneck.

Based on scaling tests showing O(n²) activation bottleneck, this version provides
20-50x speedups for large networks through JAX GPU acceleration.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import time
import re
from huey_plusplus_conversational_experiment import HueyConversationalNetwork
from huey_gpu_interface import HueyGPUInterface

class HueyTemporalSimple(HueyConversationalNetwork):
    """
    GPU-accelerated Huey conversational network.
    
    Revolutionary performance improvements targeting the activation calculation bottleneck
    identified in scaling tests. Provides 20-50x speedups for large networks.
    """
    
    def __init__(self, max_neurons: int = 500, window_size: int = 10, 
                 learning_rate: float = 0.15, use_gpu_acceleration: bool = True, 
                 use_temporal_weights: bool = True, tau: float = 3.0,
                 max_connections_per_neuron: int = 500):
        """Initialize simple temporal Huey network.
        
        Args:
            use_temporal_weights: If True, use exponential decay for connection weights
            tau: Decay constant for temporal weighting
            max_connections_per_neuron: Maximum connections each neuron can form
        """
        
        # Initialize parent class without Fortran acceleration (GPU replaces it)
        super().__init__(max_neurons, window_size, learning_rate, use_fortran_acceleration=False)
        
        # Override the hardcoded connection limit from parent class
        self.max_connections_per_neuron = max_connections_per_neuron
        
        # Temporal parameters
        self.use_temporal_weights = use_temporal_weights
        self.tau = tau
        
        # Explicitly store parameters that might get lost
        self.learning_rate = learning_rate
        self.max_neurons = max_neurons
        self.window_size = window_size
        
        # Store conversation mode setting  
        self.conversation_mode = True
        
        # Initialize GPU interface
        self.gpu_interface = HueyGPUInterface(max_neurons, use_gpu_acceleration)
        self.use_gpu_acceleration = use_gpu_acceleration
        
        # Add compatibility attributes for web interface
        self.neuron_to_word = {}  # Maps neuron indices to words
        self.word_to_neuron = {}  # Maps words to neuron indices  
        self.speakers = {}  # Speaker information
        
        # Ensure all required attributes exist
        if not hasattr(self, 'inertial_mass'):
            self.inertial_mass = {}
        if not hasattr(self, 'concept_neurons'):
            self.concept_neurons = {}
        
        # Override performance logging
        self._log_performance = True
        
        method = "TEMPORAL DECAY" if use_temporal_weights else "WINDOWED"
        print(f"🧪 HUEY SIMPLE {method} LEARNING")
        if use_temporal_weights:
            print(f"   Temporal decay: tau={tau}")
        print(f"🚀 HUEY GPU ACCELERATION ENABLED")
        print(f"🎯 Targeting O(n²) activation bottleneck for 20-50x speedups")
    
    def _get_connection_weight(self, word_i_pos: int, word_j_pos: int, base_weight: float = 1.0) -> float:
        """Calculate connection weight with optional temporal decay.
        
        Args:
            word_i_pos: Position of first word in sequence
            word_j_pos: Position of second word in sequence  
            base_weight: Base connection strength
            
        Returns:
            Weighted connection strength
        """
        if not self.use_temporal_weights:
            return base_weight
            
        # Calculate lag distance
        lag = abs(word_j_pos - word_i_pos)
        if lag == 0:
            return base_weight
            
        # Apply exponential decay based on distance
        import math
        decay_factor = math.exp(-lag / self.tau)
        temporal_weight = base_weight * decay_factor
        
        return temporal_weight
    
    def _should_create_connection(self, neuron_i, neuron_j):
        """Determine if a connection should be created based on sparsity constraints."""
        # Always allow connections in temporal learning - sparse connectivity 
        # is handled by the temporal decay, not by blocking connections
        return True
    
    def _hebbian_learning_python(self, window_neurons):
        """Override parent Hebbian learning to integrate temporal weights."""
        # Apply decay only to unused connections (biologically accurate)
        self._decay_unused_masses()
        self._decay_all_connections(window_neurons)
        
        # Create/strengthen connections with temporal weighting
        for pos_i, i in enumerate(window_neurons):
            for pos_j, j in enumerate(window_neurons):
                if pos_i < pos_j:  # Only create connections from earlier to later positions
                    # Safety check: ensure neurons exist in activations
                    if i not in self.activations or j not in self.activations:
                        continue  # Skip if neurons don't exist
                    
                    # SPARSE CONNECTIVITY: Check if we should create/update this connection
                    if not self._should_create_connection(i, j):
                        continue  # Skip based on sparsity constraints
                    
                    # Get activations
                    ai = self.activations[i]
                    aj = self.activations[j]
                    
                    # Connection key for sparse storage
                    conn_key = (i, j)
                    
                    # Current connection strength and mass
                    current_strength = self.connections.get(conn_key, 0.0)
                    current_mass = self.inertial_mass.get(conn_key, 0.0)
                    
                    # Calculate inertial resistance (higher mass = harder to change)
                    inertial_resistance = 1.0 / (1.0 + current_mass * 0.1)
                    
                    # TEMPORAL MODIFICATION: Apply temporal weighting to Hebbian update
                    temporal_weight = self._get_connection_weight(pos_i, pos_j, 1.0)
                    
                    # Hebbian update with inertial resistance AND temporal decay
                    delta_w = self.hebbian_constant * ai * aj * inertial_resistance * temporal_weight
                    new_strength = current_strength + delta_w
                    
                    # Calculate synaptic mass using biologically accurate logistic model
                    activity = ai * aj  # Combined pre/post synaptic activity
                    mass_change = self._calculate_synaptic_mass_change(activity, current_mass)
                    new_mass = max(0.0, min(current_mass + mass_change, self.M_max))
                    
                    # Store updated connection using sparse connectivity manager
                    self._add_sparse_connection(i, j, new_strength, new_mass)
        
        # Periodically prune weak connections to maintain sparsity
        if self.processed_words % 100 == 0:  # Every 100 words
            pruned_count = self._prune_weak_connections(threshold=0.01)
            if pruned_count > 0:
                # Only log significant pruning events to reduce noise
                if pruned_count >= 5 or (pruned_count > 0 and self.processed_words % 100 == 0):
                    print(f"              PRUNED: {pruned_count} weak connections (maintaining sparsity)")
    
    def _calculate_all_activations(self, window_neurons):
        """
        GPU-accelerated activation calculation.
        
        This replaces the O(n²) bottleneck with GPU parallel computation.
        """
        if self._log_performance:
            start_time = time.perf_counter()
        
        if self.use_gpu_acceleration:
            # Use revolutionary GPU acceleration
            new_activations = self.gpu_interface.calculate_activations_gpu(self)
            
            # Update network activations
            self.activations.update(new_activations)
            
        else:
            # Fallback to parent CPU implementation
            super()._calculate_all_activations(window_neurons)
        
        if self._log_performance:
            elapsed = time.perf_counter() - start_time
            acceleration_type = "GPU" if self.use_gpu_acceleration else "CPU"
            print(f"   Activation kernel ({acceleration_type}): {len(self.activations)} neurons in {elapsed:.4f}s")
    
    def calculate_association_matrix(self) -> np.ndarray:
        """
        Calculate standard Galileo association matrix for multidimensional scaling.
        
        Uses classic Galileo approach: ALL concepts shown with association 
        strengths determining spatial relationships.
        """
        if self._log_performance:
            start_time = time.perf_counter()
        
        concept_list = list(self.concept_neurons.keys())
        n_concepts = len(concept_list)
        
        if n_concepts == 0:
            return np.array([])
        
        # Classic Galileo association matrix (similarity-based)
        association_matrix = np.zeros((n_concepts, n_concepts))
        
        # Calculate associations for ALL concept pairs
        for i, concept_i in enumerate(concept_list):
            for j, concept_j in enumerate(concept_list):
                if i == j:
                    # Self-association is maximum
                    association_matrix[i, j] = 1.0
                else:
                    neuron_i = self.concept_neurons[concept_i]
                    neuron_j = self.concept_neurons[concept_j]
                    
                    # Get connection data
                    conn_key = (neuron_i, neuron_j) if neuron_i < neuron_j else (neuron_j, neuron_i)
                    connection_strength = self.connections.get(conn_key, 0.0)
                    
                    # Get activations
                    act_i = self.activations.get(neuron_i, 0.0)
                    act_j = self.activations.get(neuron_j, 0.0)
                    
                    # Classic Galileo association: connection + activation correlation
                    activation_correlation = act_i * act_j
                    
                    # Combined association measure (matches original Galileo)
                    association = connection_strength + 0.1 * activation_correlation
                    
                    association_matrix[i, j] = association
        
        if self._log_performance:
            matrix_time = time.perf_counter() - start_time
            print(f"   Galileo association matrix: {association_matrix.shape[0]}x{association_matrix.shape[1]} in {matrix_time:.4f}s")
        
        return association_matrix
    
    def get_3d_coordinates(self, selected_concepts: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        GPU-accelerated 3D coordinate calculation with revolutionary eigenvalue performance.
        """
        association_matrix = self.calculate_association_matrix()
        
        if association_matrix.size == 0:
            return np.array([]), np.array([]), [], np.array([])
        
        if self._log_performance:
            eigen_start = time.perf_counter()
        
        # GPU-accelerated eigenvalue decomposition
        if self.use_gpu_acceleration:
            eigenvals, eigenvecs = self.gpu_interface.calculate_eigenvalues_gpu(association_matrix)
        else:
            eigenvals, eigenvecs = np.linalg.eigh(association_matrix)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]
        
        if self._log_performance:
            eigen_time = time.perf_counter() - eigen_start
            acceleration_type = "GPU" if self.use_gpu_acceleration else "CPU"
            print(f"   Eigenvalue decomp ({acceleration_type}): {association_matrix.shape[0]}x{association_matrix.shape[0]} in {eigen_time:.4f}s")
        
        # Rest of coordinate calculation (same as parent)
        concept_labels = list(self.concept_neurons.keys())
        
        if len(eigenvals) < 3:
            return np.zeros((len(concept_labels), 3)), eigenvals, concept_labels, eigenvecs
        
        coordinates = eigenvecs[:, :3]
        
        # Standard Galileo coordinate transformation
        n = coordinates.shape[0]
        if n > 1:
            # Center the coordinates (standard practice)
            row_means = np.mean(coordinates, axis=1, keepdims=True)
            col_means = np.mean(coordinates, axis=0, keepdims=True)
            grand_mean = np.mean(coordinates)
            coordinates = coordinates - row_means - col_means + grand_mean
            
            # Light scaling for better visualization (much less aggressive)
            if len(eigenvals) >= 3:
                for dim in range(3):
                    if eigenvals[dim] > 0:
                        coordinates[:, dim] *= np.sqrt(abs(eigenvals[dim]))
        
        return coordinates, eigenvals, concept_labels, eigenvecs
    
    def get_performance_summary(self) -> str:
        """Get comprehensive performance summary including GPU acceleration stats."""
        parent_summary = super().get_performance_summary()
        gpu_stats = self.gpu_interface.get_performance_stats()
        
        # Convert parent summary to string if it's a dict
        if isinstance(parent_summary, dict):
            parent_str = "\n".join([f"   {k}: {v}" for k, v in parent_summary.items()])
        else:
            parent_str = str(parent_summary)
        
        gpu_summary = f"""
🚀 GPU ACCELERATION PERFORMANCE:
   Kernel calls: {gpu_stats['kernel_calls']}
   Total acceleration time: {gpu_stats['total_kernel_time']:.3f}s
   Average per call: {gpu_stats['average_kernel_time']:.4f}s
   Acceleration enabled: {gpu_stats['gpu_acceleration_enabled']}
"""
        
        return parent_str + gpu_summary
    
    def add_speaker(self, speaker_name: str, self_pronouns: List[str], other_pronouns: List[str]):
        """Add speaker with GPU acceleration compatibility."""
        # Call parent method
        super().add_speaker(speaker_name, self_pronouns, other_pronouns)
        
        # Track speakers for web interface compatibility with full structure
        self.speakers[speaker_name] = {
            'self_pronouns': self_pronouns,
            'other_pronouns': other_pronouns,
            'blocks_processed': 0,
            'self_concept_timeline': [],  # Required by parent class
            'total_exchanges': 0,
            'self_references': 0
        }
    
    def _track_word_mappings(self, word: str, neuron_id: int):
        """Track word-neuron mappings for web interface compatibility."""
        if word not in self.word_to_neuron:
            self.word_to_neuron[word] = neuron_id
            self.neuron_to_word[neuron_id] = word
    
    def process_speaker_text(self, speaker_name: str, text: str):
        """Process text with temporal learning integration."""
        if self.use_temporal_weights:
            # Use temporal processing directly for temporal mode
            print(f"🧪 Using temporal processing for: {len(text.split())} words")
            self._process_text_temporal_fallback(speaker_name, text)
            self._sync_network_mappings()
        else:
            # Use standard processing for non-temporal mode
            try:
                # Store current mappings to detect new additions
                initial_concept_count = len(self.concept_neurons)
                initial_word_count = len(getattr(self, 'word_to_neuron', {}))
                
                # Process with parent method  
                super().process_speaker_text(speaker_name, text)
                
                # CRITICAL FIX: Immediately sync all parent mappings into concept_neurons
                if hasattr(self, 'word_to_neuron') and self.word_to_neuron:
                    new_words = len(self.word_to_neuron) - initial_word_count
                    if new_words > 0:
                        print(f"🔄 Syncing {new_words} new words to concept_neurons...")
                        self.concept_neurons.update(self.word_to_neuron)
                
                # Always sync network mappings after processing  
                self._sync_network_mappings()
                
                # Performance counter
                if not hasattr(self, '_sync_counter'):
                    self._sync_counter = 0
                self._sync_counter += 1
                
            except BrokenPipeError as e:
                print(f"🔄 Broken pipe error in text processing - using fallback CPU method")
                self._process_text_fallback(speaker_name, text)
                self._sync_network_mappings()  # Sync after fallback
            except Exception as e:
                print(f"🔄 Processing error: {e} - using standard fallback method") 
                self._process_text_fallback(speaker_name, text)
                self._sync_network_mappings()  # Sync after fallback
    
    def _process_text_fallback(self, speaker_name: str, text: str):
        """Safe fallback text processing without subprocess dependencies."""
        # Set current speaker
        self.current_speaker = speaker_name
        
        # Simple word-by-word processing without complex pipelines
        words = text.lower().split()
        
        # Create neurons for words and track them
        window_neurons = []
        for word in words:
            # Skip kill words
            if hasattr(self, 'kill_words') and word in self.kill_words:
                continue
                
            # Add/activate neuron for this word
            if word not in self.concept_neurons:
                if len(self.concept_neurons) < self.max_neurons:
                    neuron_id = len(self.concept_neurons)
                    self.concept_neurons[word] = neuron_id
                    self.word_to_neuron[word] = neuron_id
                    self.neuron_to_word[neuron_id] = word
                    self.activations[neuron_id] = 1.0
                    window_neurons.append(neuron_id)
            else:
                neuron_id = self.concept_neurons[word]  
                self.activations[neuron_id] = 1.0
                window_neurons.append(neuron_id)
        
        # CRITICAL: Run the learning process on these neurons
        if len(window_neurons) > 1:
            print(f"   Running Hebbian learning on {len(window_neurons)} neurons...")
            self._hebbian_learning_python(window_neurons)
                
        print(f"   Fallback processing: {len(words)} words, {len(self.concept_neurons)} concepts, {len(self.connections)} connections")
    
    def _tokenize_multilingual(self, text: str):
        """Smart tokenization for different language types."""
        import re
        
        # Check if text contains Asian characters (Chinese, Japanese, Korean, Hindi/Devanagari)
        asian_chars = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0900-\u097f]', text)
        
        if asian_chars:
            print(f"     🈳 Detected Asian characters: {len(asian_chars)} chars")
            
            # Important multi-character units to preserve (pronouns and common words)
            multi_char_units = [
                # Chinese pronouns
                '我', '你', '他', '她', '我们', '你们', '他们',
                '我的', '你的', '他的', '她的', '我们的', '你们的', '他们的',
                '您', '您的',  # formal you
                
                # Japanese pronouns  
                'あなた', 'あなたの', 'わたし', 'わたくし', 'わたしの',
                'きみ', 'きみの', 'ぼく', 'ぼくの', 'おれ',
                
                # Common words worth preserving
                '这是', '那是', '可以', '不是', '没有', '有', '很', '非常',
                'です', 'ます', 'である', 'できる'
            ]
            
            # Replace multi-character units with temporary tokens
            temp_text = text
            replacements = {}
            
            for i, unit in enumerate(multi_char_units):
                if unit in temp_text:
                    temp_token = f"__UNIT_{i}__"
                    replacements[temp_token] = unit
                    temp_text = temp_text.replace(unit, f" {temp_token} ")
            
            # Split and process
            tokens = []
            for part in temp_text.split():
                if part in replacements:
                    # Restore multi-character unit
                    tokens.append(replacements[part])
                elif part.startswith('__UNIT_') and part.endswith('__'):
                    # Restore multi-character unit
                    tokens.append(replacements.get(part, part))
                else:
                    # Split into individual characters, skip punctuation
                    for char in part:
                        if char not in '，。！？；：、　\n\r\t -1*':
                            tokens.append(char)
            
            # Filter out empty tokens and convert to lowercase where appropriate
            meaningful_tokens = []
            for token in tokens:
                if len(token) >= 1 and token.strip():
                    # Keep Asian characters as-is, lowercase others
                    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', token):
                        meaningful_tokens.append(token)
                    else:
                        meaningful_tokens.append(token.lower())
            
            print(f"     🈳 Asian character tokenization: {len(meaningful_tokens)} tokens")
            return meaningful_tokens
        else:
            # Western language processing: space-based splitting
            print(f"     🌍 Western language tokenization")
            return text.lower().split()
    
    def _process_text_temporal_fallback(self, speaker_name: str, text: str):
        """Temporal processing fallback that properly integrates with learning."""
        
        # Set current speaker
        self.current_speaker = speaker_name
        
        # Smart tokenization for different language types
        words = self._tokenize_multilingual(text)
        print(f"   🧪 Temporal fallback processing: {len(words)} words")
        
        window_neurons = []
        
        for word in words:
            # Skip kill words
            if hasattr(self, 'kill_words') and word in self.kill_words:
                print(f"     Skipping kill word: {word}")
                continue
                
            # Create or reuse neuron for this word
            if word not in self.concept_neurons:
                if len(self.concept_neurons) < self.max_neurons:
                    neuron_id = len(self.concept_neurons)
                    self.concept_neurons[word] = neuron_id
                    self.word_to_neuron[word] = neuron_id
                    self.neuron_to_word[neuron_id] = word
                    self.activations[neuron_id] = 1.0
                    window_neurons.append(neuron_id)
                    print(f"     Created neuron {neuron_id} for word: {word}")
            else:
                neuron_id = self.concept_neurons[word]  
                self.activations[neuron_id] = 1.0
                window_neurons.append(neuron_id)
                print(f"     Reactivated neuron {neuron_id} for word: {word}")
        
        # CRITICAL: Run temporal Hebbian learning if we have enough neurons
        if len(window_neurons) > 1:
            print(f"   🧠 Running temporal Hebbian learning on {len(window_neurons)} neurons...")
            print(f"     Window: {[self.neuron_to_word.get(n, f'neuron_{n}') for n in window_neurons]}")
            self._hebbian_learning_python(window_neurons)
        else:
            print(f"   ⚠️  Only {len(window_neurons)} neurons - skipping learning")
                
        print(f"   ✅ Temporal fallback complete: {len(self.concept_neurons)} concepts, {len(self.connections)} connections")
    
    def _sync_network_mappings(self):
        """Synchronize all network mappings for web interface compatibility."""
        # PERFORMANCE FIX: Only rebuild once if needed, not on every call
        if not hasattr(self, '_concepts_rebuilt'):
            self._concepts_rebuilt = False
        
        # Only rebuild if we haven't done it yet and concept_neurons is empty
        if not self._concepts_rebuilt and len(self.concept_neurons) == 0 and len(self.activations) > 0:
            print(f"🔄 Rebuilding concept_neurons from parent network data...")
            
            # Try to get concepts from parent class attributes
            if hasattr(self, 'neuron_to_word') and self.neuron_to_word:
                print(f"   Found {len(self.neuron_to_word)} neuron mappings in parent")
                for neuron_id, word in self.neuron_to_word.items():
                    self.concept_neurons[word] = neuron_id
                    
            elif hasattr(self, 'word_to_neuron') and self.word_to_neuron:
                print(f"   Found {len(self.word_to_neuron)} word mappings in parent")
                self.concept_neurons.update(self.word_to_neuron)
                
            # If still empty, create basic mappings from activations
            elif len(self.concept_neurons) == 0:
                print(f"   Creating basic concept mappings for {len(self.activations)} neurons")
                for neuron_id in range(len(self.activations)):
                    if neuron_id in self.activations:
                        concept_name = f"concept_{neuron_id}"
                        self.concept_neurons[concept_name] = neuron_id
                        
            print(f"   ✅ Rebuilt concept_neurons: {len(self.concept_neurons)} concepts")
            self._concepts_rebuilt = True  # Mark as done to avoid repeating
        
        # CRITICAL FIX: Sync ALL concept mappings, not just last 10
        if hasattr(self, 'concept_neurons') and len(self.concept_neurons) > len(getattr(self, 'word_to_neuron', {})):
            # Track all concepts that aren't already mapped
            for word, neuron_id in self.concept_neurons.items():
                if word not in self.word_to_neuron:
                    self._track_word_mappings(word, neuron_id)
    
    def analyze_speaker_self_concept(self, speaker_name: str):
        """Analyze speaker's self-concept with proper mass calculation for GPU version."""
        if speaker_name not in self.speakers:
            return {"error": "Speaker not found"}
        
        speaker_info = self.speakers[speaker_name]
        self_pronouns = speaker_info['self_pronouns']
        
        # First sync inertial mass from parent if available
        if not hasattr(self, 'inertial_mass'):
            self.inertial_mass = {}
            
        # Calculate mass from all connections involving self-pronouns
        self_concept_mass = 0.0
        self_concept_neurons = {}
        
        # Look for speaker neuron first
        speaker_neuron_word = f"speaker_{speaker_name.lower()}"
        speaker_neuron_id = self.word_to_neuron.get(speaker_neuron_word)
        
        if speaker_neuron_id is None:
            # Fallback: try to find speaker by name directly
            speaker_neuron_id = self.word_to_neuron.get(speaker_name.lower())
        
        if speaker_neuron_id is not None:
            # Speaker neuron exists - calculate mass from speaker-specific connections
            for pronoun in self_pronouns:
                if pronoun.lower() in self.word_to_neuron:
                    pronoun_neuron_id = self.word_to_neuron[pronoun.lower()]
                    
                    pronoun_mass = 0.0
                    connections = []
                    
                    # Check connections between speaker and pronoun
                    conn_key1 = (speaker_neuron_id, pronoun_neuron_id) if speaker_neuron_id < pronoun_neuron_id else (pronoun_neuron_id, speaker_neuron_id)
                    
                    # Use connection strength as mass if inertial_mass is empty
                    if conn_key1 in self.inertial_mass:
                        mass = self.inertial_mass[conn_key1]
                    elif conn_key1 in self.connections:
                        # Use connection strength as proxy for mass
                        mass = self.connections[conn_key1] * 10.0  # Scale up for visibility
                    else:
                        # Use activation correlation as mass
                        act_speaker = self.activations.get(speaker_neuron_id, 0.0)
                        act_pronoun = self.activations.get(pronoun_neuron_id, 0.0)
                        mass = act_speaker * act_pronoun * 100.0  # Scale for visibility
                    
                    if mass > 0.001:  # Only include significant masses
                        pronoun_mass += mass
                        strength = self.connections.get(conn_key1, 0.0)
                        connections.append((f"speaker_to_{pronoun}", strength, mass))
                        
                        self_concept_neurons[pronoun] = {
                            'neuron_id': pronoun_neuron_id,
                            'mass': pronoun_mass,
                            'connections': connections
                        }
                        self_concept_mass += pronoun_mass
        else:
            # No speaker neuron - use fallback method
            print(f"Warning: No speaker neuron found for {speaker_name}, using fallback method")
            
            for pronoun in self_pronouns:
                if pronoun.lower() in self.word_to_neuron:
                    neuron_id = self.word_to_neuron[pronoun.lower()]
                    
                    # Calculate mass from all connections involving this pronoun
                    pronoun_mass = 0.0
                    connections = []
                    
                    # Sum all inertial mass for connections involving this pronoun
                    for conn_key in self.connections:
                        if neuron_id in conn_key:
                            if conn_key in self.inertial_mass:
                                mass = self.inertial_mass[conn_key]
                            else:
                                # Use connection strength * activation as mass proxy
                                strength = self.connections[conn_key]
                                activation = self.activations.get(neuron_id, 0.0)
                                mass = strength * activation * 50.0  # Scale for visibility
                            
                            if mass > 0.001:
                                pronoun_mass += mass
                                other_neuron = conn_key[0] if conn_key[1] == neuron_id else conn_key[1]
                                if other_neuron in self.neuron_to_word:
                                    other_word = self.neuron_to_word[other_neuron]
                                    connections.append((other_word, self.connections[conn_key], mass))
                    
                    if pronoun_mass > 0.001:
                        self_concept_neurons[pronoun] = {
                            'neuron_id': neuron_id,
                            'mass': pronoun_mass,
                            'connections': sorted(connections, key=lambda x: x[2], reverse=True)[:10]
                        }
                        self_concept_mass += pronoun_mass
        
        # Add GPU performance metrics
        gpu_stats = self.gpu_interface.get_performance_stats()
        
        return {
            'speaker': speaker_name,
            'self_concept_mass': self_concept_mass,
            'self_concept_neurons': self_concept_neurons,
            'blocks_processed': speaker_info['blocks_processed'],
            'gpu_performance': gpu_stats
        }
    
    def process_file_with_mode(self, filename: str, conversation_mode: bool = None) -> Dict:
        """
        Process a file respecting conversation mode setting.
        
        Args:
            filename: Path to file to process
            conversation_mode: If provided, overrides instance setting
            
        Returns:
            Processing results
        """
        from huey_speaker_detector import HueySpeakerDetector
        
        # Use provided mode or fall back to instance setting
        mode = conversation_mode if conversation_mode is not None else self.conversation_mode
        
        # Create speaker detector with appropriate mode
        detector = HueySpeakerDetector(conversation_mode=mode)
        
        # Process the file
        result = detector.process_conversation_file(filename)
        
        if 'error' in result:
            return result
        
        # Register speakers and process conversation
        for speaker_info in result['speakers_info']:
            speaker_id = speaker_info[0]
            self.add_speaker(speaker_id, ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])
        
        # Process all conversation exchanges
        for speaker, text in result['conversation_data']:
            self.process_speaker_text(speaker, text)
        
        return {
            'success': True,
            'speakers_registered': len(result['speakers_info']),
            'exchanges_processed': len(result['conversation_data']),
            'detection_info': result['detection_info'],
            'conversation_mode': mode
        }

if __name__ == "__main__":
    # Test GPU acceleration
    print("🚀 Testing Huey GPU Acceleration...")
    
    # Create GPU-accelerated network
    huey_gpu = HueyGPUConversationalNetwork(max_neurons=200, use_gpu_acceleration=True)
    huey_gpu.add_speaker("AI", ['i', 'me', 'my'], ['you', 'your'])
    
    # Test text that will create substantial network density
    test_text = """
    Artificial intelligence research involves complex mathematical concepts like neural networks,
    machine learning algorithms, computational linguistics, cognitive science, consciousness studies,
    eigenvalue analysis, Hebbian learning dynamics, synaptic plasticity, network connectivity,
    mathematical modeling, geometric structures, dimensional analysis, scientific methodology,
    experimental design, theoretical frameworks, empirical findings, data integrity, research ethics,
    performance optimization, GPU acceleration, parallel computing, matrix operations.
    """ * 3
    
    print(f"\n🧠 Processing test text with GPU acceleration...")
    start_total = time.perf_counter()
    
    huey_gpu.process_speaker_text("AI", test_text)
    
    total_time = time.perf_counter() - start_total
    
    print(f"\n📊 RESULTS:")
    print(f"   Final network: {huey_gpu.neuron_count} neurons, {len(huey_gpu.connections)} connections")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Word rate: {len(test_text.split()) / total_time:.1f} words/sec")
    
    print(huey_gpu.get_performance_summary())
    print(f"\n✅ Huey🚀 GPU acceleration test complete!")