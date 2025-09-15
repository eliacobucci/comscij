#!/usr/bin/env python3
"""
Huey🚀 Temporal Learning Experiment
Testing time-delay Hebbian learning vs windowed approach.

This experimental version implements temporal decay learning to compare 
against the working windowed method.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import time
import re
from math import exp
from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork

class HueyTemporalExperiment(HueyGPUConversationalNetwork):
    """
    Experimental temporal learning version of Huey.
    
    Tests time-delay learning with exponential decay vs windowed approach.
    Includes extensive debugging to understand why previous attempts failed.
    """
    
    def __init__(self, max_neurons: int = 500, window_size: int = 10, 
                 learning_rate: float = 0.15, use_gpu_acceleration: bool = True,
                 use_temporal_learning: bool = False,
                 tau: float = 3.0, eta_fwd: float = 1e-2, eta_fb: float = 2e-3,
                 boundary_penalty: float = 0.25):
        """Initialize experimental Huey with temporal learning option.
        
        Args:
            use_temporal_learning: If True, use time-delay method; if False, use windowed
            tau: Exponential decay constant for distance weighting
            eta_fwd: Forward learning rate (A->B)
            eta_fb: Backward/feedback learning rate (B->A) 
            boundary_penalty: Multiplier for cross-sentence learning
        """
        
        super().__init__(max_neurons, window_size, learning_rate, use_gpu_acceleration)
        
        # Temporal learning parameters
        self.use_temporal_learning = use_temporal_learning
        self.tau = tau
        self.eta_fwd = eta_fwd  
        self.eta_fb = eta_fb
        self.boundary_penalty = boundary_penalty
        self.max_lag = window_size  # Use window_size as max lag distance
        
        # Debug tracking
        self.debug_updates = []
        self.debug_zero_updates = 0
        self.debug_nonzero_updates = 0
        
        learning_method = "TEMPORAL DECAY" if use_temporal_learning else "WINDOWED"
        print(f"🧪 EXPERIMENTAL HUEY - {learning_method} LEARNING")
        if use_temporal_learning:
            print(f"   Temporal params: tau={tau}, eta_fwd={eta_fwd}, eta_fb={eta_fb}")
            print(f"   Max lag: {self.max_lag}, boundary_penalty: {boundary_penalty}")
    
    def _process_tokens_temporal(self, tokens: List[str], boundaries: Optional[List[int]] = None):
        """
        Process tokens using temporal decay learning method.
        
        Args:
            tokens: List of word tokens
            boundaries: Optional list of boundary positions (sentence/turn ends)
        """
        if not tokens:
            return
            
        print(f"   Temporal learning: {len(tokens)} tokens, boundaries: {boundaries}")
        
        # Convert tokens to neuron indices, creating new neurons as needed
        token_indices = []
        for token in tokens:
            # Skip kill words if kill_words is available
            if hasattr(self, 'kill_words') and token.lower() in self.kill_words:
                continue
                
            if token.lower() in self.concept_neurons:
                neuron_id = self.concept_neurons[token.lower()]
            else:
                # Create new neuron if we have space
                if len(self.concept_neurons) < self.max_neurons:
                    neuron_id = len(self.concept_neurons)
                    self.concept_neurons[token.lower()] = neuron_id
                    self.word_to_neuron[token.lower()] = neuron_id
                    self.neuron_to_word[neuron_id] = token.lower()
                    self.activations[neuron_id] = 0.0
                else:
                    continue  # Skip if network is full
                    
            token_indices.append(neuron_id)
        
        if len(token_indices) < 2:
            print(f"   Warning: Only {len(token_indices)} valid tokens, skipping temporal learning")
            return
            
        # Temporal decay learning with lag-based weighting  
        updates_made = 0
        print(f"   DEBUG: Starting temporal learning with {len(token_indices)} valid tokens")
        print(f"   DEBUG: max_lag={self.max_lag}, tau={self.tau}")
        
        for t in range(len(token_indices)):
            i = token_indices[t]
            token_i = self.neuron_to_word.get(i, f"neuron_{i}")
            print(f"     Processing token {t}: '{token_i}' (neuron {i})")
            
            max_lag_for_this_token = min(self.max_lag + 1, len(token_indices) - t)
            print(f"       Available lags: 1 to {max_lag_for_this_token-1}")
            
            # Look ahead within max_lag distance
            for lag in range(1, max_lag_for_this_token):
                u = t + lag
                j = token_indices[u]
                token_j = self.neuron_to_word.get(j, f"neuron_{j}")
                
                print(f"         lag={lag}: '{token_i}' -> '{token_j}' (positions {t}->{u})")
                
                if i == j:  # Skip self-connections
                    print(f"           SKIPPED: self-connection")
                    continue
                    
                # Calculate exponential decay weight
                decay_weight = exp(-lag / self.tau)
                print(f"           exp(-{lag}/{self.tau}) = {decay_weight:.6f}")
                
                # Apply boundary penalty if crossing a boundary
                penalty = self._get_boundary_penalty(boundaries, t, u)
                final_weight = decay_weight * penalty
                print(f"           penalty={penalty:.3f}, final_weight={final_weight:.6f}")
                
                # Forward learning: i -> j (token i influences token j)
                fwd_update = self.eta_fwd * final_weight
                print(f"           fwd_update = {self.eta_fwd} * {final_weight:.6f} = {fwd_update:.6f}")
                
                if fwd_update > 1e-12:  # Only update if significant
                    conn_key = (i, j) if i < j else (j, i)
                    if conn_key not in self.connections:
                        self.connections[conn_key] = 0.0
                    old_strength = self.connections[conn_key]
                    self.connections[conn_key] += fwd_update
                    print(f"           CONNECTION: {conn_key} strength {old_strength:.6f} -> {self.connections[conn_key]:.6f}")
                    updates_made += 1
                    self.debug_nonzero_updates += 1
                    
                    # Debug logging for first few updates
                    if len(self.debug_updates) < 10:
                        self.debug_updates.append({
                            'tokens': f"{token_i} -> {token_j}",
                            'lag': lag,
                            'decay_weight': decay_weight,
                            'penalty': penalty,
                            'final_weight': final_weight,
                            'update': fwd_update
                        })
                else:
                    print(f"           SKIPPED: update too small ({fwd_update:.8f})")
                    self.debug_zero_updates += 1
                
                # Backward/feedback learning: j -> i (optional, usually weaker)
                if self.eta_fb > 0:
                    back_update = self.eta_fb * final_weight
                    if back_update > 1e-12:
                        # Note: same conn_key since we store undirected connections
                        # but we could track directional weights separately if needed
                        self.connections[conn_key] += back_update
                        print(f"           FEEDBACK: added {back_update:.6f}")
                        updates_made += 1
        
        print(f"   Temporal updates: {updates_made} connections updated")
        print(f"   Debug: {self.debug_nonzero_updates} nonzero, {self.debug_zero_updates} zero updates")
        
        # Show first few debug updates
        if self.debug_updates:
            print("   Sample updates:")
            for i, update in enumerate(self.debug_updates[:3]):
                print(f"     {update['tokens']}: lag={update['lag']}, weight={update['final_weight']:.6f}, update={update['update']:.6f}")
    
    def _get_boundary_penalty(self, boundaries: Optional[List[int]], t: int, u: int) -> float:
        """Calculate boundary penalty for learning across sentence/turn boundaries."""
        if not boundaries:
            return 1.0
            
        # Check if there's a boundary between positions t and u
        for boundary_pos in boundaries:
            if t < boundary_pos <= u:
                return self.boundary_penalty
                
        return 1.0
    
    def process_speaker_text(self, speaker_name: str, text: str):
        """Process text using either temporal or windowed learning."""
        try:
            print(f"\n🧪 Processing text for {speaker_name}: {len(text.split())} words")
            print(f"   Learning method: {'TEMPORAL' if self.use_temporal_learning else 'WINDOWED'}")
            
            if self.use_temporal_learning:
                # Use temporal decay learning
                self.current_speaker = speaker_name
                
                # Simple tokenization (could be enhanced)
                words = text.lower().split()
                
                # For now, assume sentence boundaries at punctuation
                # Could be enhanced with proper sentence detection
                boundaries = []
                for i, word in enumerate(words):
                    if word.endswith(('.', '!', '?', ';')):
                        boundaries.append(i)
                
                # Process with temporal method
                self._process_tokens_temporal(words, boundaries)
                
                # Update activations for processed words
                for word in words:
                    if word in self.concept_neurons:
                        neuron_id = self.concept_neurons[word]
                        self.activations[neuron_id] = 1.0
                        
                # Sync mappings
                self._sync_network_mappings()
                
                # Calculate inertial mass for visualization compatibility
                self._calculate_inertial_mass()
                
            else:
                # Use parent windowed method
                super().process_speaker_text(speaker_name, text)
                
        except Exception as e:
            print(f"❌ Error in temporal processing: {e}")
            # Fallback to parent method
            super().process_speaker_text(speaker_name, text)
    
    def _calculate_inertial_mass(self):
        """Calculate inertial mass from temporal connections for visualization compatibility."""
        if not hasattr(self, 'inertial_mass'):
            self.inertial_mass = {}
            
        # Convert connection strengths to inertial mass
        for conn_key, strength in self.connections.items():
            # Use connection strength as mass proxy
            mass = strength * 10.0  # Scale for visibility
            self.inertial_mass[conn_key] = mass
            
        print(f"   Calculated inertial mass: {len(self.inertial_mass)} entries")
    
    def get_debug_summary(self) -> Dict:
        """Get summary of debugging information."""
        total_connections = len(self.connections)
        nonzero_connections = sum(1 for strength in self.connections.values() if strength > 1e-12)
        
        return {
            'learning_method': 'TEMPORAL' if self.use_temporal_learning else 'WINDOWED',
            'temporal_params': {
                'tau': self.tau,
                'eta_fwd': self.eta_fwd,
                'eta_fb': self.eta_fb,
                'max_lag': self.max_lag,
                'boundary_penalty': self.boundary_penalty
            } if self.use_temporal_learning else None,
            'updates': {
                'nonzero_updates': self.debug_nonzero_updates,
                'zero_updates': self.debug_zero_updates,
                'total_connections': total_connections,
                'nonzero_connections': nonzero_connections
            },
            'sample_updates': self.debug_updates[:10],
            'network_stats': {
                'concepts': len(self.concept_neurons),
                'activations': len(self.activations),
                'avg_connection_strength': np.mean(list(self.connections.values())) if self.connections else 0.0,
                'max_connection_strength': max(self.connections.values()) if self.connections else 0.0
            }
        }

if __name__ == "__main__":
    # Test both temporal and windowed methods
    print("🧪 Testing Temporal vs Windowed Learning\n")
    
    # Test data
    test_text = """
    Artificial intelligence research involves complex mathematical concepts like neural networks.
    Machine learning algorithms use computational linguistics and cognitive science principles.
    The neural network processes information through synaptic connections and Hebbian learning.
    """
    
    # Test 1: Windowed learning (control)
    print("=" * 60)
    print("TEST 1: WINDOWED LEARNING (CONTROL)")
    print("=" * 60)
    
    huey_windowed = HueyTemporalExperiment(
        max_neurons=100, 
        window_size=8,
        learning_rate=0.15,
        use_temporal_learning=False
    )
    huey_windowed.add_speaker("AI", ['i', 'me', 'my'], ['you', 'your'])
    huey_windowed.process_speaker_text("AI", test_text)
    
    windowed_debug = huey_windowed.get_debug_summary()
    print("\nWindowed Results:")
    print(f"  Concepts: {windowed_debug['network_stats']['concepts']}")
    print(f"  Connections: {windowed_debug['updates']['total_connections']}")
    print(f"  Nonzero connections: {windowed_debug['updates']['nonzero_connections']}")
    print(f"  Avg strength: {windowed_debug['network_stats']['avg_connection_strength']:.6f}")
    
    # Test 2: Temporal learning (experimental)
    print("\n" + "=" * 60)
    print("TEST 2: TEMPORAL LEARNING (EXPERIMENTAL)")
    print("=" * 60)
    
    huey_temporal = HueyTemporalExperiment(
        max_neurons=100,
        window_size=8, 
        learning_rate=0.15,
        use_temporal_learning=True,
        tau=3.0,           # Moderate decay
        eta_fwd=0.01,      # 1% learning rate
        eta_fb=0.002,      # 0.2% feedback rate
        boundary_penalty=0.5  # 50% penalty across boundaries
    )
    huey_temporal.add_speaker("AI", ['i', 'me', 'my'], ['you', 'your'])
    huey_temporal.process_speaker_text("AI", test_text)
    
    temporal_debug = huey_temporal.get_debug_summary()
    print("\nTemporal Results:")
    print(f"  Concepts: {temporal_debug['network_stats']['concepts']}")
    print(f"  Connections: {temporal_debug['updates']['total_connections']}")
    print(f"  Nonzero connections: {temporal_debug['updates']['nonzero_connections']}")
    print(f"  Avg strength: {temporal_debug['network_stats']['avg_connection_strength']:.6f}")
    print(f"  Nonzero updates: {temporal_debug['updates']['nonzero_updates']}")
    print(f"  Zero updates: {temporal_debug['updates']['zero_updates']}")
    
    print("\n" + "=" * 60)
    print("COMPARISON")  
    print("=" * 60)
    print(f"Windowed - Connections: {windowed_debug['updates']['total_connections']}, Avg: {windowed_debug['network_stats']['avg_connection_strength']:.6f}")
    print(f"Temporal - Connections: {temporal_debug['updates']['total_connections']}, Avg: {temporal_debug['network_stats']['avg_connection_strength']:.6f}")
    
    if temporal_debug['network_stats']['avg_connection_strength'] > 0:
        print("✅ Temporal learning produced non-zero connections!")
    else:
        print("❌ Temporal learning produced zero connections - debugging needed")
        
    print(f"\n🧪 Temporal Learning Experiment Complete!")