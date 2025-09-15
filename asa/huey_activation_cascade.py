"""
Interactive Activation Cascade Visualization for Huey Networks
============================================================

This module implements real-time activation cascades where users can:
1. Select input neurons (stimuli)
2. Select target neurons (responses) 
3. Watch activation flow through the network step-by-step
4. See concepts brighten in 3D visualization as activation spreads
5. View text stream of activation progression

Perfect for understanding how interviewer questions activate Feynman's responses!
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import math

@dataclass
class CascadeStep:
    """Single step in an activation cascade."""
    step_number: int
    activations: Dict[str, float]  # concept_name -> activation_level
    newly_activated: List[str]     # concepts that just became active this step
    activation_changes: Dict[str, float]  # concept_name -> change_from_previous
    total_energy: float           # sum of all activations
    description: str              # text description of what happened

class HueyActivationCascade:
    """
    Creates interactive activation cascades through Huey networks.
    
    Shows how activation flows from input neurons through the network,
    with real-time 3D visualization and text descriptions.
    """
    
    def __init__(self, network, activation_threshold=0.01, max_cascade_steps=100):
        """
        Initialize neurobiologically realistic cascade system.
        
        Args:
            network: Huey network with real connection data
            activation_threshold: Minimum activation to consider "active"
            max_cascade_steps: Maximum steps before stopping
        """
        self.network = network
        self.activation_threshold = activation_threshold
        self.max_cascade_steps = max_cascade_steps
        
        # Fixed neurobiological parameters (no personality variations)
        self.natural_decay_rate = 0.8  # Neurons retain 80% activation each step
        self.signal_scale = 0.1        # Signal strength scaling factor
        self.delay_constant = 2.0      # For delay = k / connection_strength
        self.dead_zone_theta = 1e-6    # Hard cutoff for tiny activations (prevent floating point artifacts)
        
        print(f"🧬 Neural Cascade System: decay={self.natural_decay_rate}, scale={self.signal_scale}, delay_k={self.delay_constant}")
        
        # Validate network has real data
        self._validate_network()
        
        # Build connection matrix and mappings
        self._build_network_matrices()
    
    def _validate_network(self):
        """Ensure network has real data - no fake fallbacks."""
        if not hasattr(self.network, 'neuron_to_word') or not self.network.neuron_to_word:
            raise ValueError("Network has no real concept mappings (neuron_to_word)")
        
        if not hasattr(self.network, 'word_to_neuron') or not self.network.word_to_neuron:
            raise ValueError("Network has no real word mappings (word_to_neuron)")
        
        # Check for connection data - prioritize inertial_mass (working connection source)
        self.connection_source = None
        if hasattr(self.network, 'inertial_mass') and self.network.inertial_mass:
            self.connection_source = self.network.inertial_mass
        elif hasattr(self.network, 'synaptic_strengths') and self.network.synaptic_strengths:
            self.connection_source = self.network.synaptic_strengths
        elif hasattr(self.network, 'connections') and self.network.connections:
            self.connection_source = self.network.connections  
        else:
            raise ValueError("Network has no real connection data")
        
        print(f"✅ Network validated: {len(self.network.neuron_to_word)} concepts, {len(self.connection_source)} connections")
    
    def _build_network_matrices(self):
        """Build connection matrices from real network data."""
        # Create concept mappings
        self.concepts = list(self.network.neuron_to_word.values())
        self.concept_to_id = {word: nid for nid, word in self.network.neuron_to_word.items()}
        self.id_to_concept = self.network.neuron_to_word
        
        # Create matrix index mappings  
        concept_ids = sorted(self.network.neuron_to_word.keys())
        self.id_to_matrix_idx = {concept_id: i for i, concept_id in enumerate(concept_ids)}
        self.matrix_idx_to_id = {i: concept_id for concept_id, i in self.id_to_matrix_idx.items()}
        
        # Build connection matrix
        n_concepts = len(concept_ids)
        self.connection_matrix = np.zeros((n_concepts, n_concepts))
        
        for (i, j), strength in self.connection_source.items():
            if i in self.id_to_matrix_idx and j in self.id_to_matrix_idx:
                idx_i = self.id_to_matrix_idx[i]
                idx_j = self.id_to_matrix_idx[j]
                self.connection_matrix[idx_j, idx_i] = strength  # j receives from i
        
        # Scale connections for cascade propagation (preserve relative strengths)
        max_connection = np.max(self.connection_matrix)
        if max_connection > 0:
            self.connection_matrix = self.connection_matrix / max_connection * 2.0  # Scale to [0, 2.0] range
        
        # DEBUG: Show connection weight statistics
        nonzero_connections = np.count_nonzero(self.connection_matrix)
        if nonzero_connections > 0:
            nonzero_weights = self.connection_matrix[self.connection_matrix > 0]
            weight_stats = {
                'min': np.min(nonzero_weights),
                'max': np.max(nonzero_weights), 
                'mean': np.mean(nonzero_weights),
                'std': np.std(nonzero_weights),
                'median': np.median(nonzero_weights)
            }
            print(f"✅ Built connection matrix: {n_concepts}x{n_concepts}, {nonzero_connections} connections")
            print(f"🔍 Weight distribution: min={weight_stats['min']:.6f}, max={weight_stats['max']:.6f}, mean={weight_stats['mean']:.6f}, std={weight_stats['std']:.6f}")
            
            # Show how many connections fall into different strength ranges
            strong = np.sum(nonzero_weights > weight_stats['mean'] + weight_stats['std'])
            medium = np.sum((nonzero_weights > weight_stats['mean']) & (nonzero_weights <= weight_stats['mean'] + weight_stats['std']))
            weak = np.sum(nonzero_weights <= weight_stats['mean'])
            print(f"🔍 Connection strength tiers: strong={strong}, medium={medium}, weak={weak}")
        else:
            print(f"⚠️ Built connection matrix: {n_concepts}x{n_concepts}, NO connections found!")
    
    
    def run_cascade(self, 
                   input_concepts: List[str],
                   target_concept: str,
                   max_steps: int = None,  # Will use self.max_cascade_steps if None
                   input_strength: float = 1.0) -> List[CascadeStep]:
        """
        Run an activation cascade from input concepts toward target.
        
        Args:
            input_concepts: List of concepts to initially activate
            target_concept: Target concept we're trying to activate
            max_steps: Maximum propagation steps
            input_strength: Initial activation strength for inputs
            
        Returns:
            List of CascadeStep objects showing progression
        """
        print(f"🚀 CASCADE STARTED: inputs={input_concepts} → target='{target_concept}' (network: {len(self.id_to_concept)} concepts)")
        
        # Validate inputs exist in network
        missing_concepts = []
        for concept in input_concepts + [target_concept]:
            if concept not in self.concept_to_id:
                missing_concepts.append(concept)
        
        if missing_concepts:
            raise ValueError(f"Concepts not found in network: {missing_concepts}")
        
        # Use instance max_steps if not provided
        if max_steps is None:
            max_steps = self.max_cascade_steps
        
        # Initialize activation state
        n_concepts = len(self.id_to_matrix_idx)
        current_activations = np.zeros(n_concepts)
        
        # Set initial input activations
        for concept in input_concepts:
            concept_id = self.concept_to_id[concept]
            idx = self.id_to_matrix_idx[concept_id]
            current_activations[idx] = input_strength
        
        cascade_steps = []
        previous_activations = np.zeros(n_concepts)
        
        for step in range(max_steps):
            # Record current state
            step_activations = {}
            newly_activated = []
            activation_changes = {}
            
            for idx, activation in enumerate(current_activations):
                concept_id = self.matrix_idx_to_id[idx]
                concept_name = self.id_to_concept[concept_id]
                step_activations[concept_name] = float(activation)
                
                # Track changes with higher sensitivity for science
                change = activation - previous_activations[idx] 
                if abs(change) > 0.0001:  # Capture smaller changes
                    activation_changes[concept_name] = float(change)
                
                # Track newly activated
                if (activation >= self.activation_threshold and 
                    previous_activations[idx] < self.activation_threshold):
                    newly_activated.append(concept_name)
            
            # Generate step description
            description = self._generate_step_description(
                step, newly_activated, activation_changes, target_concept
            )
            
            # Create cascade step
            cascade_step = CascadeStep(
                step_number=step,
                activations=step_activations,
                newly_activated=newly_activated,
                activation_changes=activation_changes,
                total_energy=float(np.sum(current_activations)),
                description=description
            )
            cascade_steps.append(cascade_step)
            
            # Check if target is activated
            target_id = self.concept_to_id[target_concept]
            target_idx = self.id_to_matrix_idx[target_id]
            target_activation = current_activations[target_idx]
            
            if target_activation >= self.activation_threshold:
                activation_percent = target_activation * 100
                print(f"🎯 Target '{target_concept}' activated! (step {step}, activation: {target_activation:.4f} = {activation_percent:.1f}%)")
            
            # TRUE SPREADING ACTIVATION CASCADE
            # Propagate activation one layer at a time through strongest connections
            previous_activations = current_activations.copy()
            
            if step == 0:
                # Step 0: Only input neurons are active
                current_activations = np.zeros(len(current_activations))
                for concept in input_concepts:
                    concept_id = self.concept_to_id[concept]
                    idx = self.id_to_matrix_idx[concept_id]
                    current_activations[idx] = input_strength
                print(f"🎯 CASCADE Step {step}: {len(input_concepts)} input neurons activated")
                
            else:
                # NEUROBIOLOGICALLY REALISTIC: FIRE-ONCE NEURONS WITH DELAYED SIGNAL PROPAGATION
                # Each neuron fires only once when it crosses threshold, then sends delayed signals
                
                # Apply natural neural decay (simple exponential decay)
                natural_decay_rate = 0.8  # Neurons retain 80% of activation each step
                decayed_activations = previous_activations * natural_decay_rate
                
                # Initialize signal processing
                if not hasattr(self, 'neuron_has_fired'):
                    self.neuron_has_fired = np.zeros(len(current_activations), dtype=bool)  # Track which neurons have fired
                if not hasattr(self, 'pending_signals'):
                    self.pending_signals = []  # List of (arrival_step, target_idx, signal_strength)
                
                # Identify newly firing neurons (cross threshold for first time)
                newly_firing = []
                for idx in range(len(current_activations)):
                    # Neuron fires if: above threshold AND hasn't fired before
                    if (previous_activations[idx] >= self.activation_threshold and 
                        not self.neuron_has_fired[idx]):
                        newly_firing.append(idx)
                        self.neuron_has_fired[idx] = True
                
                # For each newly firing neuron, create delayed signals for all its targets
                if newly_firing:
                    firing_names = [self.id_to_concept[self.matrix_idx_to_id[idx]] for idx in newly_firing[:5]]
                    print(f"🧠 Step {step}: {len(newly_firing)} neurons firing for first time: {firing_names}{'...' if len(newly_firing) > 5 else ''}")
                    
                    for source_idx in newly_firing:
                        source_strength = previous_activations[source_idx]
                        
                        # Get all outgoing connections
                        outgoing_strengths = self.connection_matrix[:, source_idx]
                        connected_targets = np.where(outgoing_strengths > 0)[0]
                        
                        # Create delayed signal for each target
                        for target_idx in connected_targets:
                            connection_strength = outgoing_strengths[target_idx]
                            
                            # Calculate when signal will arrive
                            transmission_delay = max(1, int(self.delay_constant / connection_strength))  # Min 1 step
                            arrival_step = step + transmission_delay
                            
                            # Calculate signal strength - simple and natural
                            # Strong connections carry more signal, proportional to source strength
                            signal_strength = connection_strength * source_strength * self.signal_scale
                            
                            # Schedule signal arrival
                            self.pending_signals.append((arrival_step, target_idx, signal_strength))
                
                # Process signals arriving at this step
                arriving_signals = []
                remaining_signals = []
                
                for arrival_step, target_idx, signal_strength in self.pending_signals:
                    if arrival_step == step:
                        arriving_signals.append((target_idx, signal_strength))
                    else:
                        remaining_signals.append((arrival_step, target_idx, signal_strength))
                
                self.pending_signals = remaining_signals  # Keep future signals
                
                # Apply arriving signals
                new_activations = np.zeros(len(current_activations))
                if arriving_signals:
                    newly_activated_names = []
                    for target_idx, signal_strength in arriving_signals:
                        new_activations[target_idx] += signal_strength
                        
                        # Track newly activated
                        if (new_activations[target_idx] > self.activation_threshold and
                            previous_activations[target_idx] <= self.activation_threshold):
                            concept_id = self.matrix_idx_to_id[target_idx]
                            concept_name = self.id_to_concept[concept_id]
                            newly_activated_names.append(concept_name)
                    
                    print(f"🌊 Step {step}: {len(arriving_signals)} signals arrived, {len(newly_activated_names)} new activations: {newly_activated_names[:3]}{'...' if len(newly_activated_names) > 3 else ''}")
                else:
                    print(f"⏱️  Step {step}: No signal arrivals")
                
                # Simple threshold activation - signals either activate neurons or they don't
                # No complex logistic function needed
                new_activations = np.maximum(new_activations, 0.0)  # Just ensure non-negative
                
                # Combine: decayed existing + new signal arrivals
                current_activations = decayed_activations + new_activations
                
                # Maintain input strength for first few steps
                if step < 3:
                    for concept in input_concepts:
                        concept_id = self.concept_to_id[concept]
                        idx = self.id_to_matrix_idx[concept_id]
                        current_activations[idx] = max(current_activations[idx], input_strength * (0.9 ** step))
            
            # Ensure non-negative activations
            current_activations = np.maximum(current_activations, 0.0)
            
            # Apply dead zone for finite-time extinction
            current_activations[current_activations < self.dead_zone_theta] = 0.0
            
            # DEBUG: Check for any negative activations (should never happen)
            negative_count = np.sum(current_activations < 0)
            if negative_count > 0:
                min_activation = np.min(current_activations)
                print(f"🚨 BUG: {negative_count} negative activations found! Min: {min_activation:.6f}")
                # Force fix any negatives
                current_activations = np.maximum(current_activations, 0.0)
            
            # Check convergence: TRUE REST means network energy approaches zero
            # SKIP convergence check for step 0 (initial setup only)
            if step > 0:
                change = np.mean(np.abs(current_activations - previous_activations))
                total_energy = np.sum(current_activations)
                max_activation = np.max(current_activations)
                
                # DEBUG: Always show first few step progressions
                if step <= 3:
                    active_count = np.sum(current_activations > self.activation_threshold)
                    print(f"🔬 Step {step}: change={change:.6f}, energy={total_energy:.6f}, active={active_count}, max_activation={max_activation:.6f}")
                
                # TRUE NEURAL REST: very low total energy and max activation
                if total_energy < 0.001 and max_activation < 0.001:
                    print(f"💤 TRUE NEURAL REST achieved at step {step} (total energy: {total_energy:.6f}, max activation: {max_activation:.6f})")
                    break
                elif change < 0.0001:  # Traditional convergence
                    print(f"🔄 Activation converged at step {step} (change: {change:.6f}, energy: {total_energy:.6f})")
                    break
                elif change < 0.001:
                    print(f"🔬 Fine-tuning activations at step {step} (change: {change:.6f}, energy: {total_energy:.6f})")
            else:
                # Step 0: Just show initial setup
                total_energy = np.sum(current_activations)
                active_count = np.sum(current_activations > self.activation_threshold)
                print(f"🔬 Step {step}: Initial setup - energy={total_energy:.6f}, active={active_count}")
        
        # Scientific summary
        final_target_activation = current_activations[self.id_to_matrix_idx[self.concept_to_id[target_concept]]]
        total_active = sum(1 for a in current_activations if a >= self.activation_threshold)
        total_energy = float(np.sum(current_activations))
        
        print(f"🧬 CASCADE ANALYSIS:")
        print(f"   Final target activation: {final_target_activation:.6f} ({final_target_activation*100:.3f}%)")
        print(f"   Total concepts activated: {total_active}/{len(current_activations)}")
        print(f"   Total network energy: {total_energy:.6f}")
        print(f"   Max final activation: {np.max(current_activations):.6f}")
        print(f"   Min final activation: {np.min(current_activations):.6f}")
        print(f"   Concepts at true zero: {np.sum(current_activations == 0.0)}/{len(current_activations)}")
        print(f"   Steps to convergence: {len(cascade_steps)}")
        
        # ChatGPT5's spectral radius analysis
        spectral_analysis = {}
        energies = [step.total_energy for step in cascade_steps]
        if len(energies) > 5:  # Need enough points for analysis
            # Fit exponential decay: E_t = E_0 * rho_eff^t
            log_energies = [np.log(max(e, 1e-10)) for e in energies]  # Avoid log(0)
            steps_range = list(range(len(log_energies)))
            
            # Linear fit to log(E_t) vs t
            if len(steps_range) > 1:
                slope = (log_energies[-1] - log_energies[0]) / (steps_range[-1] - steps_range[0])
                rho_eff = np.exp(slope)
                
                spectral_analysis['rho_eff'] = rho_eff
                spectral_analysis['slope'] = slope
                spectral_analysis['analysis_available'] = True
                
                print(f"📊 SPECTRAL RADIUS ANALYSIS (ChatGPT5 method):")
                print(f"   ρ_eff (effective loop gain): {rho_eff:.4f}")
                
                if rho_eff < 1.0:
                    # Predict time to various thresholds
                    E0 = energies[0]
                    if E0 > 0:
                        time_to_1e3 = np.log(1e-3 / E0) / slope if slope < 0 else float('inf')
                        time_to_1e6 = np.log(1e-6 / E0) / slope if slope < 0 else float('inf')
                        spectral_analysis['time_to_1e3'] = time_to_1e3
                        spectral_analysis['time_to_1e6'] = time_to_1e6
                        spectral_analysis['E0'] = E0
                        
                        print(f"   Predicted time to 10^-3: {time_to_1e3:.1f} steps")
                        print(f"   Predicted time to 10^-6: {time_to_1e6:.1f} steps")
                        
                        if time_to_1e3 < 100:
                            spectral_analysis['decay_prediction'] = "quick"
                            print(f"   ✅ Network will die out reasonably quickly")
                        else:
                            spectral_analysis['decay_prediction'] = "slow"
                            print(f"   ⏰ Network will persist for a long time (normal behavior)")
                else:
                    spectral_analysis['decay_prediction'] = "no_decay"
                    print(f"   ⚠️  Network may not decay (ρ_eff ≥ 1.0)")
        else:
            spectral_analysis['analysis_available'] = False
            spectral_analysis['reason'] = "insufficient_steps"
            print(f"📊 SPECTRAL RADIUS ANALYSIS: Need more steps for analysis")
        
        # Store analysis in final cascade step for retrieval
        if cascade_steps:
            cascade_steps[-1].spectral_analysis = spectral_analysis
        
        return cascade_steps
    
    def _generate_step_description(self, 
                                 step: int, 
                                 newly_activated: List[str],
                                 changes: Dict[str, float],
                                 target_concept: str) -> str:
        """Generate human-readable description of cascade step."""
        if step == 0:
            return "🎯 Initial activation: Input neurons fired"
        
        descriptions = []
        
        if newly_activated:
            if target_concept in newly_activated:
                descriptions.append(f"🏁 TARGET REACHED: '{target_concept}' activated!")
            
            other_activated = [c for c in newly_activated if c != target_concept]
            if other_activated:
                if len(other_activated) == 1:
                    descriptions.append(f"💡 '{other_activated[0]}' became active")
                else:
                    descriptions.append(f"💡 {len(other_activated)} concepts activated: {', '.join(other_activated[:3])}{'...' if len(other_activated) > 3 else ''}")
        
        # Find strongest changes (more sensitive for scientific analysis)
        strong_changes = {k: v for k, v in changes.items() if abs(v) > 0.01}  # More sensitive threshold
        if strong_changes:
            strongest = max(strong_changes.items(), key=lambda x: abs(x[1]))
            if strongest[1] > 0:
                descriptions.append(f"📈 '{strongest[0]}' rising {strongest[1]:+.3f}")  # Clearer: change, not absolute
            else:
                descriptions.append(f"📉 '{strongest[0]}' fading {strongest[1]:.3f}")  # Clearer: shows decay
            
            # Show secondary changes for scientific detail
            sorted_changes = sorted(strong_changes.items(), key=lambda x: abs(x[1]), reverse=True)
            if len(sorted_changes) > 1:
                secondary = sorted_changes[1]
                if abs(secondary[1]) > 0.05:  # Significant secondary change
                    if secondary[1] > 0:
                        descriptions.append(f"🔬 '{secondary[0]}' rising {secondary[1]:+.3f}")
                    else:
                        descriptions.append(f"🔬 '{secondary[0]}' fading {secondary[1]:.3f}")
        
        if not descriptions:
            descriptions.append("🔄 Activation spreading through network...")
        
        return f"Step {step}: " + " | ".join(descriptions)
    
    def create_cascade_visualization(self, 
                                   cascade_steps: List[CascadeStep],
                                   concept_positions: Dict[str, Tuple[float, float, float]]) -> go.Figure:
        """
        Create animated 3D visualization of activation cascade.
        
        Args:
            cascade_steps: Steps from run_cascade()
            concept_positions: Dict of concept_name -> (x, y, z) positions
            
        Returns:
            Plotly figure with animation frames
        """
        if not concept_positions:
            raise ValueError("Need real 3D concept positions - no fake coordinates!")
        
        fig = go.Figure()
        
        # Create frames for animation
        frames = []
        
        for step_idx, step in enumerate(cascade_steps):
            # Get concept data for this step
            concepts_data = []
            active_count = 0
            
            # FIXED: Use concept_positions keys directly (numeric IDs) and look up activations by concept name
            # This fixes the coordinate lookup issue where positions use numeric IDs as keys
            
            # DEBUG: Check coordinate-activation mismatch in large networks
            if step_idx == 1 and len(concept_positions) > 100:
                total_positions = len(concept_positions)
                total_activations = len([a for a in step.activations.values() if a > 0])
                print(f"🔍 LARGE NETWORK DEBUG: {total_positions} coordinate positions, {total_activations} non-zero activations")
                
                # Check for ID mismatches
                position_ids = set(concept_positions.keys())
                activation_concept_names = set(step.activations.keys())
                network_concept_names = set(self.id_to_concept.values())
                
                print(f"🔍 MISMATCH CHECK: Position IDs type: {type(list(position_ids)[0]) if position_ids else 'None'}")
                print(f"🔍 MISMATCH CHECK: Network concept IDs: {len(self.id_to_concept)} mappings")
                
                # Show sample of ID-to-name mappings
                sample_mappings = list(self.id_to_concept.items())[:3]
                print(f"🔍 SAMPLE MAPPINGS: {sample_mappings}")
            
            for concept_id, position in concept_positions.items():
                # Get concept name from ID
                concept_name = self.id_to_concept.get(concept_id, str(concept_id))
                
                # Get activation for this concept (default to 0 if not activated)
                activation = step.activations.get(concept_name, 0.0)
                
                # DEBUG: Track coordinate-activation mismatches
                if step_idx == 1 and len(concept_positions) > 100 and activation == 0.0:
                    if concept_name in step.activations and step.activations[concept_name] > 0:
                        print(f"🔍 MISMATCH: '{concept_name}' has activation {step.activations[concept_name]:.4f} but lookup returned 0.0")
                
                if activation > 0.0:  # ANY non-zero activation
                    active_count += 1
                
                # Color and size based on activation - ULTRA SENSITIVE TO CHANGES
                if activation > 0.0:  # Show ANY non-zero activation
                    # Much more sensitive color mapping - amplify 50x and use power scaling
                    base_intensity = min(1.0, activation * 50)  # 5x more sensitive than before
                    # Power scaling makes small changes more visible
                    intensity = base_intensity ** 0.7  # Power less than 1 emphasizes small values
                    
                    # More dynamic color range - red for high, yellow for medium, green for low
                    if intensity > 0.7:  # High activation: red-orange
                        red_val = 255
                        green_val = int(255 * (1 - intensity) * 2)  # Less green for high activation
                    elif intensity > 0.3:  # Medium activation: yellow-orange  
                        red_val = 255
                        green_val = int(255 * (0.3 + intensity * 0.7))  # More yellow
                    else:  # Low activation: green-yellow
                        red_val = int(255 * intensity * 3)  # Some red for visibility
                        green_val = 255
                    
                    color = f'rgba({red_val}, {green_val}, 0, 1.0)'
                    size = 6 + int(intensity * 10)  # Size scales with enhanced intensity
                    text_color = f'rgba({min(255, red_val + 80)}, {min(255, green_val + 80)}, 100, 1.0)'
                    text_size = 8 + int(intensity * 6)
                    
                    # DEBUG: Log successful rendering for any significant activation
                    if step_idx == 1 and activation > 0.01:  # Show any meaningful activation
                        print(f"🔍 RENDER DEBUG: '{concept_name}' (ID {concept_id}) -> activation={activation:.4f}, color={color}, size={size}")
                    
                    # SPECIAL DEBUG: Always show target concept
                    if concept_name == 'feynman' and step_idx <= 3:
                        print(f"🎯 FEYNMAN RENDER: Step {step_idx} -> activation={activation:.6f}, color={color}, size={size}")
                    
                    # DEBUG: Show ANY concept with activation in step 1
                    if step_idx == 1 and activation > 0.0001:
                        print(f"📍 STEP 1 ACTIVE: '{concept_name}' -> {activation:.6f}")
                else:
                    color = 'rgba(150, 150, 150, 0.6)'  # Brighter gray for inactive
                    size = 4  # Even smaller inactive size: reduced from 6 to 4
                    text_color = 'rgba(180, 180, 180, 0.7)'  # Dim text for inactive
                    text_size = 7  # Smaller inactive text: reduced from 8 to 7
                
                concepts_data.append({
                    'x': position[0],
                    'y': position[1], 
                    'z': position[2],
                    'text': concept_name,  # Display concept name as text
                    'activation': activation,
                    'color': color,
                    'size': size,
                    'text_color': text_color,
                    'text_size': text_size
                })
            
            # DEBUG: Print active concept count for this step
            if step_idx == 1:  # Check step 1 where we know concepts are active
                print(f"🔍 VISUAL DEBUG Step {step_idx}: {active_count} active concepts out of {len(concept_positions)} total")
                print(f"🔍 VISUAL DEBUG: concepts_data has {len(concepts_data)} entries")
                
                # Show activation ranges for debugging
                all_activations = list(step.activations.values())
                if all_activations:
                    max_act = max(all_activations)
                    min_act = min(all_activations)
                    nonzero_acts = [a for a in all_activations if a > 0]
                    print(f"🔍 ACTIVATION RANGE: max={max_act:.6f}, min={min_act:.6f}, nonzero_count={len(nonzero_acts)}")
                    if nonzero_acts:
                        print(f"🔍 NONZERO RANGE: max={max(nonzero_acts):.6f}, min={min(nonzero_acts):.6f}")
                else:
                    print(f"🔍 NO ACTIVATIONS FOUND in step.activations")
                
                # Show sample of concepts_data being passed to Plotly
                if concepts_data:
                    sample_concepts = concepts_data[:3]
                    print(f"🔍 PLOTLY DEBUG: Sample concepts_data: {sample_concepts}")
                else:
                    print(f"🔍 PLOTLY DEBUG: concepts_data is EMPTY - this is why nothing shows!")
            
            # Create scatter plot for this frame with dynamic text styling
            frame_data = go.Scatter3d(
                x=[c['x'] for c in concepts_data],
                y=[c['y'] for c in concepts_data],
                z=[c['z'] for c in concepts_data],
                mode='markers+text',
                marker=dict(
                    size=[c['size'] for c in concepts_data],
                    color=[c['color'] for c in concepts_data],
                    opacity=0.9  # Slightly more opaque
                ),
                text=[c['text'] for c in concepts_data],
                textposition="top center",
                textfont=dict(
                    size=[c['text_size'] for c in concepts_data],
                    color=[c['text_color'] for c in concepts_data],
                    family="Arial Black"  # Bold font for better visibility
                ),
                hovertemplate='<b>%{text}</b><br>' +
                            'Activation: %{customdata:.3f}<br>' +
                            '<extra></extra>',
                customdata=[c['activation'] for c in concepts_data],
                name=f'Step {step_idx}'
            )
            
            # DEBUG: Print final frame data info
            if step_idx == 1:
                print(f"🔍 FRAME DEBUG: x data has {len(frame_data.x)} points")
                print(f"🔍 FRAME DEBUG: color data has {len(frame_data.marker.color)} points")
                if frame_data.x:
                    print(f"🔍 FRAME DEBUG: Sample x,y,z: ({frame_data.x[0]:.3f}, {frame_data.y[0]:.3f}, {frame_data.z[0]:.3f})")
                    print(f"🔍 FRAME DEBUG: Sample color: {frame_data.marker.color[0]}")
                    print(f"🔍 FRAME DEBUG: Sample size: {frame_data.marker.size[0]}")
                else:
                    print(f"🔍 FRAME DEBUG: frame_data has NO POINTS - this is the root issue!")
            
            frames.append(go.Frame(
                data=[frame_data],
                name=f'Step {step_idx}',
                layout=go.Layout(
                    title=f'Activation Cascade - {step.description}',
                    annotations=[
                        dict(
                            text=f"Total Energy: {step.total_energy:.2f}",
                            x=0.02, y=0.98,
                            xref="paper", yref="paper",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.8)"
                        )
                    ]
                )
            ))
        
        # Set up initial frame
        if frames:
            fig.add_trace(frames[0].data[0])
        
        # Add animation frames
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title="🔄 Interactive Activation Cascade",
            width=1380,  # Expanded by ~15% from 1200
            height=920,  # Expanded by ~15% from 800
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2', 
                zaxis_title='Dimension 3',
                bgcolor='rgba(0,0,0,0.9)',  # Much darker background for contrast
                aspectratio=dict(x=1, y=1, z=0.8),
                camera=dict(
                    eye=dict(x=1.3, y=1.3, z=1.3),  # Zoomed in 15% more: 1.5 → 1.3
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                # Allow user interaction and preserve camera state during animation
                dragmode='orbit'
            ),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                        }],
                        'label': '▶️ Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': '⏸️ Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f'Step {i}'], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate', 
                            'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                        }],
                        'label': f'Step {i}',
                        'method': 'animate'
                    } for i in range(len(frames))
                ],
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Step: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            }]
        )
        
        return fig
    
    def get_available_concepts(self) -> List[str]:
        """Get list of available concepts for selection.""" 
        return sorted(self.concepts)
    
    def suggest_inputs_for_target(self, target_concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest input concepts most likely to activate the target.
        
        Args:
            target_concept: Target concept to activate
            top_k: Number of suggestions to return
            
        Returns:
            List of (concept_name, connection_strength) tuples
        """
        if target_concept not in self.concept_to_id:
            return []
        
        target_id = self.concept_to_id[target_concept]
        target_idx = self.id_to_matrix_idx[target_id]
        
        # Find strongest inputs to target
        input_strengths = []
        for i, strength in enumerate(self.connection_matrix[target_idx, :]):
            if strength > 0.001 and i != target_idx:
                concept_id = self.matrix_idx_to_id[i]
                concept_name = self.id_to_concept[concept_id]
                input_strengths.append((concept_name, float(strength)))
        
        # Sort by strength and return top K
        input_strengths.sort(key=lambda x: x[1], reverse=True)
        return input_strengths[:top_k]


def create_cascade_interface(huey_network):
    """Create activation cascade interface with standard scientific parameters."""
    
    # Simple adaptive threshold based on network size
    network_size = len(huey_network.neuron_to_word) if hasattr(huey_network, 'neuron_to_word') else 50
    
    # Larger networks need lower thresholds (dilution effect)
    if network_size > 200:
        activation_threshold = 0.005
    elif network_size > 100:
        activation_threshold = 0.008  
    else:
        activation_threshold = 0.01
    
    print(f"🔧 Network size: {network_size} concepts → activation threshold: {activation_threshold:.3f}")
    
    return HueyActivationCascade(huey_network, activation_threshold=activation_threshold, max_cascade_steps=100)

def get_personality_settings(personality_type):
    """Get cascade parameters for different cognitive personalities.
    
    Based on ChatGPT5's spectral radius analysis, different ρ_eff ranges serve different purposes:
    - 0.8-0.9: Quick decisive thinking (taciturn)
    - 0.9-0.95: Balanced cognition  
    - 0.95-0.999: Rich associative memory (expansive)
    - >1.0: Persistent attractor states (enthusiastic/analytical)
    """
    personalities = {
        "taciturn": {
            "activation_threshold": 0.02,
            "target_rho_eff": 0.85,        # Fast quench - decisions made quickly
            "target_cycles": 25,            # Want cascades done in ~25 steps
            "target_epsilon": 1e-3,         # Decay to 0.1% of initial energy
            "leak_lambda": 0.15,            # Base decay rate
            "dead_zone_theta": 1e-4,        # Hard cutoff for tiny activations
            "max_cascade_steps": 50         # Conservative limit
        },
        "balanced": {
            "activation_threshold": 0.01,
            "target_rho_eff": 0.90,        # Moderate memory and exploration
            "target_cycles": 50,            # Standard cascade duration
            "target_epsilon": 1e-3,         # Standard threshold
            "leak_lambda": 0.10,            # Moderate base decay
            "dead_zone_theta": 5e-5,        # Allow smaller activations
            "max_cascade_steps": 100        # Standard limit
        },
        "expansive": {
            "activation_threshold": 0.005,
            "target_rho_eff": 0.95,        # Rich fading memory - lots of connections
            "target_cycles": 75,            # Longer cascades for exploration
            "target_epsilon": 1e-3,         # Standard threshold
            "leak_lambda": 0.05,            # Lower base decay
            "dead_zone_theta": 1e-5,        # Very small hard cutoff
            "max_cascade_steps": 150        # Allow longer exploration
        },
        "enthusiastic": {
            "activation_threshold": 0.001,
            "target_rho_eff": 0.98,        # Near edge-of-stability
            "target_cycles": 100,           # Very long cascades
            "target_epsilon": 1e-4,         # Tighter convergence criteria
            "leak_lambda": 0.02,            # Minimal base decay
            "dead_zone_theta": 5e-6,        # Almost no hard cutoff
            "max_cascade_steps": 200        # Very long exploration
        },
        "analytical": {
            "activation_threshold": 0.01,
            "target_rho_eff": 0.93,        # Systematic exploration with good memory
            "target_cycles": 60,            # Thorough analysis time
            "target_epsilon": 1e-4,         # Precise convergence
            "leak_lambda": 0.08,            # Controlled base decay
            "dead_zone_theta": 1e-5,        # Small hard cutoff
            "max_cascade_steps": 120        # Adequate time for analysis
        },
        "persistent": {
            "activation_threshold": 0.001,
            "target_rho_eff": 1.02,        # Deliberately > 1.0 for attractor behavior
            "target_cycles": 200,           # Very long or indefinite
            "target_epsilon": 1e-6,         # Very tight threshold (may never reach)
            "leak_lambda": 0.01,            # Minimal decay - let connections dominate
            "dead_zone_theta": 1e-6,        # Tiny hard cutoff
            "max_cascade_steps": 300        # Allow very long persistence
        }
    }
    
    return personalities.get(personality_type, personalities["balanced"])