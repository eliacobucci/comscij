"""
Bidirectional Network Propagation for Huey
==========================================

This module implements bidirectional propagation capabilities for Huey's Hebbian networks,
allowing both forward (input→output) and backward (output→input) propagation through 
the same connection structure.

Key Features:
- Forward propagation: excite inputs, see what outputs activate
- Backward propagation: excite outputs, see what inputs would have caused them
- Maintains directional connection semantics
- Supports multiple propagation modes (single-step, iterative, convergent)
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
import copy

@dataclass 
class PropagationResult:
    """Results from a bidirectional propagation operation."""
    final_activations: Dict[str, float]
    propagation_steps: List[Dict[str, float]]  
    convergence_achieved: bool
    steps_taken: int
    initial_activations: Dict[str, float]
    direction: str  # "forward", "backward", or "bidirectional"
    activated_concepts: List[Tuple[str, float]]  # (concept, final_activation)

class HueyBidirectionalPropagator:
    """
    Implements bidirectional propagation for Huey networks.
    
    This allows us to:
    1. Forward: Start with input concepts, see what outputs emerge
    2. Backward: Start with output concepts, see what inputs would cause them
    3. Bidirectional: Allow flow in both directions simultaneously
    """
    
    def __init__(self, network, activation_threshold=0.01, max_steps=10):
        """
        Initialize the bidirectional propagator.
        
        Args:
            network: Huey network instance with connections and concepts
            activation_threshold: Minimum activation to consider significant
            max_steps: Maximum propagation steps before stopping
        """
        self.network = network
        self.activation_threshold = activation_threshold
        self.max_steps = max_steps
        self.bias = getattr(network, 'bias', -2.0)
        
        # Build forward and backward connection matrices
        self._build_directional_matrices()
    
    def _build_directional_matrices(self):
        """Build separate connection matrices for forward and backward propagation."""
        # REQUIRE REAL NETWORK DATA - NO FAKE FALLBACKS
        if not hasattr(self.network, 'neuron_to_word') or not self.network.neuron_to_word:
            raise ValueError("Network has no neuron_to_word mapping - no real concepts found")
            
        self.concept_to_id = {word: idx for idx, word in self.network.neuron_to_word.items()}
        self.id_to_concept = self.network.neuron_to_word
        
        # Verify we have actual concepts, not empty mappings
        if not self.concept_to_id:
            raise ValueError("Network neuron_to_word mapping is empty - no concepts to propagate through")
        
        # Build connection matrices
        n_concepts = len(self.id_to_concept)
        self.forward_matrix = np.zeros((n_concepts, n_concepts))
        self.backward_matrix = np.zeros((n_concepts, n_concepts))
        
        # Create index mapping
        id_list = sorted(self.id_to_concept.keys())
        self.id_to_matrix_idx = {concept_id: i for i, concept_id in enumerate(id_list)}
        self.matrix_idx_to_id = {i: concept_id for concept_id, i in self.id_to_matrix_idx.items()}
        
        # REQUIRE REAL CONNECTION DATA - NO FAKE FALLBACKS
        connection_source = None
        if hasattr(self.network, 'synaptic_strengths') and self.network.synaptic_strengths:
            connection_source = self.network.synaptic_strengths
        elif hasattr(self.network, 'connections') and self.network.connections:
            connection_source = self.network.connections
        elif hasattr(self.network, 'inertial_mass') and self.network.inertial_mass:
            connection_source = self.network.inertial_mass
        
        if not connection_source:
            raise ValueError("Network has no connection data (synaptic_strengths, connections, or inertial_mass) - cannot build propagation matrices")
        
        if connection_source:
            for (i, j), strength in connection_source.items():
                if i in self.id_to_matrix_idx and j in self.id_to_matrix_idx:
                    idx_i = self.id_to_matrix_idx[i]
                    idx_j = self.id_to_matrix_idx[j]
                    
                    # Forward: i → j (how i influences j)
                    self.forward_matrix[idx_j, idx_i] = strength  # j receives from i
                    
                    # Backward: j ← i (how j would have influenced i)  
                    self.backward_matrix[idx_i, idx_j] = strength  # i receives from j
    
    def propagate_forward(self, 
                         input_concepts: Union[str, List[str], Dict[str, float]], 
                         steps: int = 5,
                         convergence_threshold: float = 0.001) -> PropagationResult:
        """
        Forward propagation: excite input concepts, see what outputs emerge.
        
        Args:
            input_concepts: Concepts to initially activate
            steps: Number of propagation steps
            convergence_threshold: Stop when changes are below this threshold
            
        Returns:
            PropagationResult with final activations and propagation history
        """
        initial_activations = self._prepare_initial_activations(input_concepts)
        return self._propagate(initial_activations, "forward", steps, convergence_threshold)
    
    def propagate_backward(self, 
                          output_concepts: Union[str, List[str], Dict[str, float]],
                          steps: int = 5, 
                          convergence_threshold: float = 0.001) -> PropagationResult:
        """
        Backward propagation: excite output concepts, see what inputs would cause them.
        
        Args:
            output_concepts: Concepts to initially activate
            steps: Number of propagation steps  
            convergence_threshold: Stop when changes are below this threshold
            
        Returns:
            PropagationResult with final activations and propagation history
        """
        initial_activations = self._prepare_initial_activations(output_concepts)
        return self._propagate(initial_activations, "backward", steps, convergence_threshold)
    
    def propagate_bidirectional(self,
                               concepts: Union[str, List[str], Dict[str, float]],
                               steps: int = 5,
                               convergence_threshold: float = 0.001) -> PropagationResult:
        """
        Bidirectional propagation: allow activation to flow in both directions.
        
        Args:
            concepts: Concepts to initially activate
            steps: Number of propagation steps
            convergence_threshold: Stop when changes are below this threshold
            
        Returns:
            PropagationResult with final activations and propagation history
        """
        initial_activations = self._prepare_initial_activations(concepts)
        return self._propagate(initial_activations, "bidirectional", steps, convergence_threshold)
    
    def _prepare_initial_activations(self, 
                                   concepts: Union[str, List[str], Dict[str, float]]) -> Dict[int, float]:
        """Convert concept specifications into initial activation dictionary."""
        activations = {concept_id: 0.0 for concept_id in self.id_to_concept.keys()}
        concepts_found = 0
        
        if isinstance(concepts, str):
            # Single concept name
            for concept_id, concept_name in self.id_to_concept.items():
                if concepts.lower() in concept_name.lower():
                    activations[concept_id] = 1.0
                    concepts_found += 1
                    break
        
        elif isinstance(concepts, list):
            # List of concept names
            for concept in concepts:
                for concept_id, concept_name in self.id_to_concept.items():
                    if concept.lower() in concept_name.lower():
                        activations[concept_id] = 1.0
                        concepts_found += 1
                        break
        
        elif isinstance(concepts, dict):
            # Dictionary of concept_name: activation_level
            for concept, activation in concepts.items():
                for concept_id, concept_name in self.id_to_concept.items():
                    if concept.lower() in concept_name.lower():
                        activations[concept_id] = float(activation)
                        concepts_found += 1
                        break
        
        if concepts_found == 0:
            raise ValueError(f"No matching concepts found in network for: {concepts}")
        
        return activations
    
    def _propagate(self, 
                  initial_activations: Dict[int, float],
                  direction: str,
                  steps: int,
                  convergence_threshold: float) -> PropagationResult:
        """Core propagation engine."""
        
        # Convert to matrix form
        n_concepts = len(self.id_to_concept)
        current_activations = np.zeros(n_concepts)
        
        for concept_id, activation in initial_activations.items():
            if concept_id in self.id_to_matrix_idx:
                idx = self.id_to_matrix_idx[concept_id]
                current_activations[idx] = activation
        
        # Choose connection matrix
        if direction == "forward":
            connection_matrix = self.forward_matrix
        elif direction == "backward": 
            connection_matrix = self.backward_matrix
        else:  # bidirectional
            connection_matrix = (self.forward_matrix + self.backward_matrix) / 2.0
        
        # Propagation loop
        propagation_steps = []
        convergence_achieved = False
        
        for step in range(steps):
            # Store current state
            step_activations = {}
            for idx, activation in enumerate(current_activations):
                concept_id = self.matrix_idx_to_id[idx]
                concept_name = self.id_to_concept[concept_id]
                step_activations[concept_name] = float(activation)
            propagation_steps.append(step_activations)
            
            # Calculate new activations
            weighted_sums = np.dot(connection_matrix, current_activations) + self.bias
            new_activations = 1.0 / (1.0 + np.exp(-np.clip(weighted_sums, -50, 50)))
            
            # Preserve initial activations (clamped inputs)
            for concept_id, initial_activation in initial_activations.items():
                if initial_activation > 0.0 and concept_id in self.id_to_matrix_idx:
                    idx = self.id_to_matrix_idx[concept_id]
                    new_activations[idx] = max(new_activations[idx], initial_activation * 0.8)
            
            # Check convergence
            change = np.mean(np.abs(new_activations - current_activations))
            if change < convergence_threshold:
                convergence_achieved = True
                break
                
            current_activations = new_activations
        
        # Build final result
        final_activations = {}
        activated_concepts = []
        
        for idx, activation in enumerate(current_activations):
            concept_id = self.matrix_idx_to_id[idx]
            concept_name = self.id_to_concept[concept_id]
            final_activations[concept_name] = float(activation)
            
            if activation >= self.activation_threshold:
                activated_concepts.append((concept_name, float(activation)))
        
        # Sort by activation strength
        activated_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Convert initial activations back to concept names
        initial_activations_named = {}
        for concept_id, activation in initial_activations.items():
            concept_name = self.id_to_concept[concept_id]
            initial_activations_named[concept_name] = activation
        
        return PropagationResult(
            final_activations=final_activations,
            propagation_steps=propagation_steps,
            convergence_achieved=convergence_achieved,
            steps_taken=len(propagation_steps),
            initial_activations=initial_activations_named,
            direction=direction,
            activated_concepts=activated_concepts
        )
    
    def analyze_causality(self, 
                         effect_concepts: Union[str, List[str]], 
                         potential_causes: Optional[List[str]] = None,
                         top_k: int = 10) -> Dict[str, float]:
        """
        Analyze what concepts are most likely to cause the given effect concepts.
        
        Args:
            effect_concepts: Concepts to analyze as effects
            potential_causes: Specific concepts to test as causes (None = test all)
            top_k: Number of top causal relationships to return
            
        Returns:
            Dictionary of {cause_concept: causal_strength}
        """
        if isinstance(effect_concepts, str):
            effect_concepts = [effect_concepts]
        
        # Test each potential cause
        causal_strengths = {}
        
        test_concepts = potential_causes if potential_causes else list(self.id_to_concept.values())
        
        for cause_concept in test_concepts:
            # Forward propagate from this potential cause
            result = self.propagate_forward(cause_concept, steps=3)
            
            # Measure how much it activates the effect concepts
            total_effect_activation = 0.0
            for effect_concept in effect_concepts:
                activation = result.final_activations.get(effect_concept, 0.0)
                total_effect_activation += activation
            
            causal_strengths[cause_concept] = total_effect_activation / len(effect_concepts)
        
        # Return top K causal relationships
        sorted_causes = sorted(causal_strengths.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_causes[:top_k])
    
    def find_activation_path(self, 
                           start_concept: str, 
                           target_concept: str,
                           max_path_length: int = 5) -> Optional[List[Tuple[str, float]]]:
        """
        Find the activation path from start concept to target concept.
        
        Args:
            start_concept: Starting concept
            target_concept: Target concept to reach
            max_path_length: Maximum path length to search
            
        Returns:
            List of (concept, activation) tuples showing the path, or None if no path
        """
        # Use forward propagation with detailed step tracking
        result = self.propagate_forward(start_concept, steps=max_path_length)
        
        # Check if target was reached
        target_activation = result.final_activations.get(target_concept, 0.0)
        
        if target_activation < self.activation_threshold:
            return None  # No path found
        
        # Reconstruct path by finding highest activation route through steps
        path = [(start_concept, 1.0)]  # Start with initial concept
        
        current_concept = start_concept
        for step_activations in result.propagation_steps[1:]:  # Skip initial step
            # Find the most activated concept connected to current concept
            best_next = None
            best_activation = 0.0
            
            for concept, activation in step_activations.items():
                if (activation >= self.activation_threshold and 
                    concept != current_concept and 
                    concept not in [p[0] for p in path]):
                    
                    if activation > best_activation:
                        best_activation = activation
                        best_next = concept
            
            if best_next:
                path.append((best_next, best_activation))
                current_concept = best_next
                
                # If we reached the target, we're done
                if current_concept == target_concept:
                    break
        
        # Verify the path actually reaches the target
        if path[-1][0] == target_concept:
            return path
        else:
            return None


def create_bidirectional_interface(huey_network):
    """
    Create a bidirectional propagation interface for a Huey network.
    
    Args:
        huey_network: Existing Huey network instance
        
    Returns:
        HueyBidirectionalPropagator instance ready for use
    """
    return HueyBidirectionalPropagator(huey_network)


# Example usage functions
def demo_bidirectional_propagation():
    """Demonstrate bidirectional propagation capabilities."""
    
    print("🔄 HUEY BIDIRECTIONAL PROPAGATION DEMO")
    print("=" * 50)
    
    # This would be called with an actual Huey network
    print("Example usage:")
    print()
    print("# Create bidirectional interface")  
    print("propagator = create_bidirectional_interface(huey_network)")
    print()
    print("# Forward propagation: physics → ?")
    print("result = propagator.propagate_forward('physics')")
    print("print('Forward from physics:', result.activated_concepts[:5])")
    print()
    print("# Backward propagation: ? → mathematics") 
    print("result = propagator.propagate_backward('mathematics')")
    print("print('Backward to mathematics:', result.activated_concepts[:5])")
    print()
    print("# Causality analysis")
    print("causes = propagator.analyze_causality('intelligence')")
    print("print('What causes intelligence:', causes)")
    print()
    print("# Find activation path")
    print("path = propagator.find_activation_path('physics', 'consciousness')")
    print("print('Path from physics to consciousness:', path)")


if __name__ == "__main__":
    demo_bidirectional_propagation()