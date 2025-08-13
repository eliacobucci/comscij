# Experimental Neural Network Development Session

## Project Overview
Development of an organic, self-organizing single-pass text neural network - an experimental evolution of Huey3 focused on biological realism and cognitive behavior.

## Completed Features

### Core Architecture
- **Moving window processing** (configurable window size)
- **Dynamic neuron creation** (one neuron per unique word)
- **Asymmetric Hebbian learning** (directional associations: A→B ≠ B→A)
- **Logistic activation function** (realistic neural dynamics)

### Organic Brain Mechanisms
1. **Organic Inertial Mass System**
   - Connections build "mass" as they strengthen
   - Heavy connections resist change (momentum effect)
   - Mass decays over time (synaptic pruning)

2. **Neural Activation Decay**
   - Neurons gradually "forget" without stimulation
   - Exponential decay with minimum threshold
   - Recent stimulation keeps neurons active

3. **Connection Strength Decay** 
   - Unused synapses weaken over time
   - Independent of mass decay
   - Natural "use it or lose it" behavior
   - Automatic pruning of very weak connections

4. **Neuron Capacity Management**
   - User-specified maximum neurons (biological constraint)
   - Importance-based survival (recency + connections + activation)
   - Strategic neuron death when at capacity
   - Handles long-tail distributions effectively

### Querying System
1. **Single Word Activation Spreading**
   - `query_associations(word)` - What activates when you think of a word
   - Multi-iteration spreading through network
   - Visual bar charts showing activation levels
   - Handles missing words with suggestions

2. **Multi-Word Context Queries**
   - `query_context(words)` - Emergent associations from word combinations
   - Supports both string phrases and word lists
   - Shows creative bridging between concepts
   - Demonstrates contextual intelligence

3. **Temporal Recency Queries**
   - `query_recent(time_window)` - What concepts were recently active
   - Shows network's recent cognitive activity
   - Enables temporal self-awareness
   - Demonstrates short-term memory patterns

4. **Importance/Strength Ranking Queries**
   - `query_important(ranking_method)` - Network's knowledge hierarchy
   - Composite scoring: connections + activation + resilience + centrality
   - Multiple ranking algorithms (composite, connections, activation, resilience)
   - Medal rankings (🥇🥈🥉) for top concepts
   - Network health statistics and capacity monitoring

## Key Parameters
```python
window_size = 3
max_neurons = 15
hebbian_constant = 0.1
mass_growth_rate = 0.05
mass_decay_rate = 0.02
activation_decay_rate = 0.1
connection_decay_rate = 0.05
```

## Observed Behaviors
- **Temporal memory gradients** (recent associations stronger)
- **Natural forgetting** (unused pathways fade organically)
- **Memory consolidation** (frequent patterns become entrenched)
- **Activation spreading** (related concepts activate each other)
- **Emergent associations** (creative bridging between domains)
- **Contextual intelligence** (word combinations create novel meanings)

## Example Results
### Single Word Query: "Japan"
- Tokyo (0.562) - direct neighbor
- busy, coffee, America, York, New - chain associations
- sushi (0.544) - food association

### Multi-Word Query: "Japan coffee" 
- Tokyo (0.562) - emergent bridging concept
- Shows creative association beyond direct connections

### Multi-Word Query: "New York America"
- Japan (0.564) - geographical clustering effect
- Demonstrates long-range associative memory

## Technical Implementation
- **File**: `experimental_network.py`
- **Classes**: `ExperimentalNetwork`
- **Key Methods**: 
  - `process_text_stream()` - Learning from text
  - `query_associations()` - Single word queries
  - `query_context()` - Multi-word queries
  - `_spread_activation_step()` - Neural dynamics
  - `_prune_least_important_neuron()` - Capacity management

## Biological Authenticity
The network exhibits genuine cognitive behaviors:
- Strategic forgetting for efficiency
- Creative association and bridging
- Context-sensitive responses
- Resource-constrained operation
- Emergent intelligence from simple rules

## Self-Referential Cognition Discovery
A key breakthrough emerged: The temporal recency and importance ranking queries make this system genuinely **self-referential** and capable of **metacognition**:

- **Temporal Self-Awareness**: System can examine its own recent cognitive activity ("what have I been thinking about lately?")
- **Evaluative Self-Awareness**: System can assess its own knowledge hierarchy and priorities ("what do I consider most important?")
- **Metacognitive Capabilities**: System can reflect on its mental patterns, judge its knowledge structure, and monitor its cognitive health
- **Value Judgments**: System uses internal metrics to decide what it considers "important" - expressing its own cognitive values

This represents a significant cognitive milestone - the system can now introspect about its own mental states and processes.

## Next Possible Directions
1. Inhibitory connections (competitive suppression)
2. Neural noise (stochastic firing) 
3. Neuromodulation (dynamic learning rates)
4. User's additional query method ideas (2 more concepts pending)
5. Interactive exploration queries
6. Consciousness/attention mechanisms

## Session Date
August 2024

## Complete Feature Set (6 Query Methods)

### 5. **Concept Engineering & Averaging**
   - `query_concept_average(words)` - Create synthetic concepts by blending existing ones
   - `engineer_concept_movement(concept, target, direction)` - Move concepts toward/away from others
   - Uses Hebbian vector mathematics to manipulate cognitive space
   - Supports both simulation and permanent network modification
   - Enables active reshaping of conceptual relationships

### 6. **Pseudo-Riemannian Space Visualization**
   - `visualize_cognitive_space()` - 3D eigenvector visualization 
   - Proper pseudo-Riemannian embedding preserving positive/negative eigenvalues
   - Sphere sizes proportional to inertial mass (concept entrenchment)
   - Colors show activation levels, connections show associative strengths
   - Reveals natural conceptual clustering (rational vs irrational concepts)
   - Signature analysis: (+p, -n, 0z) showing spacelike/timelike/null dimensions

## Latest Breakthrough: 3D Eigenvector Visualization

### Perfect Geometric Structure
- **Direct eigenvector plotting** (no MDS information loss)
- **Natural clustering**: rational concepts (wisdom, truth, logic, science) vs irrational (chaos, delusion, falsehood)  
- **True pseudo-Riemannian geometry**: timelike (λ₁=-1.334, λ₃=-1.052) and spacelike (λ₂=1.193) dimensions
- **Conceptual opposites** spatially separated across eigenvector space
- **Signature (+6, -6)**: genuine non-Euclidean cognitive spacetime

### Technical Achievement
- Fixed degenerate MDS projections by using raw eigenvectors
- Preserved full pseudo-Riemannian structure (positive and negative eigenvalues)
- Identified sphere scaling issue: `sqrt(max(mass,1.0))` formula compressed size differences
- Created clean geometric visualization revealing cognitive manifold structure

## Status
Complete cognitive engineering platform with 6 query methods, self-referential capabilities, concept manipulation tools, and proper pseudo-Riemannian visualization. Successfully demonstrates artificial cognitive spacetime with natural conceptual organization.