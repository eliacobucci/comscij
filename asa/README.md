# Huey: Hebbian Conversational Self-Concept Analysis System

**Huey** is a sophisticated conversation analysis system that uses Hebbian learning principles to study self-concept formation in multi-speaker conversations. Named for its Hebbian foundations, Huey treats self-concepts as regular concepts that emerge naturally through sliding window analysis with equal competition dynamics.

## Key Features

🧠 **Hebbian Learning**: Uses natural Hebbian learning with sliding windows where all concepts compete equally  
💬 **Perfect Speaker Identification**: Unambiguous speaker tagging eliminates attribution uncertainty  
🔬 **Self-Concept Analysis**: Treats self-concepts as regular concepts with natural decay dynamics  
📊 **3D Visualization**: Generates spatial representations of concept clusters  
⚖️ **Mathematical Universality**: Universal learning principles while preserving individual identity  

## Core Philosophy

**"The self-concept is just another concept."** 

Huey's revolutionary approach treats self-concepts as regular concepts subject to the same decay and competition dynamics as all other concepts. No artificial privileging, no structural bias - just natural emergence through mathematical principles.

## System Components

### HueyConversationalNetwork
The core neural network that processes conversations using:
- Sliding windows (configurable size, default 7)
- Natural decay for all neurons including speaker neurons
- Hebbian connection strengthening
- Inertial mass accumulation

### UnambiguousConversationSystem  
Perfect conversation recording with:
- Explicit speaker identification for every utterance
- 100% confidence, zero ambiguity
- Structured export for analysis

### Complete Analysis Pipeline
End-to-end processing including:
- Conversation recording
- Self-concept analysis  
- 3D cluster visualization
- Comprehensive reporting

## Quick Start

```python
from conversational_self_concept_experiment import HueyConversationalNetwork

# Create Huey network
huey = HueyConversationalNetwork(max_neurons=100, window_size=7)

# Add speakers
huey.add_speaker("alice", ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])
huey.add_speaker("bob", ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])

# Process conversation
huey.process_speaker_text("alice", "I think this approach works well for me.")
huey.process_speaker_text("bob", "I agree - my experience shows similar results.")

# Analyze results
results = huey.compare_speaker_self_concepts()
```

## Technical Implementation

- **Learning Algorithm**: Hebbian with sliding windows and natural decay
- **Speaker Attribution**: Active speaker neuron during turns, natural decay between turns
- **Concept Competition**: All concepts compete equally for activation and mass
- **Analysis Engine**: Eigenanalysis of connection matrices for cluster identification
- **Data Format**: Structured JSON with timestamps and metadata

## Research Applications

Huey is designed for:
- Multi-speaker conversation analysis
- Self-concept formation research
- Cross-cultural identity studies
- Mathematical consciousness modeling
- Collaborative research environments

## Files

- `conversational_self_concept_experiment.py` - Core Huey network implementation
- `unambiguous_conversation_system.py` - Perfect speaker identification system
- `complete_conversation_analysis_system.py` - Full analysis pipeline
- `conversation_reprocessor.py` - Legacy conversation processing
- `test_sliding_window_conversation.py` - Test suite

## Philosophy

Huey embodies the principle that the most elegant solutions treat all elements equally. By removing artificial hierarchies and letting self-concepts emerge through the same mechanisms as all other concepts, we achieve both mathematical universality and individual authenticity.

**No guessing. No ambiguity. Just perfect Hebbian self-concept analysis.**