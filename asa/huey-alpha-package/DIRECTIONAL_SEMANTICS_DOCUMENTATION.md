# Directional Semantics in Huey: User Guide

## Understanding Directional Connections

### What Are Directional Connections?

In Huey's neural network, connections between concepts are **directional** and **asymmetric**. This means:

```
Connection (A → B) ≠ Connection (B → A)
```

This is fundamentally different from traditional text analysis methods that treat word relationships as symmetric.

### Why Does Direction Matter?

**Real-world example:**
- "New York" appears thousands of times in text
- "York New" appears almost never
- This isn't random - it reflects **semantic constraints** in English grammar

**In conversation analysis:**
- `(I → think)` vs `(think → I)` capture different discourse patterns
- `(Claude → developed)` vs `(developed → Claude)` show different semantic roles
- Direction reveals **who does what to whom**

## How Huey Uses Directional Connections

### 1. Sliding Window Processing

When Huey processes text like: *"Einstein developed relativity theory"*

It creates these directional connections:
```
(Einstein → developed) - Einstein as agent
(developed → relativity) - relativity as object  
(relativity → theory) - theory as classification
```

Each connection has different strength based on:
- **Frequency**: How often this pattern appears
- **Context**: Position within the sliding window
- **Reinforcement**: Hebbian learning strengthens repeated patterns

### 2. Connection Storage

Connections are stored as tuples `(neuron_i, neuron_j)` where:
- **neuron_i**: The "source" concept (left side)
- **neuron_j**: The "target" concept (right side)

Example:
```python
connections = {
    (einstein_neuron, developed_neuron): 0.75,
    (developed_neuron, einstein_neuron): 0.12,  # Much weaker reverse
    (new_neuron, york_neuron): 0.93,
    (york_neuron, new_neuron): 0.003           # Almost nonexistent
}
```

### 3. Self-Concept Formation

For speakers developing self-concepts, direction reveals:

**Active Self-Construction:**
```
(I → am), (I → think), (I → believe)  [Strong outbound connections]
```

**Passive Attribution:**
```
(others → describe), (people → see), (you → told)  [Inbound connections to self]
```

Strong self-concepts show **high outbound/inbound ratios** - the speaker actively constructs their identity rather than passively accepting external descriptions.

## Interpreting Huey's Results

### In Concept Association Queries

When you query associations for "Einstein", Huey considers both:

**Outgoing connections (Einstein → X):**
- developed (0.75)
- physics (0.68)  
- theory (0.63)

**Incoming connections (X → Einstein):**
- scientist (0.71)
- genius (0.65)
- German (0.43)

This reveals Einstein's **semantic roles**: as an agent who develops/creates, and as a target of descriptions (scientist, genius).

### In Multi-Concept Analysis

When analyzing multiple concepts, directional patterns show:
- **Shared outgoing**: Common things these concepts do/create
- **Shared incoming**: Common descriptions/attributes
- **Interaction patterns**: How concepts relate to each other directionally

### In Visualizations

Network graphs show:
- **Arrow direction**: Semantic relationship flow
- **Arrow thickness**: Connection strength  
- **Node positioning**: Concepts cluster by directional similarity

## Practical Examples

### Example 1: Speaker Analysis

**Speaker A's patterns:**
```
(I → think): 0.82    (Strong self-agency)
(think → I): 0.11    (Weak reverse)
Ratio: 7.4:1         (High agency)
```

**Speaker B's patterns:**
```
(I → think): 0.34    (Moderate self-agency)  
(think → I): 0.51    (Stronger reverse - more passive)
Ratio: 0.67:1        (Lower agency)
```

Speaker A shows stronger self-concept formation through active language patterns.

### Example 2: Concept Evolution

Tracking how concepts develop directional patterns over time:

```
Early conversation:
(AI → learn): 0.23
(learn → AI): 0.67   (AI seen as passive learner)

Later conversation:  
(AI → learn): 0.71
(learn → AI): 0.31   (AI becomes active learner)
```

This shows AI's semantic role shifting from passive to active.

### Example 3: Domain Knowledge

Scientific concepts show predictable directional patterns:

```
(scientist → discover): High (agents of discovery)
(discover → scientist): Low  (discovery doesn't create scientists)

(theory → explain): High     (theories explain things)
(explain → theory): Lower    (explaining doesn't always create theories)
```

## Technical Implementation

### Connection Keys

In the code, you'll see connections referenced as:
```python
conn_key = (neuron_i, neuron_j)  # Always in this order
reverse_key = (neuron_j, neuron_i)  # Reverse direction
```

### Strength Calculation

Connection strength updates use:
```python
# Hebbian learning with directional bias
strength(i → j) += learning_rate * activation_i * activation_j * position_weight
```

Where `position_weight` accounts for word order in the sliding window.

### Asymmetry Metrics

To measure directional asymmetry:
```python
forward_strength = connections.get((i, j), 0.0)
reverse_strength = connections.get((j, i), 0.0)
asymmetry_ratio = forward_strength / max(reverse_strength, 0.001)
```

## Best Practices for Analysis

### 1. Consider Both Directions
Always examine both `(A → B)` and `(B → A)` for complete understanding.

### 2. Look for Asymmetry Patterns  
High asymmetry ratios (>5:1) indicate strong semantic directionality.

### 3. Track Changes Over Time
Directional patterns evolving during conversation reveal concept development.

### 4. Compare Speakers
Different directional patterns between speakers show different semantic roles and agency levels.

### 5. Use Multi-Concept Analysis
Shared directional patterns across concepts reveal deeper semantic relationships.

## Troubleshooting

### Why are connections so asymmetric?
This is **correct behavior**. Natural language has inherent directional constraints. Symmetric connections would lose semantic information.

### What if reverse connections are zero?
Normal for many concept pairs. "New York" → strong. "York New" → near zero. This preserves grammatical reality.

### How to interpret weak bidirectional connections?
Suggests concepts that can appear in either order contextually, like "quantum physics" vs "physics quantum" (less common but valid).

---

This directional approach makes Huey uniquely powerful for understanding semantic relationships, conversational agency, and self-concept formation through the natural asymmetries of human language.