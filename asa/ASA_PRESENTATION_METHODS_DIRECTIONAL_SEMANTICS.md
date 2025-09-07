# Methods: Directional Semantic Connections in Self-Concept Analysis
## ASA Annual Conference Presentation

---

### Method Innovation: Beyond Symmetric Co-occurrence

**Traditional Approach:**
```
If "word A" and "word B" appear together → connection(A,B) = connection(B,A)
```

**Our Directional Approach:**
```
connection(A → B) ≠ connection(B → A)
```
*Preserves semantic asymmetries inherent in natural language*

---

### The Frequency-Meaning Connection

**Empirical Observation:**
- "New York": ~10,000 occurrences
- "York New": ~3 occurrences (errors/poetry)

**Theoretical Implication:**
Asymmetric frequency reflects **grammatical constraints** and **semantic roles**

**Methodological Advantage:**
Directional connections capture **who does what to whom**

---

### Technical Implementation

#### Sliding Window Processing
- **Window size**: 7 tokens (optimized for grammatical phrases)
- **Directional learning**: Hebbian strengthening with positional weighting
- **Connection storage**: Sparse matrix (i → j) tuples

#### Mathematical Formulation
```
strength(i → j) = Σ learning_rate × activation_i × activation_j × position_weight(i,j)
```

#### Asymmetry Measurement
```
asymmetry_ratio = strength(A → B) / strength(B → A)
```

---

### Self-Concept Detection Through Directionality

#### Agency Patterns
**High Self-Concept Speakers:**
- `(I → think)`: 0.82 vs `(think → I)`: 0.11 *(Ratio: 7.4:1)*
- `(I → am)`: 0.85 vs `(am → I)`: 0.09 *(Ratio: 9.4:1)*
- `(I → believe)`: 0.73 vs `(believe → I)`: 0.15 *(Ratio: 4.9:1)*

**Low Self-Concept Speakers:**
- `(I → think)`: 0.34 vs `(think → I)`: 0.51 *(Ratio: 0.67:1)*
- More **passive construction** vs **active self-reference**

#### Measurement Protocol
1. **Extract pronoun-verb connections** for each speaker
2. **Calculate asymmetry ratios** for self-reference patterns  
3. **Aggregate directional strength** as self-concept mass
4. **Compare across speakers** and conversation time

---

### Data Processing Pipeline

#### Step 1: Conversation Parsing
- **Speaker detection**: Multi-format support (labels, alternating, etc.)
- **Text cleaning**: Preserve semantic content, remove noise
- **Tokenization**: Semantic-aware splitting with kill-word exclusion

#### Step 2: Network Construction  
- **Sliding windows**: 7-token overlapping sequences
- **Neuron creation**: Dynamic vocabulary expansion
- **Connection learning**: Asymmetric Hebbian strengthening
- **Sparse storage**: Efficient (i,j) tuple indexing

#### Step 3: Analysis Extraction
- **Self-concept quantification**: Speaker-specific pronoun patterns
- **Temporal tracking**: Connection evolution over conversation
- **Comparative analysis**: Between-speaker differences

---

### Validation Metrics

#### Network-Level Validation
- **Sparsity maintenance**: ~15% connectivity (biologically plausible)
- **Asymmetry distribution**: Power-law expected for natural language
- **Stability**: Connection strengths converge over conversation

#### Self-Concept Validation  
- **Face validity**: High-agency speakers show expected patterns
- **Temporal consistency**: Self-concept mass increases over engagement
- **Cross-speaker reliability**: Consistent measurement across individuals

#### Linguistic Validation
- **Grammar preservation**: Syntactically valid patterns strengthened
- **Semantic coherence**: Related concepts cluster directionally
- **Domain knowledge**: Expert-validated concept relationships

---

### Sample Size and Participants

#### Current Dataset
- **Conversations**: 500+ human-AI exchanges
- **Speakers**: 200+ unique individuals  
- **Tokens processed**: 2.3M+ with directional tracking
- **Connection patterns**: 100K+ asymmetric relationships documented

#### Demographic Coverage
- **Age range**: 18-75 years
- **Education levels**: High school through PhD
- **Conversation types**: Technical, casual, therapeutic, educational
- **Languages**: Primarily English (cross-linguistic extension planned)

---

### Statistical Analysis Plan

#### Primary Hypothesis Testing
```
H₁: Self-concept mass = f(directional_asymmetry_ratios)
H₂: Agency_patterns → Conversation_engagement  
H₃: Temporal_development → Strengthened_self_reference
```

#### Analysis Framework
- **Mixed-effects modeling**: Speaker as random effect
- **Time-series analysis**: Connection evolution patterns
- **Network analysis**: Centrality and clustering metrics
- **Comparative statistics**: Between-speaker differences

---

### Advantages Over Traditional Methods

#### Methodological Improvements
- **Semantic preservation**: Grammatical relationships maintained
- **Dynamic learning**: Real-time connection adaptation  
- **Computational efficiency**: Sparse network scaling
- **Interpretability**: Clear directional relationship meaning

#### Theoretical Contributions
- **Agency detection**: Who acts vs who is acted upon
- **Identity formation**: Active vs passive self-construction
- **Discourse analysis**: Conversational flow patterns
- **Cross-domain application**: Beyond self-concept analysis

---

### Limitations and Future Directions

#### Current Limitations
- **Language specificity**: Optimized for English syntax
- **Context window**: Fixed 7-token sliding window
- **Computational resources**: Memory scaling with vocabulary size

#### Planned Extensions
- **Cross-linguistic validation**: SOV, VSO language testing
- **Multi-modal integration**: Visual and auditory directional patterns
- **Temporal dynamics**: Variable window sizes and decay rates
- **Clinical applications**: Therapeutic conversation analysis

---

### Reproducibility and Open Science

#### Code Availability
- **Full implementation**: Open-source release planned
- **Documentation**: Comprehensive user guides provided
- **Example datasets**: Anonymized conversation samples
- **Replication materials**: Analysis scripts and protocols

#### Methodological Transparency  
- **Parameter selection**: Empirically justified choices
- **Validation procedures**: Step-by-step protocols documented
- **Statistical assumptions**: Clearly stated and tested
- **Limitations disclosure**: Known constraints acknowledged

---

### Expected Contributions to the Field

#### Methodological Innovation
- **New paradigm**: Directional semantic analysis framework
- **Measurement precision**: Quantitative self-concept detection
- **Scalable application**: Large conversation dataset processing

#### Theoretical Advancement
- **Self-concept theory**: Linguistic formation mechanisms
- **Social interaction**: Conversational agency patterns  
- **Computational linguistics**: Semantic role preservation

#### Practical Applications
- **Therapeutic assessment**: Automated self-concept tracking
- **Educational research**: Student engagement measurement
- **AI development**: Human-like conversation understanding

---

**Conclusion**: Directional semantic connections represent a fundamental methodological advance, capturing asymmetric linguistic patterns that reveal self-concept formation through conversational agency. This approach provides both theoretical insights into identity development and practical tools for quantitative social science research.

---

*This research was conducted by the Galileo Company research team using the Huey conversational analysis platform.*