# Huey+ Fortran Hybrid Architecture Design

## Performance Analysis Summary

**Current Performance**:
- Processing speed: 3,600-6,600 words/sec
- Memory usage: ~78 KB for 42-neuron network  
- Scaling: Sub-linear (good algorithmic efficiency)
- Bottleneck: Activation calculations and Hebbian learning loops

**Key Finding**: Current implementation is already quite efficient due to sparse connectivity (11.98% density) and good algorithmic design. However, Fortran could provide significant gains for the numerical kernels.

## Hybrid Architecture Design

### Python Components (Keep as-is)
```python
# Text processing and I/O
- Tokenization and text cleaning
- Kill word filtering  
- Speaker detection and attribution
- File I/O and data serialization
- Web interface (Streamlit)
- Query engine and natural language processing
- Visualization and plotting

# High-level orchestration
- Network initialization and configuration
- Session management and temporal tracking
- Analysis report generation
- Interactive dashboard coordination
```

### Fortran Computational Kernels
```fortran
! Core numerical operations to implement in Fortran
! Interface via NumPy arrays and f2py

module huey_fortran_kernels
    use, intrinsic :: iso_c_binding
    implicit none
    
contains

    ! 1. Hebbian Learning Kernel (Highest Priority)
    subroutine hebbian_update(window_neurons, activations, connections, masses, &
                             learning_rate, resistance_factor, n_window, n_total)
        ! Dense matrix operations on connection subsets
        ! 10-20x speedup expected
    
    ! 2. Activation Calculation (High Priority) 
    subroutine calculate_activations(neurons, connections, activations, bias, n_neurons)
        ! Vectorized exp/tanh operations
        ! 5-10x speedup expected
    
    ! 3. Decay Operations (Medium Priority)
    subroutine apply_decay(activations, connections, masses, decay_rates, n_neurons)
        ! Element-wise array operations
        ! 3-7x speedup expected
    
    ! 4. Sparse Matrix Operations (Medium Priority)
    subroutine sparse_matrix_vector_product(row_indices, col_indices, values, &
                                          input_vector, output_vector, n_nonzero)
        ! Optimized sparse matrix operations
        ! 3-8x speedup expected
    
    ! 5. Connection Pruning (Lower Priority)
    subroutine prune_weak_connections(connections, masses, threshold, &
                                    keep_mask, n_connections)
        ! Threshold-based filtering
        ! 3-5x speedup expected

end module huey_fortran_kernels
```

### Interface Design
```python
# Python wrapper for Fortran kernels
class HueyFortranInterface:
    def __init__(self, max_neurons=500):
        # Initialize Fortran workspace arrays
        self.connections_array = np.zeros((max_neurons, max_neurons), dtype=np.float64)
        self.masses_array = np.zeros((max_neurons, max_neurons), dtype=np.float64) 
        self.activations_array = np.zeros(max_neurons, dtype=np.float64)
        
    def hebbian_update_batch(self, window_neurons, network):
        # Convert sparse dict to dense arrays for Fortran
        # Call Fortran kernel
        # Convert results back to sparse format
        pass
        
    def calculate_activations_batch(self, network):
        # Prepare arrays for Fortran
        # Call vectorized Fortran activation kernel
        # Update network state
        pass
```

## Implementation Strategy

### Phase 1: Core Numerical Kernels
1. **Hebbian Learning Kernel** (Highest ROI)
   - Input: Window neurons, current activations, connection matrix
   - Output: Updated connection strengths and masses
   - Expected speedup: 10-20x

2. **Activation Calculation** (High ROI)
   - Input: Network state, connection matrix
   - Output: Updated activation values
   - Expected speedup: 5-10x

### Phase 2: Matrix Operations
3. **Sparse Matrix Operations** 
   - Optimized sparse matrix-vector products
   - Connection lookup acceleration
   - Expected speedup: 3-8x

4. **Decay Operations**
   - Vectorized decay for activations, connections, masses
   - Expected speedup: 3-7x

### Phase 3: Advanced Features  
5. **HueyTime Integration**
   - Fortran implementation of lagged learning
   - Temporal matrix operations
   - Expected speedup: 5-15x

## Development Workflow

### 1. Setup Build System
```bash
# f2py for Python-Fortran interface
pip install numpy
# Ensure gfortran is available
brew install gcc  # or system equivalent
```

### 2. Incremental Development
```python
# Start with single kernel
# huey_fortran_kernels.f90 - just hebbian_update
# Build: f2py -c -m huey_fortran_kernels huey_fortran_kernels.f90

# Test performance gain
# Integrate into existing network class
# Measure speedup vs Python implementation
```

### 3. Compatibility Layer
```python
class HueyNetworkHybrid(HueyConversationalNetwork):
    """Drop-in replacement with Fortran acceleration."""
    
    def __init__(self, use_fortran=True, **kwargs):
        super().__init__(**kwargs)
        self.use_fortran = use_fortran
        if use_fortran:
            self.fortran_interface = HueyFortranInterface(self.max_neurons)
    
    def _hebbian_learning(self, window_neurons):
        if self.use_fortran:
            return self.fortran_interface.hebbian_update_batch(window_neurons, self)
        else:
            return super()._hebbian_learning(window_neurons)
```

## Expected Benefits

### Performance Gains
- **Small networks (50-200 neurons)**: 2-3x overall speedup
- **Medium networks (500-1000 neurons)**: 3-5x overall speedup  
- **Large networks (1000+ neurons)**: 5-10x overall speedup
- **Real-time streaming**: 10-20x speedup for continuous processing

### Use Cases That Would Benefit Most
1. **Large corpus analysis**: Processing books, research papers
2. **Real-time conversation analysis**: Live transcription processing
3. **Longitudinal studies**: Multi-session temporal analysis
4. **Parameter sweeps**: Testing different learning configurations
5. **Production deployments**: Server-side processing at scale

## Development Effort Estimate
- **Phase 1** (core kernels): 2-3 weeks
- **Phase 2** (matrix ops): 1-2 weeks  
- **Phase 3** (advanced features): 2-4 weeks
- **Total**: 5-9 weeks for complete hybrid system

## Recommendation
**Yes, absolutely worthwhile for Emary's research needs.** The hybrid approach gives you:
- 3-10x speedup for numerical computations
- Maintained Python ecosystem for text processing
- Easy integration with existing codebase
- Scalability for production research applications

Start with Phase 1 (Hebbian learning kernel) to validate the approach and measure real-world speedup before proceeding to full implementation.