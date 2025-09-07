! Huey++ Fortran Computational Kernels
! High-performance numerical routines for Hebbian learning and neural network operations
! 
! Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
!
! Compile with: f2py -c -m huey_fortran_kernels huey_fortran_kernels.f90

module huey_fortran_kernels
    use, intrinsic :: iso_c_binding
    implicit none
    
    ! Constants
    real(kind=c_double), parameter :: EPSILON = 1.0d-12
    real(kind=c_double), parameter :: PI = 3.141592653589793d0
    
contains

    ! =========================================================================
    ! HEBBIAN LEARNING KERNEL (Priority 1 - Highest Performance Impact)
    ! =========================================================================
    
    subroutine hebbian_update_batch(window_neurons, n_window, activations, n_neurons, &
                                   connections, masses, n_max_connections, &
                                   learning_rate, resistance_factor, M_max, &
                                   A_0, k_steepness, updated_connections, updated_masses) &
                                   bind(c, name='hebbian_update_batch')
        
        ! Input parameters
        integer(c_int), intent(in) :: n_window, n_neurons, n_max_connections
        integer(c_int), intent(in) :: window_neurons(n_window)
        real(c_double), intent(in) :: activations(n_neurons)
        real(c_double), intent(in) :: connections(n_max_connections, 3)  ! (i, j, strength)
        real(c_double), intent(in) :: masses(n_max_connections, 3)       ! (i, j, mass)
        real(c_double), intent(in) :: learning_rate, resistance_factor
        real(c_double), intent(in) :: M_max, A_0, k_steepness
        
        ! Output parameters
        real(c_double), intent(out) :: updated_connections(n_max_connections, 3)
        real(c_double), intent(out) :: updated_masses(n_max_connections, 3)
        
        ! Local variables
        integer :: i, j, pos_i, pos_j, neuron_i, neuron_j, conn_idx
        real(kind=c_double) :: ai, aj, activity, current_strength, current_mass
        real(kind=c_double) :: inertial_resistance, delta_w, new_strength
        real(kind=c_double) :: logistic_mass, mass_difference, mass_change, new_mass
        real(kind=c_double) :: homeostatic_correction, ltd_correction
        logical :: found_connection
        
        ! Initialize output arrays with input values
        updated_connections = connections
        updated_masses = masses
        
        ! Process all pairs in the window (earlier to later only - temporal causality)
        do pos_i = 1, n_window - 1
            do pos_j = pos_i + 1, n_window
                
                neuron_i = window_neurons(pos_i)
                neuron_j = window_neurons(pos_j)
                
                ! Skip self-connections
                if (neuron_i == neuron_j) cycle
                
                ! Get activations (1-indexed for Fortran, but 0-indexed data)
                ai = activations(neuron_i + 1)
                aj = activations(neuron_j + 1)
                
                ! Find connection in sparse matrix
                found_connection = .false.
                conn_idx = 0
                
                do i = 1, n_max_connections
                    if (int(connections(i, 1)) == neuron_i .and. int(connections(i, 2)) == neuron_j) then
                        found_connection = .true.
                        conn_idx = i
                        exit
                    end if
                end do
                
                if (.not. found_connection) cycle
                
                ! Get current values
                current_strength = connections(conn_idx, 3)
                current_mass = masses(conn_idx, 3)
                
                ! Calculate inertial resistance (higher mass = harder to change)
                inertial_resistance = 1.0d0 / (1.0d0 + current_mass * resistance_factor)
                
                ! Hebbian update: Δw = learning_rate × ai × aj × resistance
                delta_w = learning_rate * ai * aj * inertial_resistance
                new_strength = current_strength + delta_w
                
                ! Calculate synaptic mass using biologically accurate logistic model
                activity = ai * aj
                
                ! Core logistic function for Long Term Potentiation (LTP)
                if (abs(activity - A_0) > 50.0d0) then
                    ! Handle extreme values to prevent overflow
                    if (activity > A_0) then
                        logistic_mass = M_max
                    else
                        logistic_mass = 0.0d0
                    end if
                else
                    logistic_mass = M_max / (1.0d0 + exp(-k_steepness * (activity - A_0)))
                end if
                
                ! Homeostatic scaling correction (reduces growth at very high activity)
                homeostatic_correction = -0.1d0 * max(0.0d0, activity - 0.8d0) * current_mass
                
                ! Long Term Depression (LTD) correction for weak activity
                if (activity < 0.15d0) then
                    ltd_correction = -0.1d0 * (0.15d0 - activity) * current_mass
                else
                    ltd_correction = 0.0d0
                end if
                
                ! Calculate mass change toward logistic target
                mass_difference = logistic_mass - current_mass
                mass_change = 0.15d0 * mass_difference + homeostatic_correction + ltd_correction
                
                ! Update mass (bounded by [0, M_max])
                new_mass = max(0.0d0, min(current_mass + mass_change, M_max))
                
                ! Store updates
                updated_connections(conn_idx, 3) = new_strength
                updated_masses(conn_idx, 3) = new_mass
                
            end do
        end do
        
    end subroutine hebbian_update_batch
    
    ! =========================================================================
    ! ACTIVATION CALCULATION KERNEL (Priority 2 - High Performance Impact)
    ! =========================================================================
    
    subroutine calculate_activations_batch(window_neurons, n_window, n_neurons, &
                                         connections, n_connections, &
                                         current_activations, bias, &
                                         new_activations) &
                                         bind(c, name='calculate_activations_batch')
        
        ! Input parameters
        integer(c_int), intent(in) :: n_window, n_neurons, n_connections
        integer(c_int), intent(in) :: window_neurons(n_window)
        real(c_double), intent(in) :: connections(n_connections, 3)  ! (i, j, strength)
        real(c_double), intent(in) :: current_activations(n_neurons)
        real(c_double), intent(in) :: bias
        
        ! Output parameters
        real(c_double), intent(out) :: new_activations(n_neurons)
        
        ! Local variables
        integer :: neuron_idx, other_idx, i
        real(kind=c_double) :: weighted_sum, activation
        logical :: is_window_neuron
        
        ! Calculate activations for ALL neurons
        do neuron_idx = 1, n_neurons
            
            ! Check if this neuron is in the current window
            is_window_neuron = .false.
            do i = 1, n_window
                if (window_neurons(i) + 1 == neuron_idx) then ! Adjust for 0-indexed input
                    is_window_neuron = .true.
                    exit
                end if
            end do
            
            if (is_window_neuron) then
                ! Window neurons get direct input activation
                new_activations(neuron_idx) = 1.0d0
            else
                ! Non-window neurons: calculate weighted sum from connections
                weighted_sum = bias
                
                ! Sum weighted inputs from all other neurons
                do i = 1, n_connections
                    other_idx = int(connections(i, 1)) + 1  ! Adjust for 0-indexed input
                    
                    ! Check if this connection targets our neuron
                    if (int(connections(i, 2)) + 1 == neuron_idx .and. other_idx /= neuron_idx) then
                        weighted_sum = weighted_sum + connections(i, 3) * current_activations(other_idx)
                    end if
                end do
                
                ! Apply logistic activation function: 1 / (1 + e^(-weighted_sum))
                if (weighted_sum > 500.0d0) then
                    activation = 1.0d0  ! Handle overflow
                else if (weighted_sum < -500.0d0) then
                    activation = 0.0d0  ! Handle underflow
                else
                    activation = 1.0d0 / (1.0d0 + exp(-weighted_sum))
                end if
                
                new_activations(neuron_idx) = activation
                
            end if
        end do
        
    end subroutine calculate_activations_batch
    
    ! =========================================================================
    ! DECAY OPERATIONS (Priority 3 - Medium Performance Impact)
    ! =========================================================================
    
    subroutine apply_activation_decay(activations, n_neurons, decay_rate, &
                                    minimum_activation, decayed_activations) &
                                    bind(c, name='apply_activation_decay')
        
        ! Input parameters
        integer(c_int), intent(in) :: n_neurons
        real(c_double), intent(in) :: activations(n_neurons)
        real(c_double), intent(in) :: decay_rate, minimum_activation
        
        ! Output parameters
        real(c_double), intent(out) :: decayed_activations(n_neurons)
        
        ! Local variables
        integer :: i
        real(kind=c_double) :: decayed_value
        
        ! Apply exponential decay to all activations
        !$OMP PARALLEL DO PRIVATE(decayed_value)
        do i = 1, n_neurons
            ! Exponential decay: new = old * (1 - decay_rate)
            decayed_value = activations(i) * (1.0d0 - decay_rate)
            
            ! Don't let activations go below minimum
            decayed_activations(i) = max(decayed_value, minimum_activation)
        end do
        !$OMP END PARALLEL DO
        
    end subroutine apply_activation_decay
    
    subroutine apply_connection_decay(connections, n_connections, active_pairs, &
                                    n_active_pairs, decay_rate, minimum_connection, &
                                    decayed_connections) &
                                    bind(c, name='apply_connection_decay')
        
        ! Input parameters
        integer(c_int), intent(in) :: n_connections, n_active_pairs
        real(c_double), intent(in) :: connections(n_connections, 3)  ! (i, j, strength)
        integer(c_int), intent(in) :: active_pairs(n_active_pairs, 2)  ! (i, j) pairs that are active
        real(c_double), intent(in) :: decay_rate, minimum_connection
        
        ! Output parameters
        real(c_double), intent(out) :: decayed_connections(n_connections, 3)
        
        ! Local variables
        integer :: conn_idx, pair_idx, conn_i, conn_j
        real(kind=c_double) :: decayed_strength
        logical :: is_active
        
        ! Copy input to output first
        decayed_connections = connections
        
        ! Apply decay only to inactive connections
        do conn_idx = 1, n_connections
            
            conn_i = int(connections(conn_idx, 1))
            conn_j = int(connections(conn_idx, 2))
            
            ! Check if this connection is active (being reinforced)
            is_active = .false.
            do pair_idx = 1, n_active_pairs
                if (active_pairs(pair_idx, 1) == conn_i .and. &
                    active_pairs(pair_idx, 2) == conn_j) then
                    is_active = .true.
                    exit
                end if
            end do
            
            if (.not. is_active) then
                ! Decay this connection
                decayed_strength = connections(conn_idx, 3) * (1.0d0 - decay_rate)
                
                ! Don't let connections go below minimum
                decayed_connections(conn_idx, 3) = max(decayed_strength, minimum_connection)
            end if
            
        end do
        
    end subroutine apply_connection_decay
    
    ! =========================================================================
    ! UTILITY FUNCTIONS
    ! =========================================================================
    
    subroutine sparse_matrix_vector_product(row_indices, col_indices, values, &
                                          n_nonzero, input_vector, n_vector, &
                                          output_vector) &
                                          bind(c, name='sparse_matrix_vector_product')
        
        ! Input parameters
        integer(c_int), intent(in) :: n_nonzero, n_vector
        integer(c_int), intent(in) :: row_indices(n_nonzero), col_indices(n_nonzero)
        real(c_double), intent(in) :: values(n_nonzero)
        real(c_double), intent(in) :: input_vector(n_vector)
        
        ! Output parameters
        real(c_double), intent(out) :: output_vector(n_vector)
        
        ! Local variables
        integer :: i, row_idx, col_idx
        
        ! Initialize output
        output_vector = 0.0d0
        
        ! Sparse matrix-vector multiplication
        !$OMP PARALLEL DO PRIVATE(row_idx, col_idx)
        do i = 1, n_nonzero
            row_idx = row_indices(i) + 1  ! Adjust for 0-indexed input
            col_idx = col_indices(i) + 1  ! Adjust for 0-indexed input
            
            if (row_idx >= 1 .and. row_idx <= n_vector .and. &
                col_idx >= 1 .and. col_idx <= n_vector) then
                !$OMP ATOMIC
                output_vector(row_idx) = output_vector(row_idx) + values(i) * input_vector(col_idx)
            end if
        end do
        !$OMP END PARALLEL DO
        
    end subroutine sparse_matrix_vector_product
    
end module huey_fortran_kernels