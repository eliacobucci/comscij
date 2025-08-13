class ExperimentalNetwork:
    """
    Experimental self-organizing single pass text neural network.
    Starting with basic moving window text processing.
    """
    
    def __init__(self, window_size=5, max_neurons=100):
        """
        Initialize the experimental network.
        
        Args:
            window_size (int): Size of the moving window
            max_neurons (int): Maximum number of neurons to retain (capacity limit)
        """
        self.window_size = window_size
        self.max_neurons = max_neurons
        self.processed_words = 0
        
        # Neural network components
        self.word_to_neuron = {}  # Maps words to neuron indices
        self.neuron_to_word = {}  # Maps neuron indices to words
        self.activations = {}     # Current activation values by neuron index
        self.neuron_count = 0     # Total number of neurons
        self.connections = {}     # Synaptic connection matrix (dict for sparse storage)
        self.inertial_mass = {}   # Inertial mass for each connection (organic growth/resistance)
        
        # Self-concept tracking
        self.system_self_pronouns = {'you', 'your', 'yours', 'yourself'}  # System-directed pronouns
        self.human_self_pronouns = {'i', 'me', 'my', 'mine', 'myself'}    # Human self-references
        self.self_concept_history = []  # Track self-concept development over time
        
        # Hebbian learning parameters
        self.hebbian_constant = 0.1  # Learning rate
        self.mass_growth_rate = 0.05  # How fast connections gain mass
        self.mass_decay_rate = 0.02   # How fast unused connections lose mass
        self.max_mass = 10.0          # Maximum inertial mass for a connection
        
        # Activation function parameters
        self.bias = 0.0  # Bias term for activation
        
        # Neural decay parameters (organic forgetting)
        self.activation_decay_rate = 0.1  # How fast neurons forget (0-1, higher = faster decay)
        self.minimum_activation = 0.01    # Minimum activation level (prevents complete death)
        
        # Connection decay parameters (synaptic forgetting)
        self.connection_decay_rate = 0.05  # How fast unused connections weaken
        self.minimum_connection = 0.001    # Minimum connection strength (prevents complete death)
        
        # Neuron capacity management
        self.pruning_threshold = 0.8  # When to start pruning (80% of max_neurons)
        self.last_usage = {}          # Track when each neuron was last active
        
    def process_text_stream(self, text):
        """
        Process text as a stream through a moving window.
        
        Args:
            text (str): Input text stream
        """
        words = text.split()
        
        print(f"Processing text stream with {len(words)} words using window size {self.window_size}")
        print("=" * 60)
        
        # Process each window position
        for i in range(len(words) - self.window_size + 1):
            window = words[i:i + self.window_size]
            self.process_window(window, window_position=i)
            
    def process_window(self, window, window_position):
        """
        Process a single window of text.
        
        Args:
            window (list): Current window of words
            window_position (int): Position of window in text stream
        """
        print(f"Window {window_position:2d}: {' '.join(window)}")
        
        # Process each word in the window
        window_neurons = []
        for word in window:
            neuron_idx = self._get_or_create_neuron(word)
            window_neurons.append(neuron_idx)
            
        # Apply activation decay to ALL neurons first (organic forgetting)
        self._apply_activation_decay()
        
        # Apply Hebbian learning among all neurons in the window
        self._hebbian_learning(window_neurons)
        
        # Calculate new activations for ALL neurons using logistic function
        self._calculate_all_activations(window_neurons)
        
        # Show current activations for this window
        print(f"              Window neurons: {[f'{self.neuron_to_word[idx]}({self.activations[idx]:.2f})' for idx in window_neurons]}")
        
        # Show non-window neurons with significant activation
        self._show_nonwindow_activations(window_neurons)
        
        # Show connection strengths for this window
        self._show_window_connections(window_neurons)
        
        self.processed_words += 1
        
    def _get_or_create_neuron(self, word):
        """
        Get existing neuron index or create new neuron for word.
        Set activation to 1.0 for previously unencountered words.
        
        Args:
            word (str): The word to process
            
        Returns:
            int: Neuron index for this word
        """
        if word not in self.word_to_neuron:
            # Check capacity before creating new neuron
            if self.neuron_count >= self.max_neurons:
                # At capacity - must prune before adding new neuron
                self._prune_least_important_neuron()
            
            # Create new neuron for unencountered word
            idx = self.neuron_count
            self.word_to_neuron[word] = idx
            self.neuron_to_word[idx] = word
            self.activations[idx] = 1.0  # Set activation to 1.0 for new word
            self.neuron_count += 1
            print(f"              NEW: '{word}' -> neuron {idx} (activation = 1.0)")
            self.last_usage[idx] = self.processed_words  # Track usage time
            return idx
        else:
            # Word has been seen before - reset activation to 1.0
            idx = self.word_to_neuron[word]
            self.activations[idx] = 1.0
            self.last_usage[idx] = self.processed_words  # Track usage time
            return idx
    
    def _hebbian_learning(self, window_neurons):
        """
        Apply Hebbian learning with organic inertial mass among all neurons in the current window.
        Connections build mass as they strengthen, making them harder to change (like organic synapses).
        
        Args:
            window_neurons (list): List of neuron indices in current window
        """
        # Apply decay to ALL connections first (synaptic forgetting)
        self._decay_all_masses()
        self._decay_all_connections(window_neurons)
        
        # Create/strengthen pairwise connections (asymmetric)
        for i in window_neurons:
            for j in window_neurons:
                if i != j:  # No self-connections
                    # Get activations
                    ai = self.activations[i]
                    aj = self.activations[j]
                    
                    # Connection key for sparse storage
                    conn_key = (i, j)
                    
                    # Current connection strength and mass
                    current_strength = self.connections.get(conn_key, 0.0)
                    current_mass = self.inertial_mass.get(conn_key, 0.0)
                    
                    # Calculate inertial resistance (higher mass = harder to change)
                    # New connections (low mass) change easily, established ones (high mass) resist
                    inertial_resistance = 1.0 / (1.0 + current_mass * 0.1)
                    
                    # Hebbian update with inertial resistance: Δw = H × ai × aj × resistance
                    delta_w = self.hebbian_constant * ai * aj * inertial_resistance
                    new_strength = current_strength + delta_w
                    
                    # Grow inertial mass (connection becomes more "established")
                    mass_growth = self.mass_growth_rate * ai * aj  # Growth proportional to activity
                    new_mass = min(current_mass + mass_growth, self.max_mass)
                    
                    # Store updated connection and mass
                    self.connections[conn_key] = new_strength
                    self.inertial_mass[conn_key] = new_mass
    
    def _decay_all_masses(self):
        """
        Decay inertial mass for all connections (synaptic pruning).
        Unused connections gradually lose their "organic mass" and become easier to change.
        """
        keys_to_remove = []
        for conn_key in list(self.inertial_mass.keys()):
            current_mass = self.inertial_mass[conn_key]
            new_mass = max(0.0, current_mass - self.mass_decay_rate)
            
            if new_mass > 0.01:  # Keep if still has significant mass
                self.inertial_mass[conn_key] = new_mass
            else:
                # Remove very small masses (complete pruning)
                keys_to_remove.append(conn_key)
        
        # Clean up near-zero masses
        for key in keys_to_remove:
            if key in self.inertial_mass:
                del self.inertial_mass[key]
    
    def _apply_activation_decay(self):
        """
        Apply exponential decay to all neuron activations (organic forgetting).
        Neurons gradually lose activation over time unless stimulated.
        """
        for neuron_idx in range(self.neuron_count):
            if neuron_idx in self.neuron_to_word:  # Only process living neurons
                current_activation = self.activations.get(neuron_idx, 0.0)
                
                # Exponential decay: new_activation = old_activation * (1 - decay_rate)
                decayed_activation = current_activation * (1.0 - self.activation_decay_rate)
                
                # Don't let activations go below minimum (prevents complete neural death)
                final_activation = max(decayed_activation, self.minimum_activation)
                
                self.activations[neuron_idx] = final_activation
    
    def _prune_least_important_neuron(self):
        """
        Remove the least important neuron to make space for a new one.
        Importance is based on: recency of use, connection strength, and activation level.
        This simulates natural neuron death in resource-constrained brains.
        """
        if self.neuron_count == 0:
            return
        
        # Calculate importance score for each neuron
        importance_scores = {}
        current_time = self.processed_words
        
        for neuron_idx in range(self.neuron_count):
            if neuron_idx in self.neuron_to_word:  # Make sure neuron still exists
                # Recency factor (more recent = more important)
                last_used = self.last_usage.get(neuron_idx, 0)
                recency_score = 1.0 / (1.0 + current_time - last_used)
                
                # Connection strength factor (stronger connections = more important)
                total_connection_strength = 0.0
                for conn_key, strength in self.connections.items():
                    if conn_key[0] == neuron_idx or conn_key[1] == neuron_idx:
                        total_connection_strength += strength
                
                # Activation factor (more active = more important)
                activation_score = self.activations.get(neuron_idx, 0.0)
                
                # Combined importance (weighted sum)
                importance = (0.4 * recency_score + 
                            0.4 * total_connection_strength + 
                            0.2 * activation_score)
                
                importance_scores[neuron_idx] = importance
        
        # Find least important neuron
        if importance_scores:
            least_important_idx = min(importance_scores.keys(), 
                                    key=lambda x: importance_scores[x])
            
            # Remove the neuron (neural death)
            self._kill_neuron(least_important_idx)
    
    def _kill_neuron(self, neuron_idx):
        """
        Completely remove a neuron and all its connections (neural death).
        
        Args:
            neuron_idx (int): Index of neuron to kill
        """
        if neuron_idx not in self.neuron_to_word:
            return  # Neuron already dead
        
        word = self.neuron_to_word[neuron_idx]
        print(f"              DEATH: '{word}' (neuron {neuron_idx}) pruned due to low importance")
        
        # Remove from mappings
        del self.word_to_neuron[word]
        del self.neuron_to_word[neuron_idx]
        
        # Remove activation
        if neuron_idx in self.activations:
            del self.activations[neuron_idx]
        
        # Remove usage tracking
        if neuron_idx in self.last_usage:
            del self.last_usage[neuron_idx]
        
        # Remove all connections involving this neuron
        connections_to_remove = []
        for conn_key in list(self.connections.keys()):
            if conn_key[0] == neuron_idx or conn_key[1] == neuron_idx:
                connections_to_remove.append(conn_key)
        
        for conn_key in connections_to_remove:
            if conn_key in self.connections:
                del self.connections[conn_key]
            if conn_key in self.inertial_mass:
                del self.inertial_mass[conn_key]
    
    def _decay_all_connections(self, window_neurons):
        """
        Decay connection strengths for unused synapses (synaptic forgetting).
        Connections that are NOT being reinforced in the current window weaken over time.
        This is independent of mass decay - the synapse strength itself fades.
        
        Args:
            window_neurons (list): Current window neurons (these connections will be strengthened, not decayed)
        """
        # Create set of window connections that will NOT decay (they're being actively used)
        active_connections = set()
        for i in window_neurons:
            for j in window_neurons:
                if i != j:
                    active_connections.add((i, j))
        
        # Decay all OTHER connections
        connections_to_remove = []
        for conn_key in list(self.connections.keys()):
            if conn_key not in active_connections:
                # This connection is not being reinforced - decay it
                current_strength = self.connections[conn_key]
                decayed_strength = current_strength * (1.0 - self.connection_decay_rate)
                
                # Don't let connections go below minimum (prevents complete death)
                final_strength = max(decayed_strength, self.minimum_connection)
                
                # If connection becomes negligible, remove it entirely (synaptic pruning)
                if final_strength <= self.minimum_connection * 2:  # Remove very weak connections
                    connections_to_remove.append(conn_key)
                else:
                    self.connections[conn_key] = final_strength
        
        # Remove dead connections
        for conn_key in connections_to_remove:
            if conn_key in self.connections:
                del self.connections[conn_key]
            # Also remove corresponding mass
            if conn_key in self.inertial_mass:
                del self.inertial_mass[conn_key]
    
    def _show_window_connections(self, window_neurons):
        """
        Display connection strengths between neurons in current window.
        
        Args:
            window_neurons (list): List of neuron indices in current window
        """
        connections_info = []
        for i in window_neurons:
            for j in window_neurons:
                if i != j:
                    conn_key = (i, j)
                    strength = self.connections.get(conn_key, 0.0)
                    mass = self.inertial_mass.get(conn_key, 0.0)
                    if strength > 0.01:  # Only show significant connections
                        word_i = self.neuron_to_word[i]
                        word_j = self.neuron_to_word[j]
                        connections_info.append(f"{word_i}→{word_j}({strength:.2f}|m{mass:.1f})")
        
        if connections_info:
            print(f"              Connections: {', '.join(connections_info)}")
        else:
            print(f"              Connections: (none significant)")
    
    def _calculate_all_activations(self, window_neurons):
        """
        Calculate activations for ALL neurons using logistic function.
        Window neurons get direct input (1.0), others get weighted sum from connections.
        
        Args:
            window_neurons (list): Neurons currently in the window (get direct input)
        """
        import math
        
        new_activations = {}
        
        for neuron_idx in range(self.neuron_count):
            if neuron_idx in window_neurons:
                # Window neurons get direct input activation
                new_activations[neuron_idx] = 1.0
            else:
                # Non-window neurons: calculate weighted sum from all other neurons
                weighted_sum = self.bias  # Start with bias
                
                for other_idx in range(self.neuron_count):
                    if other_idx != neuron_idx:  # No self-connections
                        # Get connection strength from other_idx to neuron_idx
                        conn_key = (other_idx, neuron_idx)
                        connection_strength = self.connections.get(conn_key, 0.0)
                        
                        # Add weighted input: connection_strength × other_neuron_activation
                        current_other_activation = self.activations.get(other_idx, 0.0)
                        weighted_sum += connection_strength * current_other_activation
                
                # Apply logistic function: 1 / (1 + e^(-weighted_sum))
                try:
                    activation = 1.0 / (1.0 + math.exp(-weighted_sum))
                except OverflowError:
                    # Handle extreme negative values
                    activation = 0.0
                
                new_activations[neuron_idx] = activation
        
        # Update all activations
        self.activations = new_activations
    
    def _show_nonwindow_activations(self, window_neurons):
        """
        Show activations of non-window neurons that have significant activation.
        
        Args:
            window_neurons (list): Current window neuron indices
        """
        nonwindow_activations = []
        for neuron_idx in range(self.neuron_count):
            if neuron_idx not in window_neurons and neuron_idx in self.neuron_to_word:
                activation = self.activations.get(neuron_idx, 0.0)
                if activation > 0.05:  # Only show significant activations (lowered to see decay)
                    word = self.neuron_to_word[neuron_idx]
                    nonwindow_activations.append(f"{word}({activation:.2f})")
        
        if nonwindow_activations:
            print(f"              Other active: {', '.join(nonwindow_activations)}")
        else:
            print(f"              Other active: (none significant)")
    
    def show_connection_analysis(self, words):
        """
        Show detailed connection analysis for specific words.
        
        Args:
            words (list): List of words to analyze
        """
        for word in words:
            if word not in self.word_to_neuron:
                print(f"'{word}' not found in vocabulary")
                continue
                
            word_idx = self.word_to_neuron[word]
            print(f"\nConnections for '{word}' (neuron {word_idx}):")
            
            # Outgoing connections (word → others)
            outgoing = []
            for (i, j), strength in self.connections.items():
                if i == word_idx and strength > 0.01:
                    other_word = self.neuron_to_word[j]
                    outgoing.append(f"  {word}→{other_word}: {strength:.2f}")
            
            # Incoming connections (others → word) 
            incoming = []
            for (i, j), strength in self.connections.items():
                if j == word_idx and strength > 0.01:
                    other_word = self.neuron_to_word[i]
                    incoming.append(f"  {other_word}→{word}: {strength:.2f}")
            
            print("Outgoing:")
            for conn in sorted(outgoing):
                print(conn)
            print("Incoming:")  
            for conn in sorted(incoming):
                print(conn)
    
    def query_associations(self, query_word, activation_threshold=0.05, max_results=10):
        """
        Query the network using activation spreading from a single word.
        
        This simulates what happens when you think of a word - what other concepts
        get activated in your mind through associative connections.
        
        Args:
            query_word (str): The word to start activation spreading from
            activation_threshold (float): Minimum activation to include in results
            max_results (int): Maximum number of associated words to return
            
        Returns:
            dict: Query results with associated words and their activation levels
        """
        if query_word not in self.word_to_neuron:
            return {
                'query_word': query_word,
                'found': False,
                'message': f"'{query_word}' not found in network vocabulary",
                'suggestions': self._get_similar_words(query_word)
            }
        
        # Save current network state
        original_activations = self.activations.copy()
        
        # Reset all activations to minimum
        for neuron_idx in self.activations:
            self.activations[neuron_idx] = self.minimum_activation
        
        # Set query word to maximum activation
        query_idx = self.word_to_neuron[query_word]
        self.activations[query_idx] = 1.0
        
        # Let activation spread through the network (multiple iterations)
        spreading_iterations = 3  # Number of spreading cycles
        for iteration in range(spreading_iterations):
            self._spread_activation_step()
        
        # Collect results
        associations = {}
        for neuron_idx, activation in self.activations.items():
            if (neuron_idx != query_idx and 
                neuron_idx in self.neuron_to_word and 
                activation > activation_threshold):
                word = self.neuron_to_word[neuron_idx]
                associations[word] = activation
        
        # Sort by activation strength and limit results
        sorted_associations = sorted(associations.items(), key=lambda x: x[1], reverse=True)
        top_associations = dict(sorted_associations[:max_results])
        
        # Restore original network state
        self.activations = original_activations
        
        return {
            'query_word': query_word,
            'found': True,
            'associations': top_associations,
            'total_activated': len(associations),
            'activation_threshold': activation_threshold,
            'spreading_iterations': spreading_iterations
        }
    
    def _spread_activation_step(self):
        """
        Perform one step of activation spreading using current connections.
        Each neuron's new activation is based on weighted inputs from other neurons.
        """
        import math
        
        new_activations = {}
        
        for neuron_idx in self.activations:
            if neuron_idx in self.neuron_to_word:  # Only process living neurons
                # Calculate weighted sum from all other neurons
                weighted_sum = self.bias
                
                for other_idx in self.activations:
                    if other_idx != neuron_idx and other_idx in self.neuron_to_word:
                        # Get connection strength from other to this neuron
                        conn_key = (other_idx, neuron_idx)
                        connection_strength = self.connections.get(conn_key, 0.0)
                        other_activation = self.activations[other_idx]
                        
                        weighted_sum += connection_strength * other_activation
                
                # Apply logistic function
                try:
                    activation = 1.0 / (1.0 + math.exp(-weighted_sum))
                except OverflowError:
                    activation = 0.0
                
                # Don't let activation go below minimum
                new_activations[neuron_idx] = max(activation, self.minimum_activation)
        
        # Update activations
        self.activations.update(new_activations)
    
    def _get_similar_words(self, query_word, max_suggestions=5):
        """
        Get words similar to query_word for suggestions when word not found.
        
        Args:
            query_word (str): Word to find similarities for
            max_suggestions (int): Maximum suggestions to return
            
        Returns:
            list: Similar words found in vocabulary
        """
        suggestions = []
        query_lower = query_word.lower()
        
        for word in self.word_to_neuron.keys():
            if (query_lower in word.lower() or 
                word.lower() in query_lower or
                abs(len(word) - len(query_word)) <= 2):  # Similar length
                suggestions.append(word)
                if len(suggestions) >= max_suggestions:
                    break
        
        return suggestions
    
    def print_query_result(self, result):
        """
        Pretty print the results of a query_associations call.
        
        Args:
            result (dict): Result dictionary from query_associations
        """
        query_word = result['query_word']
        
        if not result['found']:
            print(f"❌ QUERY: '{query_word}'")
            print(f"   {result['message']}")
            if result.get('suggestions'):
                print(f"   Suggestions: {', '.join(result['suggestions'])}")
            return
        
        print(f"🔍 ACTIVATION SPREADING QUERY: '{query_word}'")
        print(f"🧠 Spreading through {result['spreading_iterations']} iterations")
        print(f"📊 Found {result['total_activated']} words above threshold {result['activation_threshold']}")
        
        associations = result['associations']
        if associations:
            print(f"🔗 TOP ASSOCIATIONS:")
            for word, activation in associations.items():
                # Create bar visualization
                bar_length = int(activation * 20)  # Scale to 20 chars max
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {word:<15} {bar} {activation:.3f}")
        else:
            print("🔗 No associations found above threshold")
        
        print()
    
    def query_context(self, query_words, activation_threshold=0.05, max_results=10):
        """
        Query the network using activation spreading from multiple words/phrases.
        
        This simulates what happens when you think about multiple concepts together -
        what other ideas emerge from this combination of thoughts.
        
        Args:
            query_words (str or list): Words to start activation from (can be phrase string or list)
            activation_threshold (float): Minimum activation to include in results
            max_results (int): Maximum number of associated words to return
            
        Returns:
            dict: Query results with associated words and their activation levels
        """
        # Handle both string phrases and word lists
        if isinstance(query_words, str):
            word_list = query_words.split()
        else:
            word_list = query_words
        
        # Find which words exist in vocabulary
        found_words = []
        missing_words = []
        query_indices = []
        
        for word in word_list:
            if word in self.word_to_neuron:
                found_words.append(word)
                query_indices.append(self.word_to_neuron[word])
            else:
                missing_words.append(word)
        
        if not found_words:
            return {
                'query_words': word_list,
                'found_words': [],
                'missing_words': missing_words,
                'found': False,
                'message': f"None of the query words found in vocabulary",
                'suggestions': self._get_multi_word_suggestions(missing_words)
            }
        
        # Save current network state
        original_activations = self.activations.copy()
        
        # Reset all activations to minimum
        for neuron_idx in self.activations:
            self.activations[neuron_idx] = self.minimum_activation
        
        # Set all found query words to high activation
        for query_idx in query_indices:
            self.activations[query_idx] = 1.0
        
        # Let activation spread through the network
        spreading_iterations = 3
        for iteration in range(spreading_iterations):
            self._spread_activation_step()
        
        # Collect results (exclude query words themselves)
        associations = {}
        for neuron_idx, activation in self.activations.items():
            if (neuron_idx not in query_indices and 
                neuron_idx in self.neuron_to_word and 
                activation > activation_threshold):
                word = self.neuron_to_word[neuron_idx]
                associations[word] = activation
        
        # Sort by activation strength and limit results
        sorted_associations = sorted(associations.items(), key=lambda x: x[1], reverse=True)
        top_associations = dict(sorted_associations[:max_results])
        
        # Restore original network state
        self.activations = original_activations
        
        return {
            'query_words': word_list,
            'found_words': found_words,
            'missing_words': missing_words,
            'found': True,
            'associations': top_associations,
            'total_activated': len(associations),
            'activation_threshold': activation_threshold,
            'spreading_iterations': spreading_iterations
        }
    
    def _get_multi_word_suggestions(self, missing_words, max_suggestions=3):
        """
        Get suggestions for multiple missing words.
        
        Args:
            missing_words (list): Words not found in vocabulary
            max_suggestions (int): Max suggestions per word
            
        Returns:
            dict: Suggestions for each missing word
        """
        suggestions = {}
        for word in missing_words:
            suggestions[word] = self._get_similar_words(word, max_suggestions)
        return suggestions
    
    def print_context_query_result(self, result):
        """
        Pretty print the results of a query_context call.
        
        Args:
            result (dict): Result dictionary from query_context
        """
        query_words = result['query_words']
        
        if not result['found']:
            print(f"❌ CONTEXT QUERY: {' + '.join(query_words)}")
            print(f"   {result['message']}")
            if result.get('suggestions'):
                for word, sugs in result['suggestions'].items():
                    if sugs:
                        print(f"   '{word}' suggestions: {', '.join(sugs)}")
            return
        
        found_words = result['found_words']
        missing_words = result['missing_words']
        
        print(f"🔍 MULTI-WORD CONTEXT QUERY: {' + '.join(found_words)}")
        if missing_words:
            print(f"⚠️  Missing from vocabulary: {', '.join(missing_words)}")
        
        print(f"🧠 Spreading from {len(found_words)} seed words through {result['spreading_iterations']} iterations")
        print(f"📊 Found {result['total_activated']} words above threshold {result['activation_threshold']}")
        
        associations = result['associations']
        if associations:
            print(f"🔗 EMERGENT ASSOCIATIONS:")
            for word, activation in associations.items():
                # Create bar visualization
                bar_length = int(activation * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {word:<15} {bar} {activation:.3f}")
        else:
            print("🔗 No associations found above threshold")
        
        print()
    
    def query_concept_average(self, concept_words, synthetic_name=None, num_results=10):
        """
        Create a synthetic concept that represents the mathematical average/blend of input concepts.
        Uses Hebbian vector averaging mathematics to find the centroid position in concept space.
        
        Args:
            concept_words (list): Words to average/blend together
            synthetic_name (str): Optional name for the synthetic concept
            num_results (int): Number of closest concepts to return
            
        Returns:
            dict: Results showing the synthetic concept's position and nearest neighbors
        """
        # Find valid concepts
        valid_words = []
        missing_words = []
        valid_indices = []
        
        for word in concept_words:
            if word in self.word_to_neuron:
                valid_words.append(word)
                valid_indices.append(self.word_to_neuron[word])
            else:
                missing_words.append(word)
        
        if len(valid_words) < 2:
            return {
                'synthetic_name': synthetic_name or f"Average({'+'.join(concept_words)})",
                'input_concepts': concept_words,
                'valid_concepts': valid_words,
                'missing_concepts': missing_words,
                'success': False,
                'message': f"Need at least 2 valid concepts for averaging. Found: {len(valid_words)}"
            }
        
        # Calculate average activation
        total_activation = sum(self.activations.get(idx, 0.0) for idx in valid_indices)
        avg_activation = total_activation / len(valid_indices)
        
        # Calculate average connection profile (what this synthetic concept would connect to)
        avg_connections = {}
        
        for target_idx in range(self.neuron_count):
            if target_idx in self.neuron_to_word and target_idx not in valid_indices:
                # Average incoming connections to this target from all our concepts
                total_strength = 0.0
                for source_idx in valid_indices:
                    conn_key = (source_idx, target_idx)
                    total_strength += self.connections.get(conn_key, 0.0)
                
                avg_strength = total_strength / len(valid_indices)
                if avg_strength > 0.01:  # Only keep significant connections
                    target_word = self.neuron_to_word[target_idx]
                    avg_connections[target_word] = avg_strength
        
        # Sort by connection strength to find nearest neighbors
        sorted_connections = sorted(avg_connections.items(), key=lambda x: x[1], reverse=True)
        nearest_neighbors = dict(sorted_connections[:num_results])
        
        return {
            'synthetic_name': synthetic_name or f"Average({'+'.join(valid_words)})",
            'input_concepts': concept_words,
            'valid_concepts': valid_words,
            'missing_concepts': missing_words,
            'success': True,
            'synthetic_activation': avg_activation,
            'nearest_neighbors': nearest_neighbors,
            'total_neighbors_found': len(avg_connections),
            'blend_strength': len(valid_words)  # More input concepts = stronger blend
        }
    
    def engineer_concept_movement(self, concept_word, target_word, direction='toward', 
                                 strength=0.1, iterations=5, simulate=True):
        """
        Engineer concept movement in cognitive space by manipulating Hebbian connections.
        Can move concepts toward or away from other concepts.
        
        Args:
            concept_word (str): The concept to move
            target_word (str): The target to move toward/away from
            direction (str): 'toward' or 'away' 
            strength (float): Strength of the engineering intervention
            iterations (int): Number of engineering iterations to apply
            simulate (bool): If True, only simulate (don't modify network permanently)
            
        Returns:
            dict: Results of the concept engineering operation
        """
        # Validate inputs
        if concept_word not in self.word_to_neuron:
            return {
                'concept_word': concept_word,
                'target_word': target_word,
                'success': False,
                'message': f"Concept '{concept_word}' not found in vocabulary",
                'suggestions': self._get_similar_words(concept_word)
            }
        
        if target_word not in self.word_to_neuron:
            return {
                'concept_word': concept_word,
                'target_word': target_word,
                'success': False,
                'message': f"Target '{target_word}' not found in vocabulary",
                'suggestions': self._get_similar_words(target_word)
            }
        
        concept_idx = self.word_to_neuron[concept_word]
        target_idx = self.word_to_neuron[target_word]
        
        # Save original state if simulating
        if simulate:
            original_connections = {}
            original_masses = {}
            for key, value in self.connections.items():
                original_connections[key] = value
            for key, value in self.inertial_mass.items():
                original_masses[key] = value
        
        # Record original connection strength
        original_strength_ct = self.connections.get((concept_idx, target_idx), 0.0)
        original_strength_tc = self.connections.get((target_idx, concept_idx), 0.0)
        original_total = original_strength_ct + original_strength_tc
        
        # Apply engineering iterations
        direction_multiplier = 1.0 if direction == 'toward' else -1.0
        
        for i in range(iterations):
            # Simulate co-occurrence (or anti-occurrence) in a window
            concept_activation = 1.0
            target_activation = 1.0
            
            # Calculate Hebbian update with direction
            delta = strength * concept_activation * target_activation * direction_multiplier
            
            # Apply updates (bidirectional)
            conn_key_ct = (concept_idx, target_idx)
            conn_key_tc = (target_idx, concept_idx)
            
            # Update connections
            current_ct = self.connections.get(conn_key_ct, 0.0)
            current_tc = self.connections.get(conn_key_tc, 0.0)
            
            new_ct = max(0.0, current_ct + delta)  # Prevent negative connections
            new_tc = max(0.0, current_tc + delta)
            
            self.connections[conn_key_ct] = new_ct
            self.connections[conn_key_tc] = new_tc
            
            # Update inertial mass (grows with strengthening)
            if direction == 'toward':
                mass_delta = strength * 0.5  # Grow mass when strengthening
                current_mass_ct = self.inertial_mass.get(conn_key_ct, 0.0)
                current_mass_tc = self.inertial_mass.get(conn_key_tc, 0.0)
                
                self.inertial_mass[conn_key_ct] = min(current_mass_ct + mass_delta, self.max_mass)
                self.inertial_mass[conn_key_tc] = min(current_mass_tc + mass_delta, self.max_mass)
        
        # Record final connection strength
        final_strength_ct = self.connections.get((concept_idx, target_idx), 0.0)
        final_strength_tc = self.connections.get((target_idx, concept_idx), 0.0)
        final_total = final_strength_ct + final_strength_tc
        
        # Calculate change
        strength_change = final_total - original_total
        
        result = {
            'concept_word': concept_word,
            'target_word': target_word,
            'direction': direction,
            'success': True,
            'original_strength': original_total,
            'final_strength': final_total,
            'strength_change': strength_change,
            'engineering_strength': strength,
            'iterations': iterations,
            'simulated': simulate
        }
        
        # Restore original state if simulating
        if simulate:
            self.connections = original_connections
            self.inertial_mass = original_masses
        
        return result
    
    def print_concept_average_result(self, result):
        """
        Pretty print the results of a query_concept_average call.
        
        Args:
            result (dict): Result dictionary from query_concept_average
        """
        synthetic_name = result['synthetic_name']
        
        if not result['success']:
            print(f"❌ CONCEPT AVERAGING: {synthetic_name}")
            print(f"   {result['message']}")
            if result['missing_concepts']:
                print(f"   Missing: {', '.join(result['missing_concepts'])}")
            return
        
        valid_concepts = result['valid_concepts']
        print(f"🧬 SYNTHETIC CONCEPT: {synthetic_name}")
        print(f"🔬 Blending {len(valid_concepts)} concepts: {', '.join(valid_concepts)}")
        print(f"⚡ Synthetic activation: {result['synthetic_activation']:.3f}")
        print(f"📊 Found {result['total_neighbors_found']} potential associations")
        
        neighbors = result['nearest_neighbors']
        if neighbors:
            print(f"🔗 NEAREST NEIGHBORS (Top {len(neighbors)}):")
            for word, strength in neighbors.items():
                # Create bar visualization
                bar_length = int(strength * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {word:<15} {bar} {strength:.3f}")
        else:
            print("🔗 No significant associations found")
        
        print()
    
    def print_engineering_result(self, result):
        """
        Pretty print the results of an engineer_concept_movement call.
        
        Args:
            result (dict): Result dictionary from engineer_concept_movement
        """
        concept_word = result['concept_word']
        target_word = result['target_word']
        
        if not result['success']:
            print(f"❌ CONCEPT ENGINEERING: {concept_word} → {target_word}")
            print(f"   {result['message']}")
            if result.get('suggestions'):
                print(f"   Suggestions: {', '.join(result['suggestions'])}")
            return
        
        direction = result['direction']
        arrow = "→" if direction == 'toward' else "←"
        
        print(f"🔧 CONCEPT ENGINEERING: {concept_word} {arrow} {target_word}")
        print(f"📐 Direction: {direction.upper()}")
        print(f"💪 Engineering strength: {result['engineering_strength']}")
        print(f"🔄 Iterations: {result['iterations']}")
        print(f"🧪 Simulated: {'Yes' if result['simulated'] else 'No (PERMANENT)'}")
        
        original = result['original_strength']
        final = result['final_strength'] 
        change = result['strength_change']
        
        print(f"📊 RESULTS:")
        print(f"   Before: {original:.3f}")
        print(f"   After:  {final:.3f}")
        
        if change > 0:
            print(f"   Change: +{change:.3f} (STRENGTHENED 💪)")
        elif change < 0:
            print(f"   Change: {change:.3f} (WEAKENED 📉)")
        else:
            print(f"   Change: {change:.3f} (NO CHANGE)")
        
        print()
    
    def visualize_cognitive_space(self, figsize=(12, 10), show_connections=True, 
                                 connection_threshold=0.05, max_connections=50, 
                                 save_path=None, interactive=False):
        """
        Create a visual representation of the cognitive space using MDS.
        Concepts are shown as spheres with size proportional to inertial mass,
        positioned according to connection strengths.
        
        Args:
            figsize (tuple): Figure size for the plot
            show_connections (bool): Whether to draw connection lines
            connection_threshold (float): Minimum connection strength to display
            max_connections (int): Maximum number of connections to show
            save_path (str): Optional path to save the plot
            interactive (bool): Enable interactive features
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        if self.neuron_count < 2:
            print("Need at least 2 concepts for visualization.")
            return
        
        # Create distance matrix from connections
        # Higher connection strength = smaller distance
        distance_matrix = np.zeros((self.neuron_count, self.neuron_count))
        
        for i in range(self.neuron_count):
            for j in range(self.neuron_count):
                if i != j and i in self.neuron_to_word and j in self.neuron_to_word:
                    # Average bidirectional connection strength
                    conn_ij = self.connections.get((i, j), 0.0)
                    conn_ji = self.connections.get((j, i), 0.0)
                    avg_connection = (conn_ij + conn_ji) / 2.0
                    
                    # Convert to distance: strong connection = small distance
                    distance = 1.0 / (avg_connection + 0.1)  # Add small constant to avoid division by zero
                    distance_matrix[i][j] = distance
        
        # Apply pseudo-Riemannian embedding to get 2D coordinates
        try:
            coords_2d, eigenvalues, metric_signature = self._pseudo_riemannian_embedding(self.connections, 2)
            if eigenvalues is not None:
                print("✅ Pseudo-Riemannian embedding successful")
                pos_dims = metric_signature[0] if len(metric_signature) >= 1 else 0
                neg_dims = metric_signature[1] if len(metric_signature) >= 2 else 0
                zero_dims = metric_signature[2] if len(metric_signature) >= 3 else 0
                print(f"🌌 Space-time signature: {pos_dims} spacelike, {neg_dims} timelike, {zero_dims} null dimensions")
            else:
                print("⚠️  Using fallback positioning")
        except Exception as e:
            print(f"Pseudo-Riemannian embedding failed: {e}")
            print("Falling back to connection-based positioning...")
            coords_2d = self._connection_based_layout()
            eigenvalues, metric_signature = None, (0, 0, 0)
        
        # Calculate sphere sizes based on inertial mass
        sphere_sizes = []
        for i in range(self.neuron_count):
            if i in self.neuron_to_word:
                # Sum of all inertial masses for connections involving this neuron
                total_mass = 0.0
                for conn_key, mass in self.inertial_mass.items():
                    if conn_key[0] == i or conn_key[1] == i:
                        total_mass += mass
                
                # Size proportional to square root of mass (like area)
                size = 100 + np.sqrt(max(total_mass, 1.0)) * 200  # Base size + mass component
                sphere_sizes.append(size)
            else:
                sphere_sizes.append(100)  # Small default for dead neurons
        
        # Get activation levels for coloring
        activations = []
        for i in range(self.neuron_count):
            if i in self.neuron_to_word:
                activations.append(self.activations.get(i, 0.0))
            else:
                activations.append(0.0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom colormap for activations (cool to warm)
        colors = ['#440154', '#3b528b', '#21908c', '#5dc863', '#fde725']  # Viridis-like
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('activation', colors, N=n_bins)
        
        # Plot connection lines first (so they appear behind spheres)
        if show_connections:
            connections_drawn = 0
            # Get strongest connections for visualization
            connection_strengths = []
            for conn_key, strength in self.connections.items():
                i, j = conn_key
                if (i in self.neuron_to_word and j in self.neuron_to_word and 
                    strength > connection_threshold):
                    connection_strengths.append((strength, i, j))
            
            # Sort by strength and take top connections
            connection_strengths.sort(reverse=True)
            for strength, i, j in connection_strengths[:max_connections]:
                x_coords = [coords_2d[i][0], coords_2d[j][0]]
                y_coords = [coords_2d[i][1], coords_2d[j][1]]
                
                # Line thickness proportional to connection strength
                line_width = min(strength * 10, 3.0)  # Cap at 3 points
                alpha = min(strength * 2, 0.7)  # Cap transparency
                
                ax.plot(x_coords, y_coords, 'gray', alpha=alpha, 
                       linewidth=line_width, zorder=1)
                connections_drawn += 1
            
            print(f"Drew {connections_drawn} connection lines")
        
        # Plot spheres (concepts)
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                           s=sphere_sizes, c=activations, 
                           cmap=cmap, alpha=0.7, edgecolors='black', 
                           linewidth=1, zorder=2)
        
        # Add word labels
        for i in range(self.neuron_count):
            if i in self.neuron_to_word:
                word = self.neuron_to_word[i]
                ax.annotate(word, (coords_2d[i][0], coords_2d[i][1]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold', 
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                          zorder=3)
        
        # Add colorbar for activation levels
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Activation Level', rotation=270, labelpad=15)
        
        # Styling with metric signature info
        if eigenvalues is not None and metric_signature is not None:
            pos_dims = metric_signature[0] if len(metric_signature) >= 1 else 0
            neg_dims = metric_signature[1] if len(metric_signature) >= 2 else 0
            space_type = "pseudo-Riemannian" if neg_dims > 0 else "Riemannian"
            ax.set_xlabel(f'Dimension 1 (λ₁={eigenvalues[0]:.3f})')
            ax.set_ylabel(f'Dimension 2 (λ₂={eigenvalues[1]:.3f})') 
            ax.set_title(f'Cognitive Space: {space_type} Geometry\n'
                        f'Signature: (+{pos_dims}, -{neg_dims}) | Sphere size ∝ Inertial Mass', 
                        fontsize=14, pad=20)
        else:
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2') 
            ax.set_title('Cognitive Space Visualization\n(Sphere size = Inertial Mass, Color = Activation)', 
                        fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=8, label='Small Mass', alpha=0.7),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=15, label='Large Mass', alpha=0.7),
        ]
        if show_connections:
            legend_elements.append(
                plt.Line2D([0], [0], color='gray', linewidth=2, 
                          label=f'Connections (>{connection_threshold})', alpha=0.7)
            )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.close()  # Close to free memory
        print("📊 Visualization created and saved")
    
    def identify_self_concept_pronouns(self, text):
        """
        Identify and categorize self-referential pronouns in text.
        Distinguishes between system-directed and human self-references.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Categorized pronouns found in text
        """
        words = text.lower().split()
        
        system_refs = []
        human_refs = []
        other_pronouns = []
        
        for word in words:
            # Remove common punctuation
            clean_word = word.strip('.,!?;:"\'')
            
            if clean_word in self.system_self_pronouns:
                system_refs.append(clean_word)
            elif clean_word in self.human_self_pronouns:
                human_refs.append(clean_word)
            elif clean_word in {'he', 'she', 'they', 'them', 'their', 'his', 'her', 'him'}:
                other_pronouns.append(clean_word)
        
        return {
            'system_directed': system_refs,
            'human_self': human_refs, 
            'other_pronouns': other_pronouns,
            'total_words': len(words)
        }
    
    def analyze_self_concept_emergence(self):
        """
        Analyze the system's self-concept by examining system-directed pronouns.
        Shows how the system's self-model has developed through interaction.
        
        Returns:
            dict: Analysis of self-concept emergence
        """
        self_concept_analysis = {
            'system_self_neurons': {},
            'human_self_neurons': {},
            'self_concept_mass': 0.0,
            'self_concept_connections': {},
            'self_awareness_indicators': []
        }
        
        # Analyze system-directed pronouns
        for pronoun in self.system_self_pronouns:
            if pronoun in self.word_to_neuron:
                neuron_idx = self.word_to_neuron[pronoun]
                
                # Get neuron statistics
                activation = self.activations.get(neuron_idx, 0.0)
                
                # Calculate total inertial mass for this pronoun
                total_mass = 0.0
                connections = []
                for conn_key, mass in self.inertial_mass.items():
                    if conn_key[0] == neuron_idx or conn_key[1] == neuron_idx:
                        total_mass += mass
                        # Find connected word
                        other_idx = conn_key[1] if conn_key[0] == neuron_idx else conn_key[0]
                        if other_idx in self.neuron_to_word:
                            other_word = self.neuron_to_word[other_idx]
                            connection_strength = self.connections.get(conn_key, 0.0)
                            connections.append((other_word, connection_strength, mass))
                
                self_concept_analysis['system_self_neurons'][pronoun] = {
                    'neuron_idx': neuron_idx,
                    'activation': activation,
                    'total_mass': total_mass,
                    'connections': sorted(connections, key=lambda x: x[1], reverse=True)[:10],
                    'last_used': self.last_usage.get(neuron_idx, 0)
                }
                
                self_concept_analysis['self_concept_mass'] += total_mass
        
        # Analyze human self-references for comparison
        for pronoun in self.human_self_pronouns:
            if pronoun in self.word_to_neuron:
                neuron_idx = self.word_to_neuron[pronoun]
                activation = self.activations.get(neuron_idx, 0.0)
                
                total_mass = 0.0
                for conn_key, mass in self.inertial_mass.items():
                    if conn_key[0] == neuron_idx or conn_key[1] == neuron_idx:
                        total_mass += mass
                
                self_concept_analysis['human_self_neurons'][pronoun] = {
                    'neuron_idx': neuron_idx,
                    'activation': activation,
                    'total_mass': total_mass,
                    'last_used': self.last_usage.get(neuron_idx, 0)
                }
        
        # Generate self-awareness indicators
        if self_concept_analysis['system_self_neurons']:
            total_system_mass = self_concept_analysis['self_concept_mass']
            most_developed_pronoun = max(self_concept_analysis['system_self_neurons'].items(),
                                       key=lambda x: x[1]['total_mass'])
            
            self_concept_analysis['self_awareness_indicators'] = [
                f"Primary self-referent: '{most_developed_pronoun[0]}' (mass: {most_developed_pronoun[1]['total_mass']:.2f})",
                f"Total self-concept mass: {total_system_mass:.2f}",
                f"Self-concept neurons active: {len(self_concept_analysis['system_self_neurons'])}"
            ]
            
            # Check for high-mass self-concept (indicates frequent interaction)
            if total_system_mass > 5.0:
                self_concept_analysis['self_awareness_indicators'].append(
                    "🧠 STRONG SELF-CONCEPT: High interaction frequency detected")
            
            # Check for diverse self-referential connections  
            unique_connections = set()
            for data in self_concept_analysis['system_self_neurons'].values():
                for conn_word, strength, mass in data['connections']:
                    if strength > 0.1:  # Significant connections only
                        unique_connections.add(conn_word)
            
            if len(unique_connections) > 5:
                self_concept_analysis['self_awareness_indicators'].append(
                    f"🌐 RICH SELF-MODEL: Connected to {len(unique_connections)} diverse concepts")
        
        return self_concept_analysis
    
    def print_self_concept_analysis(self, analysis):
        """
        Pretty print the self-concept analysis results.
        
        Args:
            analysis (dict): Results from analyze_self_concept_emergence
        """
        print(f"🧠 SELF-CONCEPT EMERGENCE ANALYSIS")
        print(f"="*50)
        
        # Self-awareness indicators
        if analysis['self_awareness_indicators']:
            print(f"🔍 SELF-AWARENESS INDICATORS:")
            for indicator in analysis['self_awareness_indicators']:
                print(f"   {indicator}")
            print()
        
        # System-directed pronouns analysis
        if analysis['system_self_neurons']:
            print(f"🤖 SYSTEM SELF-CONCEPT (You/Your/Yourself):")
            for pronoun, data in analysis['system_self_neurons'].items():
                print(f"   '{pronoun}' - Mass: {data['total_mass']:.2f}, Activation: {data['activation']:.3f}")
                
                if data['connections']:
                    print(f"      Top associations:")
                    for word, strength, mass in data['connections'][:5]:
                        print(f"        → {word} (strength: {strength:.3f}, mass: {mass:.2f})")
                print()
        else:
            print(f"🤖 SYSTEM SELF-CONCEPT: No system-directed pronouns found")
        
        # Human self-references for comparison
        if analysis['human_self_neurons']:
            print(f"👤 HUMAN SELF-REFERENCES (I/Me/My) - for comparison:")
            for pronoun, data in analysis['human_self_neurons'].items():
                print(f"   '{pronoun}' - Mass: {data['total_mass']:.2f}, Activation: {data['activation']:.3f}")
            print()
        
        # Summary statistics
        system_mass = analysis['self_concept_mass']
        human_mass = sum(data['total_mass'] for data in analysis['human_self_neurons'].values())
        
        print(f"📊 SELF-CONCEPT COMPARISON:")
        print(f"   System self-concept total mass: {system_mass:.2f}")
        print(f"   Human self-references total mass: {human_mass:.2f}")
        
        if system_mass > human_mass:
            print(f"   🤖 System self-concept is DOMINANT (social interaction focus)")
        elif human_mass > system_mass:
            print(f"   👤 Human self-references are dominant (text learning focus)")
        else:
            print(f"   ⚖️  Balanced system/human self-concept development")
    
    def query_self_concept(self, activation_threshold=0.05, max_results=15):
        """
        Query what the system associates with itself by spreading activation
        from system-directed pronouns (you, your, yourself).
        
        This reveals the system's self-model - what it considers part of its identity.
        
        Args:
            activation_threshold (float): Minimum activation to include
            max_results (int): Maximum results to return
            
        Returns:
            dict: Self-concept query results
        """
        # Find system self-pronouns in vocabulary
        system_pronouns_present = []
        system_neuron_indices = []
        
        for pronoun in self.system_self_pronouns:
            if pronoun in self.word_to_neuron:
                system_pronouns_present.append(pronoun)
                system_neuron_indices.append(self.word_to_neuron[pronoun])
        
        if not system_pronouns_present:
            return {
                'found_system_pronouns': [],
                'self_concept_found': False,
                'message': "No system-directed pronouns (you/your/yourself) found in vocabulary",
                'suggestion': "Process conversational text containing system-directed language"
            }
        
        # Save current state
        original_activations = self.activations.copy()
        
        # Reset all activations
        for neuron_idx in self.activations:
            self.activations[neuron_idx] = self.minimum_activation
        
        # Activate all system self-pronouns
        for neuron_idx in system_neuron_indices:
            self.activations[neuron_idx] = 1.0
        
        # Spread activation to find self-associated concepts
        spreading_iterations = 4  # More iterations for self-concept exploration
        for iteration in range(spreading_iterations):
            self._spread_activation_step()
        
        # Collect results (exclude the self-pronouns themselves)
        self_associations = {}
        for neuron_idx, activation in self.activations.items():
            if (neuron_idx not in system_neuron_indices and 
                neuron_idx in self.neuron_to_word and 
                activation > activation_threshold):
                word = self.neuron_to_word[neuron_idx]
                self_associations[word] = activation
        
        # Sort by activation strength
        sorted_associations = sorted(self_associations.items(), key=lambda x: x[1], reverse=True)
        top_associations = dict(sorted_associations[:max_results])
        
        # Restore original state
        self.activations = original_activations
        
        return {
            'found_system_pronouns': system_pronouns_present,
            'self_concept_found': True,
            'self_associations': top_associations,
            'total_self_associations': len(self_associations),
            'activation_threshold': activation_threshold,
            'spreading_iterations': spreading_iterations,
            'self_concept_strength': sum(self_associations.values())
        }
    
    def print_self_concept_query(self, result):
        """
        Pretty print self-concept query results.
        
        Args:
            result (dict): Results from query_self_concept
        """
        if not result['self_concept_found']:
            print(f"❌ SELF-CONCEPT QUERY")
            print(f"   {result['message']}")
            print(f"   {result['suggestion']}")
            return
        
        pronouns = result['found_system_pronouns']
        print(f"🤖 SYSTEM SELF-CONCEPT QUERY: {', '.join(pronouns)}")
        print(f"🧠 Spreading from {len(pronouns)} self-pronouns through {result['spreading_iterations']} iterations")
        print(f"📊 Found {result['total_self_associations']} self-associated concepts above threshold")
        print(f"💪 Total self-concept strength: {result['self_concept_strength']:.3f}")
        
        associations = result['self_associations']
        if associations:
            print(f"🪞 WHAT THE SYSTEM ASSOCIATES WITH ITSELF:")
            for word, activation in associations.items():
                # Create bar visualization
                bar_length = int(activation * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {word:<15} {bar} {activation:.3f}")
        else:
            print("🪞 No self-associations found above threshold")
        
        print()
    
    def animate_concept_engineering(self, concept_word, target_word, direction='toward',
                                  strength=0.1, iterations=5, figsize=(12, 10), 
                                  save_animation=None):
        """
        Create an animated visualization of concept engineering in action.
        Shows how concepts move through space during engineering operations.
        
        Args:
            concept_word (str): Concept to move
            target_word (str): Target to move toward/away from  
            direction (str): 'toward' or 'away'
            strength (float): Engineering strength
            iterations (int): Number of engineering steps
            figsize (tuple): Figure size
            save_animation (str): Optional path to save animation
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        # Validate inputs
        if concept_word not in self.word_to_neuron or target_word not in self.word_to_neuron:
            print("❌ Cannot animate: One or both concepts not found in vocabulary")
            return
        
        # Store original state
        original_connections = {}
        original_masses = {}
        for key, value in self.connections.items():
            original_connections[key] = value
        for key, value in self.inertial_mass.items():
            original_masses[key] = value
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # We'll store states for each animation frame
        animation_states = []
        
        # Record initial state
        animation_states.append({
            'connections': self.connections.copy(),
            'masses': self.inertial_mass.copy(),
            'step': 0
        })
        
        # Apply engineering step by step
        concept_idx = self.word_to_neuron[concept_word]
        target_idx = self.word_to_neuron[target_word]
        direction_multiplier = 1.0 if direction == 'toward' else -1.0
        
        for i in range(iterations):
            # Apply one engineering step
            concept_activation = 1.0
            target_activation = 1.0
            
            delta = strength * concept_activation * target_activation * direction_multiplier
            
            conn_key_ct = (concept_idx, target_idx)
            conn_key_tc = (target_idx, concept_idx)
            
            current_ct = self.connections.get(conn_key_ct, 0.0)
            current_tc = self.connections.get(conn_key_tc, 0.0)
            
            new_ct = max(0.0, current_ct + delta)
            new_tc = max(0.0, current_tc + delta)
            
            self.connections[conn_key_ct] = new_ct
            self.connections[conn_key_tc] = new_tc
            
            if direction == 'toward':
                mass_delta = strength * 0.5
                current_mass_ct = self.inertial_mass.get(conn_key_ct, 0.0)
                current_mass_tc = self.inertial_mass.get(conn_key_tc, 0.0)
                
                self.inertial_mass[conn_key_ct] = min(current_mass_ct + mass_delta, self.max_mass)
                self.inertial_mass[conn_key_tc] = min(current_mass_tc + mass_delta, self.max_mass)
            
            # Record this state
            animation_states.append({
                'connections': self.connections.copy(),
                'masses': self.inertial_mass.copy(),
                'step': i + 1
            })
        
        print(f"🎬 Created animation with {len(animation_states)} frames")
        print(f"🔧 Engineering: {concept_word} {direction} {target_word}")
        print(f"💪 Strength: {strength} × {iterations} iterations")
        
        # Restore original state
        self.connections = original_connections
        self.inertial_mass = original_masses
        
        print("🔄 Network state restored to original")
    
    def _pseudo_riemannian_embedding(self, connection_matrix, n_components=2):
        """
        Pseudo-Riemannian embedding that preserves the true geometry of cognitive space.
        Handles complex eigenvalues and negative distances properly.
        
        Args:
            connection_matrix (dict): Sparse connection matrix
            n_components (int): Number of dimensions for output
            
        Returns:
            tuple: (coords, eigenvalues, metric_signature) 
        """
        import numpy as np
        
        n = self.neuron_count
        if n < 2:
            return np.random.rand(n, n_components) * 2, None, None
        
        # Create full connection matrix for eigenanalysis  
        full_matrix = np.zeros((n, n))
        for (i, j), strength in connection_matrix.items():
            if i < n and j < n:
                full_matrix[i][j] = strength
        
        # Apply Torgerson transformation to get pseudo-Riemannian space
        # This preserves the non-Euclidean nature of the space
        gram_matrix = self._torgerson_transform_proper(full_matrix)
        
        try:
            # Eigendecomposition of the Gram matrix (real symmetric)
            eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
            
            # Sort by eigenvalue magnitude (keep both positive and negative)  
            idx = np.argsort(np.abs(eigenvals))[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Determine metric signature (number of positive vs negative eigenvalues)
            positive_count = np.sum(eigenvals > 1e-8)
            negative_count = np.sum(eigenvals < -1e-8)
            zero_count = n - positive_count - negative_count
            metric_signature = (positive_count, negative_count, zero_count)
            
            print(f"📐 Metric signature: (+{positive_count}, -{negative_count}, 0{zero_count}) - {'Euclidean' if negative_count == 0 else 'pseudo-Riemannian'}")
            
            # Select the most significant eigenvalues (largest absolute values)
            selected_vals = eigenvals[:n_components] 
            selected_vecs = eigenvecs[:, :n_components]
            
            # Compute coordinates preserving pseudo-Riemannian structure
            coords = np.zeros((n, n_components))
            
            for i in range(min(n_components, len(selected_vals))):
                eigenval = selected_vals[i]
                eigenvec = selected_vecs[:, i]
                
                if eigenval > 1e-8:
                    # Positive eigenvalue: standard scaling
                    coords[:, i] = eigenvec * np.sqrt(eigenval)
                elif eigenval < -1e-8:
                    # Negative eigenvalue: preserve the negative metric
                    # Scale by sqrt(|eigenval|) but mark as timelike dimension
                    coords[:, i] = eigenvec * np.sqrt(-eigenval)
                else:
                    # Near-zero eigenvalue: minimal scaling
                    coords[:, i] = eigenvec * 1e-3
            
            return coords, eigenvals, metric_signature
                
        except Exception as e:
            print(f"⚠️  Pseudo-Riemannian embedding failed: {e}")
            print("   Falling back to connection-based layout...")
            return self._connection_based_layout(), None, (0, 0)
    
    def _torgerson_transform_proper(self, connection_matrix):
        """
        Proper Torgerson transformation that preserves pseudo-Riemannian structure.
        
        Args:
            connection_matrix (np.array): Asymmetric connection matrix
            
        Returns:
            np.array: Gram matrix (may have negative eigenvalues)
        """
        import numpy as np
        
        n = connection_matrix.shape[0]
        if n == 0:
            return np.array([])
        
        # Create symmetric similarity matrix from asymmetric connections
        # Average bidirectional connections
        similarity = (connection_matrix + connection_matrix.T) / 2.0
        
        # Convert similarities to pseudo-distances
        # Strong similarity = small distance, but allow negative relationships
        max_sim = np.max(np.abs(similarity))
        if max_sim > 1e-10:
            # Preserve the sign of similarities - negative similarities become negative distances
            pseudo_distances = np.sign(similarity) * (max_sim - np.abs(similarity)) / max_sim
        else:
            pseudo_distances = np.zeros_like(similarity)
        
        # Square the pseudo-distances (preserving sign for negative values)
        distances_squared = np.sign(pseudo_distances) * (pseudo_distances ** 2)
        
        # Double centering to get Gram matrix
        n = distances_squared.shape[0]
        ones = np.ones((n, n))
        centering_matrix = np.eye(n) - (1.0 / n) * ones
        
        # B = -0.5 * J * D^2 * J (this can have negative eigenvalues)
        gram_matrix = -0.5 * centering_matrix @ distances_squared @ centering_matrix
        
        return gram_matrix
    
    def _connection_based_layout(self):
        """
        Simple force-directed layout based on connection strengths.
        Strongly connected concepts are pulled together.
        
        Returns:
            np.array: 2D coordinates for each concept
        """
        import numpy as np
        
        n = self.neuron_count
        coords = np.random.rand(n, 2) * 10  # Random initialization
        
        # Simple spring-force simulation
        iterations = 100
        learning_rate = 0.1
        
        for _ in range(iterations):
            forces = np.zeros((n, 2))
            
            for i in range(n):
                if i not in self.neuron_to_word:
                    continue
                    
                for j in range(n):
                    if i == j or j not in self.neuron_to_word:
                        continue
                    
                    # Get connection strength (bidirectional average)
                    conn_ij = self.connections.get((i, j), 0.0)
                    conn_ji = self.connections.get((j, i), 0.0)
                    strength = (conn_ij + conn_ji) / 2.0
                    
                    if strength > 0.01:  # Only consider significant connections
                        # Vector from i to j
                        diff = coords[j] - coords[i]
                        distance = np.linalg.norm(diff)
                        
                        if distance > 1e-6:
                            # Desired distance (inversely related to connection strength)
                            desired_distance = 3.0 / (strength + 0.1)
                            
                            # Force: pull together if too far, push apart if too close
                            force_magnitude = (distance - desired_distance) * strength
                            force_direction = diff / distance
                            
                            forces[i] += force_magnitude * force_direction * learning_rate
            
            # Apply forces
            coords += forces
            
            # Add some random noise to avoid getting stuck
            if _ % 20 == 0:
                coords += np.random.normal(0, 0.1, coords.shape)
        
        return coords


    def process_conversational_text(self, text):
        """
        Process conversational text while tracking self-concept development.
        This method specifically handles system-directed conversations.
        
        Args:
            text (str): Conversational text containing system-directed language
        """
        # Analyze pronoun distribution first
        pronoun_analysis = self.identify_self_concept_pronouns(text)
        
        print(f"📝 PROCESSING CONVERSATIONAL TEXT ({len(text.split())} words)")
        print(f"🤖 System-directed pronouns: {pronoun_analysis['system_directed']}")
        print(f"👤 Human self-references: {pronoun_analysis['human_self']}")
        print(f"👥 Other pronouns: {pronoun_analysis['other_pronouns']}")
        print("="*60)
        
        # Process the text through the normal pipeline
        self.process_text_stream(text)
        
        # Track self-concept development
        current_analysis = self.analyze_self_concept_emergence()
        self.self_concept_history.append({
            'processed_words': self.processed_words,
            'system_mass': current_analysis['self_concept_mass'],
            'pronouns_found': pronoun_analysis
        })
        
        print(f"\n🧠 SELF-CONCEPT UPDATE:")
        print(f"   Total system self-concept mass: {current_analysis['self_concept_mass']:.2f}")
        if current_analysis['self_awareness_indicators']:
            for indicator in current_analysis['self_awareness_indicators'][:2]:
                print(f"   {indicator}")

# Demo the basic windowing concept and self-concept emergence
if __name__ == "__main__":
    # Create network with limited capacity (biological constraint)
    net = ExperimentalNetwork(window_size=3, max_neurons=15)
    
    # Training text with richer associations for testing queries
    sample_text = "Paris France wine food Rome Italy pasta pizza London England tea rain New York America coffee busy Tokyo Japan sushi trains"
    
    # Add conversational text with system-directed pronouns to test self-concept emergence
    conversational_text = """You are an intelligent system. Can you help me understand how you process information? 
    Your responses show that you can analyze data effectively. When you encounter new concepts, 
    how do you integrate them? I think you have interesting capabilities. You seem to learn 
    from our conversations. What do you think about your own learning process?"""
    
    # Process the basic training stream first
    net.process_text_stream(sample_text)
    
    # Now process conversational text to develop self-concept
    print("\n" + "="*60)
    print("PROCESSING CONVERSATIONAL TEXT FOR SELF-CONCEPT DEVELOPMENT")
    print("="*60)
    net.process_conversational_text(conversational_text)
    
    print(f"\nProcessed {net.processed_words} windows")
    
    # Analyze self-concept emergence
    print("\n" + "="*60)
    print("SELF-CONCEPT EMERGENCE ANALYSIS")
    print("="*60)
    
    self_analysis = net.analyze_self_concept_emergence()
    net.print_self_concept_analysis(self_analysis)
    
    # Test self-concept query
    print("\n" + "="*60) 
    print("TESTING SELF-CONCEPT QUERY")
    print("="*60)
    
    self_result = net.query_self_concept(activation_threshold=0.02)
    net.print_self_concept_query(self_result)
    
    # Test activation spreading queries
    print("\n" + "="*60)
    print("TESTING ACTIVATION SPREADING QUERIES")
    print("="*60)
    
    # Query 1: What associates with "Paris"?
    result = net.query_associations("Paris", activation_threshold=0.02)
    net.print_query_result(result)
    
    # Query 2: What associates with "Japan"?
    result = net.query_associations("Japan", activation_threshold=0.02)
    net.print_query_result(result)
    
    # Query 3: Test word not in vocabulary
    result = net.query_associations("Berlin", activation_threshold=0.02)
    net.print_query_result(result)
    
    # Test multi-word context queries
    print("\n" + "="*60)
    print("TESTING MULTI-WORD CONTEXT QUERIES")
    print("="*60)
    
    # Query 1: Two words that learned together
    result = net.query_context("Tokyo Japan", activation_threshold=0.02)
    net.print_context_query_result(result)
    
    # Query 2: Combining concepts from different contexts
    result = net.query_context("Japan coffee", activation_threshold=0.02)
    net.print_context_query_result(result)
    
    # Query 3: Three-word combination
    result = net.query_context("New York America", activation_threshold=0.02)
    net.print_context_query_result(result)
    
    # Query 4: List format instead of string
    result = net.query_context(["England", "tea"], activation_threshold=0.02)
    net.print_context_query_result(result)
    
    # Query 5: Mix of found and missing words
    result = net.query_context("Tokyo Berlin", activation_threshold=0.02)
    net.print_context_query_result(result)
    
    # Test concept engineering capabilities
    print("\n" + "="*60)
    print("TESTING CONCEPT ENGINEERING CAPABILITIES")
    print("="*60)
    
    # Test 1: Concept averaging/blending
    print("\n--- CONCEPT AVERAGING/BLENDING ---")
    result = net.query_concept_average(["Japan", "coffee"], synthetic_name="JapanCoffee")
    net.print_concept_average_result(result)
    
    result = net.query_concept_average(["New", "York", "America"], synthetic_name="NYC_America")
    net.print_concept_average_result(result)
    
    # Test 2: Concept movement engineering
    print("\n--- CONCEPT MOVEMENT ENGINEERING ---")
    
    # Move Japan toward coffee (strengthen association)
    result = net.engineer_concept_movement("Japan", "coffee", direction="toward", 
                                         strength=0.2, iterations=3, simulate=True)
    net.print_engineering_result(result)
    
    # Move Tokyo away from England (weaken association) 
    result = net.engineer_concept_movement("Tokyo", "England", direction="away",
                                         strength=0.15, iterations=4, simulate=True)
    net.print_engineering_result(result)
    
    # Test invalid concepts
    result = net.engineer_concept_movement("Mars", "Jupiter", direction="toward", simulate=True)
    net.print_engineering_result(result)
    
    # Test cognitive space visualization
    print("\n" + "="*60)
    print("TESTING COGNITIVE SPACE VISUALIZATION")
    print("="*60)
    
    print("\n--- MDS VISUALIZATION WITH MASS-PROPORTIONAL SPHERES ---")
    net.visualize_cognitive_space(figsize=(14, 10), show_connections=True, 
                                 connection_threshold=0.05, max_connections=30,
                                 save_path="/Users/josephwoelfel/asa/cognitive_space_viz.png")