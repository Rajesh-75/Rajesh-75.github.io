# Constants for mathematical operations
E = 2.718281828459

def exp(x):
    """Simple exponential function: e^x"""
    return E ** x

# --- ACTIVATION FUNCTIONS ---

def tanh(vector):
    """Hyperbolic Tangent: maps values to (-1, 1)"""
    # Formula: (e^z - e^-z) / (e^z + e^-z)
    activated = []
    for z in vector:
        e_p = exp(z)
        e_n = exp(-z)
        activated.append((e_p - e_n) / (e_p + e_n))
    return activated

def softmax(vector):
    """Softmax: maps values to a probability distribution (sum to 1)"""
    # Formula: e^zi / sum(e^zj)
    # Note: In practice, we subtract max(vector) for numerical stability
    exps = [exp(z) for z in vector]
    total = sum(exps)
    return [e / total for e in exps]

# --- LINEAR TRANSFORMATION ---

def dot_product(row, input_vector):
    """Standard dot product: sum(w_i * x_i)"""
    return sum(w * x for w, x in zip(row, input_vector))

def compute_layer(input_vector, weight_matrix, bias_vector, activation_fn):
    """Calculates Z = WX + b and then applies activation"""
    z_vector = []
    for i in range(len(weight_matrix)):
        # Calculate pre-activation (Z) for each neuron in the layer
        z_i = dot_product(weight_matrix[i], input_vector) + bias_vector[i]
        z_vector.append(z_i)
    
    # Apply the activation function (Tanh or Softmax)
    return activation_fn(z_vector)

# --- THE 7-LAYER PIPELINE ---

def forward_prop(X, all_weights, all_biases):
    """
    X: Input list (vector)
    all_weights: List of 7 weight matrices
    all_biases: List of 7 bias vectors
    """
    current_a = X # Activation starts as the input
    
    # Process Layers 1 through 6 (Hidden Layers with Tanh)
    for i in range(6):
        current_a = compute_layer(current_a, all_weights[i], all_biases[i], tanh)
    
    # Process Layer 7 (Output Layer with Softmax)
    output_probabilities = compute_layer(current_a, all_weights[6], all_biases[6], softmax)
    
    return output_probabilities

# --- COST FUNCTION ---

def cross_entropy_cost(y_pred, y_true):
    """Cross-Entropy Loss: -sum(y_true * log(y_pred))"""
    import math # Only for the log function
    loss = 0
    for i in range(len(y_true)):
        # We add a tiny epsilon to avoid log(0)
        loss += y_true[i] * math.log(y_pred[i] + 1e-15)
    return -loss
    

# --- DERIVATIVES OF ACTIVATIONS ---

def tanh_derivative(a):
    """Derivative of tanh(z) where a = tanh(z)"""
    # Formula: 1 - tanh(z)^2
    return [1.0 - (val ** 2) for val in a]

# Note: The derivative of (Softmax + Cross-Entropy) simplifies 
# beautifully to: (Predictions - Labels)

# --- THE BACKWARD PASS ---

def backward_prop(y_true, y_pred, cache, weights):
    """
    y_true: Target labels (one-hot encoded)
    y_pred: Output from forward_prop (Softmax results)
    cache: Dictionary containing 'a' (activations) and 'z' (pre-activations) for each layer
    weights: Current weights of the 7 layers
    """
    gradients = {"dW": [], "db": []}
    
    # 1. Output Layer Error (Layer 7: Softmax + Cross-Entropy)
    # delta = dL/dz. For Softmax/CE, it's just (predictions - truth)
    delta = [p - t for p, t in zip(y_pred, y_true)]
    
    # 2. Backpropagate through 7 layers (7 to 1)
    # We iterate backwards from index 6 down to 0
    for l in range(6, -1, -1):
        # A. Calculate Gradient for Weights (dW = delta * a_prev^T)
        # a_prev is the activation from the layer BEFORE this one
        # If l=0, a_prev is the original input X
        a_prev = cache['a'][l-1] if l > 0 else cache['input']
        
        layer_dW = []
        for d in delta:
            # Each neuron's delta multiplied by each input from the prev layer
            row_grads = [d * ap for ap in a_prev]
            layer_dW.append(row_grads)
            
        # B. Calculate Gradient for Biases (db = delta)
        layer_db = delta
        
        # Save these for the weight update later
        gradients["dW"].insert(0, layer_dW)
        gradients["db"].insert(0, layer_db)
        
        # C. Calculate Delta for the PREVIOUS layer (if not at the start)
        if l > 0:
            # delta_prev = (W_curr^T * delta_curr) * activation_derivative(z_prev)
            new_delta = []
            current_W = weights[l]
            
            # W^T * delta
            for j in range(len(current_W[0])): # For each neuron in the PREVIOUS layer
                error_sum = 0
                for i in range(len(delta)): # For each neuron in the CURRENT layer
                    error_sum += current_W[i][j] * delta[i]
                
                # Multiply by Tanh derivative of the previous layer's activation
                # Note: cache['a'][l-1] contains the tanh activations
                deriv = 1.0 - (cache['a'][l-1][j] ** 2)
                new_delta.append(error_sum * deriv)
            
            delta = new_delta # Update delta to move backward one more step
            
    return gradients

# --- THE UPDATE STEP (Optimization) ---

def update_weights(weights, biases, gradients, learning_rate):
    """Adjusts weights based on gradients"""
    for l in range(7):
        # Update Weights: W = W - (lr * dW)
        for i in range(len(weights[l])):
            for j in range(len(weights[l][i])):
                weights[l][i][j] -= learning_rate * gradients["dW"][l][i][j]
        
        # Update Biases: b = b - (lr * db)
        for i in range(len(biases[l])):
            biases[l][i] -= learning_rate * gradients["db"][l][i]
            
    return weights, biases    