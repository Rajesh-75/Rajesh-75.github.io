import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Suppress TF warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Math Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Note: When using BCE + Sigmoid, the output delta simplifies to (A - Y).
    # We keep this function for the hidden layers.
    return x * (1 - x)

def compute_bce_loss(y_true, y_pred):
    epsilon = 1e-15 # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- 2. n-Layer Training with Adam ---
def train_n_layer_adam(X, y, layer_sizes, learning_rate=0.01, epochs=2000):
    num_layers = len(layer_sizes)
    weights = []
    
    # Adam Parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = [] # First moment vector
    v = [] # Second moment vector
    t = 0  # Time step

    # Initialize Weights and Adam state
    for i in range(num_layers - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
        weights.append(w)
        m.append(np.zeros_like(w))
        v.append(np.zeros_like(w))

    loss_history = []

    for epoch in range(epochs):
        t += 1
        # --- Forward Prop ---
        activations = [X]
        for i in range(num_layers - 1):
            activations.append(sigmoid(np.dot(activations[-1], weights[i])))

        # --- Backward Prop (BCE Gradient) ---
        # The derivative of BCE loss w.r.t the input of the sigmoid is simply (A - Y)
        output_delta = (activations[-1] - y) 
        deltas = [output_delta]

        # Backpropagate through hidden layers
        for i in range(num_layers - 2, 0, -1):
            hidden_error = deltas[0].dot(weights[i].T)
            hidden_delta = hidden_error * sigmoid_derivative(activations[i])
            deltas.insert(0, hidden_delta)

        # --- Adam Weight Update ---
        for i in range(num_layers - 1):
            grad = activations[i].T.dot(deltas[i]) / X.shape[0]
            
            # Update moving averages
            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * (grad**2)
            
            # Bias correction
            m_hat = m[i] / (1 - beta1**t)
            v_hat = v[i] / (1 - beta2**t)
            
            # Update weights
            weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Log Loss
        current_loss = compute_bce_loss(y, activations[-1])
        loss_history.append(current_loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: BCE Loss = {current_loss:.6f}")

    return weights, loss_history

# --- 3. TensorFlow Comparison (One Step) ---
def compare_step_tf(X, y, layer_sizes, np_weights):
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

    model = tf.keras.Sequential()
    
    # Loop through layers
    for i in range(1, len(layer_sizes)):
        # Key Fix: Add input_shape to the VERY FIRST layer only
        if i == 1:
            model.add(tf.keras.layers.Dense(
                layer_sizes[i], 
                use_bias=False, 
                activation='sigmoid',
                input_shape=(layer_sizes[i-1],) # This "builds" the weights immediately
            ))
        else:
            model.add(tf.keras.layers.Dense(
                layer_sizes[i], 
                use_bias=False, 
                activation='sigmoid'
            ))
    
    # Now that input_shape is defined, set_weights will work perfectly
    for i, w in enumerate(np_weights):
        model.layers[i].set_weights([w.astype(np.float32)])

    with tf.GradientTape() as tape:
        preds = model(X_tf)
        # Binary Crossentropy comparison
        bce = tf.keras.losses.BinaryCrossentropy()
        loss = bce(y_tf, preds)

    tf_grads = tape.gradient(loss, model.trainable_variables)
    return preds.numpy(), [g.numpy() for g in tf_grads]

# --- 4. Execution ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
layer_sizes = [2, 8, 4, 1] # More neurons for faster XOR learning

print("Training NumPy Network with Adam Optimizer...")
trained_weights, losses = train_n_layer_adam(X, y, layer_sizes, learning_rate=0.05, epochs=1500)

print("\nFinal Predictions:")
final_act = X
for w in trained_weights:
    final_act = sigmoid(np.dot(final_act, w))
print(np.round(final_act, 3))

# Compare one step with TF
tf_pred, tf_grads = compare_step_tf(X, y, layer_sizes, trained_weights)
print(f"\nMath Verification: Mean Diff in Prediction = {np.abs(final_act - tf_pred).mean():.10f}")

# --- 5. Visualization ---
plt.figure(figsize=(10, 5))
plt.plot(losses, color='blue', lw=2)
plt.title("Training Loss (Binary Cross-Entropy)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
