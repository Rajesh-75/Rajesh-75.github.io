import matplotlib.pyplot as plt

def draw_neural_net(layers):
    fig, ax = plt.subplots(figsize=(12, 8))
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    
    layer_sizes = layers
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes (Neurons)
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for j in range(layer_size):
            x = i * h_spacing + left
            y = layer_top - j * v_spacing
            
            # --- LABELING THE LAST NODE (LOSS) ---
            if i == len(layer_sizes) - 1:
                # We place the text slightly above the node (y + offset)
                ax.text(x, y + v_spacing/2, "Loss", color='red', 
                        ha='center', va='bottom', fontweight='bold', fontsize=14)
                # Label the incoming error signal
                ax.text(x - h_spacing/4, y + v_spacing/4, r"$\frac{\partial L}{\partial A_{out}}$", 
                        color='red', fontsize=12)

            # Highlight specific nodes for the math (Layer 1 and 2)
            is_ai_minus_1 = (i == 1 and j == 2)
            is_zi = (i == 2 and j == 2)
            
            node_color = 'orange' if (is_ai_minus_1 or is_zi) else 'skyblue'
            circle = plt.Circle((x, y), v_spacing / 4., color=node_color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            if is_ai_minus_1:
                ax.text(x, y + v_spacing/3, r'$A_{i-1}$', fontsize=14, ha='center', fontweight='bold')
            if is_zi:
                ax.text(x, y + v_spacing/3, r'$Z_{i}$', fontsize=14, ha='center', fontweight='bold')
                ax.text(x, y - v_spacing/2, r'$dZ_i$', color='red', fontsize=12, ha='center')

            # Connections (Edges)
            if i > 0:
                prev_layer_size = layer_sizes[i-1]
                prev_layer_top = v_spacing * (prev_layer_size - 1) / 2. + (top + bottom) / 2.
                for k in range(prev_layer_size):
                    line_x = [(i - 1) * h_spacing + left, i * h_spacing + left]
                    line_y = [prev_layer_top - k * v_spacing, layer_top - j * v_spacing]
                    
                    is_wi = (i == 2 and j == 2 and k == 2)
                    color = 'red' if is_wi else 'gray'
                    alpha = 0.8 if is_wi else 0.3
                    lw = 2 if is_wi else 1
                    
                    line = plt.Line2D(line_x, line_y, c=color, alpha=alpha, lw=lw, zorder=3)
                    ax.add_artist(line)
                    
                    if is_wi:
                        mid_x, mid_y = sum(line_x)/2, sum(line_y)/2
                        ax.text(mid_x, mid_y + 0.02, r'$W_i$', fontsize=12, color='darkred', ha='center')
                        ax.text(mid_x, mid_y - 0.06, r'$\frac{\partial L}{\partial W_i}$', 
                                fontsize=14, color='red', ha='center')

    # Formula & Backprop Arrow
    ax.annotate('', xy=(0.2, 0.05), xytext=(0.8, 0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.5, 0.02, "Backpropagation: Gradients flow from Loss back to Weights", 
            color='red', ha='center', fontweight='bold')
    # Formula Box
    formula = r"$\frac{\partial L}{\partial W_i} = dZ_i \cdot (A_{i-1})^T$"
    plt.text(0.5, 0.95, f"Weight Gradient Calculation:\n{formula}", 
             fontsize=15, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    plt.axis('off')
    plt.title("Backpropagation Calculus Visualization", fontsize=16)
    plt.show()

draw_neural_net([5, 5, 5, 1])
