import random # Only for initialization

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999, epsilon=1e-8):
        self.eta = learning_rate
        self.b1, self.b2 = b1, b2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        if self.m is None:
            self.m = [[0.0 for _ in row] for row in weights]
            self.v = [[0.0 for _ in row] for row in weights]
        
        self.t += 1
        new_w = [[0.0 for _ in row] for row in weights]
        
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                g = gradients[i][j]
                # 1. Moment Updates
                self.m[i][j] = self.b1 * self.m[i][j] + (1 - self.b1) * g
                self.v[i][j] = self.b2 * self.v[i][j] + (1 - self.b2) * (g**2)
                
                # 2. Bias Correction
                m_hat = self.m[i][j] / (1 - self.b1**self.t)
                v_hat = self.v[i][j] / (1 - self.b2**self.t)
                
                # 3. Parameter Update
                new_w[i][j] = weights[i][j] - (self.eta * m_hat) / (v_hat**0.5 + self.epsilon)
        return new_w

class ConvLayer:
    def __init__(self, kernel_size):
        # Small random initialization
        self.kernel = [[random.uniform(-0.1, 0.1) for _ in range(kernel_size)] 
                       for _ in range(kernel_size)]
        self.bias = 0.0
        self.optimizer = AdamOptimizer()

    def forward(self, image):
        self.last_input = image
        h, w = len(image), len(image[0])
        kh, kw = len(self.kernel), len(self.kernel[0])
        output = [[0.0 for _ in range(w-kw+1)] for _ in range(h-kh+1)]
        
        # Standard 2D Convolution (Sliding Window)
        for i in range(h-kh+1):
            for j in range(w-kw+1):
                sum_val = 0
                for ki in range(kh):
                    for kj in range(kw):
                        sum_val += image[i+ki][j+kj] * self.kernel[ki][kj]
                output[i][j] = sum_val + self.bias
        return output

    def backward(self, d_out):
        kh, kw = len(self.kernel), len(self.kernel[0])
        ih, iw = len(self.last_input), len(self.last_input[0])
        d_kernel = [[0.0 for _ in range(kw)] for _ in range(kh)]
        d_bias = 0.0

        # Step 2: dL/dW (Valid Correlation between Input and Error Map)
        for i in range(kh):
            for j in range(kw):
                for oi in range(len(d_out)):
                    for oj in range(len(d_out[0])):
                        d_kernel[i][j] += self.last_input[i+oi][j+oj] * d_out[oi][oj]

        # dL/db is just the sum of the error map
        for row in d_out:
            d_bias += sum(row)

        # Step 3: dA_prev (Full Correlation of d_out with the 180-rotated kernel)
        # This is what lets us chain backward() calls across stacked ConvLayers.
        rotated = [[self.kernel[kh-1-ki][kw-1-kj] for kj in range(kw)] for ki in range(kh)]
        padded = [[0.0] * (len(d_out[0]) + 2*(kh-1)) for _ in range(len(d_out) + 2*(kh-1))]
        for i in range(len(d_out)):
            for j in range(len(d_out[0])):
                padded[i+kh-1][j+kw-1] = d_out[i][j]

        d_input = [[0.0 for _ in range(iw)] for _ in range(ih)]
        for i in range(ih):
            for j in range(iw):
                sum_val = 0
                for ki in range(kh):
                    for kj in range(kw):
                        sum_val += padded[i+ki][j+kj] * rotated[ki][kj]
                d_input[i][j] = sum_val

        # Update kernel weights using Adam (bias updated with plain SGD here for simplicity)
        self.kernel = self.optimizer.update(self.kernel, d_kernel)
        self.bias -= self.optimizer.eta * d_bias
        return d_input