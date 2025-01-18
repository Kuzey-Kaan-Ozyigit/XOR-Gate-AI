import numpy as np

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dataset (X)
X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
            ])

# True outputs (y)
y = np.array([[0], [1], [1], [0]])

# Weights and biases
np.random.seed(1)
input_layer_size = 2  # Input layer size (2 inputs)
hidden_layer_size = 4  # First hidden layer size
hidden_layer_size_2 = 4  # Second hidden layer size
output_layer_size = 1  # Output layer size (1 output)

# Weights for the hidden layer
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size)
bias_hidden = np.zeros((1, hidden_layer_size))

# Weights for the second hidden layer
weights_hidden_hidden = np.random.randn(hidden_layer_size, hidden_layer_size_2)
bias_hidden_2 = np.zeros((1, hidden_layer_size_2))

# Weights for the output layer
weights_hidden_output = np.random.randn(hidden_layer_size_2, output_layer_size)
bias_output = np.zeros((1, output_layer_size))

_output = np.zeros(y.shape)

# Training process (10000 iterations)
learning_rate = 0.1
for epoch in range(10000):
    # Forward computation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)  # Output of the first hidden layer

    hidden_layer_input_2 = np.dot(hidden_layer_output, weights_hidden_hidden) + bias_hidden_2
    hidden_layer_output_2 = sigmoid(hidden_layer_input_2)  # Output of the second hidden layer

    final_input = np.dot(hidden_layer_output_2, weights_hidden_output) + bias_output
    output = sigmoid(final_input)

    for i in range(len(output)):
        if output[i] >= 0.5:
            _output[i] = 1
        else:
            _output[i] = 0

    # Error computation
    error = y - output  # Difference between the true value and the prediction

    # Backpropagation (updating weights)
    d_output = error * sigmoid_derivative(output)  # Derivative of the error at the output
    d_hidden_layer_2 = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output_2)  # Derivative for the second hidden layer
    d_hidden_layer_1 = d_hidden_layer_2.dot(weights_hidden_hidden.T) * sigmoid_derivative(hidden_layer_output)  # Derivative for the first hidden layer

    # Update weights
    weights_hidden_output += hidden_layer_output_2.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_hidden_hidden += hidden_layer_output.T.dot(d_hidden_layer_2) * learning_rate
    bias_hidden_2 += np.sum(d_hidden_layer_2, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden_layer_1) * learning_rate
    bias_hidden += np.sum(d_hidden_layer_1, axis=0, keepdims=True) * learning_rate

    # Print error every 1000 iterations
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Outputs after training is completed
print("\nOutputs After Training:")
print(output)
print(_output)
