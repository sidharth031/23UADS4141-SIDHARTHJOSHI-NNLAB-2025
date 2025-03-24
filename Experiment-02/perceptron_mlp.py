import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, X):
        return step_function(np.dot(X, self.weights) + self.bias)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.forward(xi)
                error = target - output
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

    def predict(self, X):
        return self.forward(X)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_hidden1 = np.array([1, 1, 1, 0])
y_hidden2 = np.array([0, 1, 1, 1])
y_output = np.array([0, 1, 1, 0])

# Train first hidden neuron
hidden1 = Perceptron(input_size=2)
hidden1.train(X, y_hidden1)

# Train second hidden neuron
hidden2 = Perceptron(input_size=2)
hidden2.train(X, y_hidden2)

# Combine hidden layer outputs
hidden_output = np.column_stack((hidden1.predict(X), hidden2.predict(X)))

# Train output neuron
output_neuron = Perceptron(input_size=2)
output_neuron.train(hidden_output, y_output)

# Predictions
predictions = output_neuron.predict(hidden_output)
print("Predictions for XOR:", predictions.flatten())
