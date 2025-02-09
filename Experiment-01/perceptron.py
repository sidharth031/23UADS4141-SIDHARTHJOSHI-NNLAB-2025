# Objective - To implement the Perceptron Learning Algorithm using numpy in Python. Evaluate performance of a single perceptron for NAND and XOR truth tables as input dataset.

import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)
                y_pred = self.activation(np.dot(self.weights, x_i))
                self.weights += self.lr * (y[i] - y_pred) * x_i


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_nand = np.array([1, 1, 1, 0])  
Y_xor = np.array([0, 1, 1, 0])   


perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X, Y_nand)

print("NAND Perceptron Output:")
for x in X:
    print(f"Input: {x}, Output: {perceptron_nand.predict(x)}")


perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X, Y_xor)

print("XOR Perceptron Output:")
for x in X:
    print(f"Input: {x}, Output: {perceptron_xor.predict(x)}")  

"""
Explanation of Code -
Class Perceptron

- Initializes weights (including bias) to zeros.
- Uses step function activation to classify inputs.
- Updates weights using the Perceptron Learning Rule:
  w = w + learning_rate × error × x

Training the Perceptron

- It loops over the dataset multiple times (epochs).
- If the perceptron makes an incorrect prediction, it updates the weights.

Testing on NAND and XOR

- For NAND: The perceptron successfully learns and correctly classifies the input.
- For XOR: The perceptron fails because XOR is not linearly separable.


"""

