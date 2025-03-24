import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values and convert to float32 for consistency
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Flatten the 28x28 images into 1D vectors of size 784
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# One-hot encode the labels
y_train = tf.one_hot(y_train, depth=10, dtype=tf.float32)
y_test = tf.one_hot(y_test, depth=10, dtype=tf.float32)

# Network configuration
input_dim = 784
hidden_dim1 = 128
hidden_dim2 = 64
output_dim = 10
learning_rate = 0.01
epochs = 10
batch_size = 100

# Function to initialize weights with random values
def initialize_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1, dtype=tf.float32))

# Initialize weights and biases
W1 = initialize_weights([input_dim, hidden_dim1])
b1 = tf.Variable(tf.zeros([hidden_dim1], dtype=tf.float32))

W2 = initialize_weights([hidden_dim1, hidden_dim2])
b2 = tf.Variable(tf.zeros([hidden_dim2], dtype=tf.float32))

W3 = initialize_weights([hidden_dim2, output_dim])
b3 = tf.Variable(tf.zeros([output_dim], dtype=tf.float32))

# Define the MLP model
class MLP(tf.Module):
    def __init__(self):
        super().__init__()
        # Assign initialized weights and biases to the model
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3

    def __call__(self, X):
        # Forward pass through the network
        X = tf.cast(X, tf.float32)  
        hidden1 = tf.nn.relu(tf.matmul(X, self.W1) + self.b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self.W2) + self.b2)
        output = tf.matmul(hidden2, self.W3) + self.b3
        return output

# Create an instance of the model
model = MLP()

# Loss function and optimizer
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# Training loop
for epoch in range(epochs):
    num_batches = len(x_train) // batch_size
    epoch_loss = 0

    for batch in range(num_batches):
        batch_x = x_train[batch * batch_size:(batch + 1) * batch_size]
        batch_y = y_train[batch * batch_size:(batch + 1) * batch_size]

        with tf.GradientTape() as tape:
            predictions = model(batch_x)
            loss = loss_function(batch_y, predictions)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3])
        optimizer.apply_gradients(zip(gradients, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]))

        epoch_loss += loss.numpy() / num_batches

    # Calculate training accuracy
    train_output = model(x_train)
    train_pred = tf.argmax(train_output, axis=1)
    train_actual = tf.argmax(y_train, axis=1)
    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(train_pred, train_actual), tf.float32))

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy.numpy():.4f}")

# Evaluate the model on the test dataset
test_output = model(x_test)
test_pred = tf.argmax(test_output, axis=1)
test_actual = tf.argmax(y_test, axis=1)
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(test_pred, test_actual), tf.float32))

print(f"Test Accuracy: {test_accuracy.numpy():.4f}")

# Function to test the model on a single image
def predict_single_image(index):
    img = x_test[index].reshape(1, 784)
    actual_label = np.argmax(y_test[index])

    logits = model(img)
    probabilities = tf.nn.softmax(logits)
    predicted_label = np.argmax(probabilities.numpy())

    # Display the image
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

    print(f"Probabilities: {probabilities.numpy()}")

# Test with a random image
predict_single_image(np.random.randint(0, len(x_test)))
