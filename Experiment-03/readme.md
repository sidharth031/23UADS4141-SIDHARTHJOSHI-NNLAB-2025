## ✅ **Objective**

To build a **three-layer neural network** using **TensorFlow** (**without Keras**) for classifying **MNIST handwritten digits**, applying **feed-forward** and **backpropagation** techniques.

---

## ✅ **Model Description**

The neural network architecture consists of:

- **Input Layer:** 784 neurons for the flattened 28x28 pixel images.  
- **Hidden Layer 1:** 128 neurons activated by **ReLU**, capturing complex patterns.  
- **Hidden Layer 2:** 64 neurons with **ReLU**, adding more feature extraction power.  
- **Output Layer:** 10 neurons representing digit classes (0–9).  
- **Feed-Forward:** Passes the input data through the layers to generate predictions.  
- **Backpropagation:** Uses **Stochastic Gradient Descent (SGD)** to adjust weights by minimizing the loss.

---

## ✅ **Code Breakdown**

### 1. **Data Preparation**

- **Load Dataset:** Import the MNIST dataset.  
- **Preprocessing:**  
  - Normalize the pixel values between `[0, 1]` for efficient learning.  
  - Flatten the images into 1D vectors (28x28 → 784).  
  - Apply **one-hot encoding** to the labels for multi-class classification.  

---

### 2. **Network Parameters**

- **Architecture:**  
  - Input → 128 hidden neurons → 64 hidden neurons → 10 output neurons.  
- **Hyperparameters:**  
  - Learning rate = `0.01`  
  - Epochs = `10`  
  - Batch size = `100`  
- **Weight Initialization:**  
  - Random initialization for weights.  
  - Biases initialized to zeros.  

---

### 3. **Neural Network Class (`NeuralNetwork`)**

- **Feed-Forward:**  
  - First Layer: Applies `ReLU(W1 * X + b1)`  
  - Second Layer: `ReLU(W2 * L1 + b2)`  
  - Output Layer: `W3 * L2 + b3` (logits)  
- **Backpropagation:**  
  - Uses `tf.GradientTape()` to compute gradients.  
  - Updates weights and biases with **SGD**.  

---

### 4. **Training Loop**

- **For each epoch:**  
  - Iterates over mini-batches, calculates loss, and updates weights.  
  - Displays the loss and accuracy after every epoch.  

---

### 5. **Model Evaluation**

- **Test Accuracy:**  
  - Uses the trained model to predict on test data.  
  - Calculates the accuracy by comparing predictions with actual labels.  

---

### 6. **Single Image Prediction**

- **Random Test Image:**  
  - Selects a random test image.  
  - Makes predictions and visualizes the image.  
- **Probability Distribution:**  
  - Displays predicted probabilities for all digit classes.

---

## ✅ **Performance Insights**

- **Training Accuracy:** Increases with each epoch, indicating effective learning.  
- **Test Accuracy:** Achieves reliable performance on unseen data.  
- **Visualization:** Displays correct predictions and confidence scores.  

---

## ✅ **Limitations**

- **Static Hyperparameters:** Fixed learning rate and network size.  
- **No Regularization:** No dropout or L2 regularization applied.  
- **Simplified Backpropagation:** Uses TensorFlow's auto-differentiation instead of manual backpropagation.

---

## ✅ **Future Improvements**

- **Add Dropout Layers:** To reduce overfitting.  
- **Dynamic Learning Rate:** Gradually reduce the learning rate during training.  
- **Regularization:** Include L2 regularization for better generalization.  
- **Batch Normalization:** Improve convergence speed and stability.

Output
Epoch 1/10, Loss: 1.2349, Training Accuracy: 0.8257
Epoch 2/10, Loss: 0.5088, Training Accuracy: 0.8760
Epoch 3/10, Loss: 0.3977, Training Accuracy: 0.8940
Epoch 4/10, Loss: 0.3505, Training Accuracy: 0.9045
Epoch 5/10, Loss: 0.3211, Training Accuracy: 0.9113
Epoch 6/10, Loss: 0.2993, Training Accuracy: 0.9168
Epoch 7/10, Loss: 0.2817, Training Accuracy: 0.9215
Epoch 8/10, Loss: 0.2668, Training Accuracy: 0.9253
Epoch 9/10, Loss: 0.2537, Training Accuracy: 0.9286
Epoch 10/10, Loss: 0.2421, Training Accuracy: 0.9318
Test Accuracy: 0.9325

![image](https://github.com/user-attachments/assets/5ef969b4-5b2c-4479-8a3b-c0540c97cc73)

My Comments:

The model is doing a pretty good job, reaching over 93% accuracy on test data.

It has three layers and uses ReLU, which helps it learn better.

The loss keeps going down, meaning the model is getting better at recognizing numbers.
