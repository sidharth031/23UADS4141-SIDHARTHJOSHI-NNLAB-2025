## Objective

To implement a **multi-layer perceptron (MLP)** with one hidden layer using **NumPy** in Python and demonstrate its ability to learn the **XOR** Boolean function.

---

## Description of the Model

An **MLP** is a feedforward neural network with at least one hidden layer, capable of learning **non-linearly separable** functions like XOR. Key components include:

- **Hidden Layer:** Introduces non-linearity, enabling XOR learning.
- **Step Activation Function:** Outputs 0 or 1 based on thresholding.
- **Weight & Bias Updates:** Adjusted using a simple learning rule.
- **Training Process:** Sequential training of hidden and output layers.

---

## Description of Code

1. **Step Function:**

   - Outputs 1 if input ≥ 0, else 0.

2. **Perceptron Class:**

   - **Initialization:** Random weights and bias.
   - **Forward Pass:** Computes output using weighted sum + step function.
   - **Training:** Updates weights/bias using error correction.
   - **Prediction:** Generates output for new data.

3. **MLP Architecture for XOR:**
   - **Hidden Layer:** Two perceptrons trained with intermediate targets.
   - **Output Layer:** Single perceptron trained on hidden layer outputs.

---

## Performance Evaluation

- **XOR Learning:** Successfully predicts XOR outputs:
  - **Input:** `[[0,0],[0,1],[1,0],[1,1]]`
  - **Output:** `[0, 1, 1, 0]`
- **Accuracy:** Perfect on XOR dataset, demonstrating MLP's ability to solve non-linear problems.

---

## Limitations

- **Step Function:** Prevents gradient-based optimization.
- **Manual Architecture:** Fixed hidden layer structure.

---

## Scope for Improvement

- Use **sigmoid** or **ReLU** for gradient-friendly learning.
- Implement **backpropagation** for efficient weight updates.
