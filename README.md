# Neural network from scratch with NumPy

This repository contains a small neural network implemented from scratch (pure NumPy and Algebra) to classify handwritten digits from the MNIST-format CSV (`train.csv`). The goal is educational: implement forward/backward propagation and training using raw linear algebra and basic Python/NumPy.

---

## Contents

- `train.csv` - original MNIST-style dataset where each row is `[label, pixel1, pixel2, ..., pixel784]`
- `test.csv` - original MNIST-style dataset where each row is `[label, pixel1, pixel2, ..., pixel784]` But not implemented yet.
- `handwritten-digit-nn-pure-numpy.ipynb` — main code
- `README.md` — this file.

---

## Project overview

- **Input:** 28×28 grayscale images flattened to 784 features, scaled to `[0,1]` by dividing by 255.
- **Architecture:** 2-layer fully-connected network (one hidden layer):

  - Input layer: 784 units
  - Hidden layer: 10 units, ReLU activation
  - Output layer: 10 units, softmax activation

- **Loss:** cross-entropy loss (softmax + CE)
- **Optimizer:** vanilla SGD (single global learning rate `alpha`) on full dataset (no explicit mini-batches in original script)

> I wrote the implementation for educational purposes and report \~80% dev accuracy, which is a good result for this simple architecture.

---

## Architecture and math (brief)

Let `m` be the number of examples in a batch (columns). Notation below follows shapes used in the code.

Forward pass:

- `Z1 = W1.dot(X) + b1` → shape `(10, m)`
- `A1 = ReLU(Z1)`
- `Z2 = W2.dot(A1) + b2` → shape `(10, m)`
- `A2 = softmax(Z2)` → columnwise softmax giving probability vectors for each sample

Loss (averaged over batch):

```
L = - 1/m * sum_over_examples sum_k Y_k * log(A2_k)
```

Backpropagation (column-wise / batched):

- `dZ2 = A2 - Y_one_hot` (shape `(10, m)`) — cross-entropy + softmax simplification
- `dW2 = 1/m * dZ2.dot(A1.T)` (shape `(10, 10)`)
- `db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)` (shape `(10,1)`)
- `dZ1 = W2.T.dot(dZ2) * ReLU'(Z1)`
- `dW1 = 1/m * dZ1.dot(X.T)` (shape `(10, 784)`)
- `db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)`

Update:

- `W -= alpha * dW`
- `b -= alpha * db`

---

## How to run

1. Put `train.csv` in `./data/train.csv`.
2. Install requirements in your environment.
3. Go to the `handwritten-digit-nn-pure-numpy.ipynb` notebook.
4. Run all cells end-to-end.
5. After training with `gradient_descent` function, use the helper to check accuracy on the dev set or test individual examples.

## Extending the project

- Add a second hidden layer or increase hidden units (reduces bias).
- Try ReLU with He initialization.
- Implement mini-batch SGD and momentum or Adam.
- Replace FC with a small ConvNet for much better accuracy.

---

## License & credits

Educational project. Feel free to reuse for learning and experimentation.

---
