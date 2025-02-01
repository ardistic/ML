import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

# -------------------------
# Data Loading & Preprocessing
# -------------------------
data = pd.read_csv('/home/arduuh/digit_recognizer_data/train.csv')  # Adjust path if needed
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Create development set (first 1000 samples)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

# Create training set (remaining samples)
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
_, m_train = X_train.shape  # m_train is the number of training examples

# -------------------------
# Neural Network Functions
# -------------------------
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2 

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Subtract max for numerical stability (broadcasted over columns)
    Z -= np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1      # (10, m)
    A1 = ReLU(Z1)                # (10, m)
    Z2 = np.dot(W2, A1) + b2      # (10, m)
    A2 = softmax(Z2)             # (10, m)
    return Z1, A1, Z2, A2 

def ReLU_deriv(Z):
    return (Z > 0).astype(float)

def one_hot(Y):
    Y = Y.astype(int).flatten()  # Ensure Y is a 1D integer array
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m_samples = Y.size  # Number of training samples
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y                   # (10, m)
    dW2 = (1 / m_samples) * np.dot(dZ2, A1.T)  # (10, 10)
    db2 = (1 / m_samples) * np.sum(dZ2, axis=1, keepdims=True)  # (10, 1)
    dZ1 = np.dot(W2.T, dZ2) * ReLU_deriv(Z1)  # (10, m)
    dW1 = (1 / m_samples) * np.dot(dZ1, X.T)   # (10, 784)
    db1 = (1 / m_samples) * np.sum(dZ1, axis=1, keepdims=True)  # (10, 1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y.flatten()) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print("Iteration:", i, "Accuracy:", accuracy)
    return W1, b1, W2, b2

# -------------------------
# Train the Neural Network
# -------------------------
print("Training started...")
start_time = time.time()
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 450)
training_time = time.time() - start_time
print("Training finished in {:.2f} seconds".format(training_time))

# Evaluate on the development set
_, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
dev_accuracy = get_accuracy(get_predictions(A2_dev), Y_dev)
print("Dev Set Accuracy: {:.2f}%".format(dev_accuracy * 100))


