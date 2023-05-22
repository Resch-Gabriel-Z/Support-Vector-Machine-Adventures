import numpy as np


def gradient_descent(weights, bias, lr, X, y, steps):
    """
    a method to implement gradient descent for hard-margin svm's.
    This is not a general gradient descent algorithm but inherently implemented the gradients of the objective function
    in the unconstrained convex hard-margin svm.

    Args:
        weights: the current weights of the model
        bias: the current bias
        lr: the learning rate
        X: the datapoints of our dataset
        y: the classes of our dataset
        steps: the number of iterations

    Returns: the updated weights and bias

    """
    for _ in range(steps):
        for idx, x_i in enumerate(X):
            if y[idx] * (np.dot(x_i, weights) - bias) < 1:
                bias -= lr * y[idx]
                weights += lr * (y[idx] * x_i)

    return weights, bias


def stochastic_gradient_descent(weights, bias, lr, X, y, batch_size, steps):
    """
    a method to implement the stochastic gradient descent for hard-margin svm's.
    As above, we do not implement a general version.
    We create a batch index each iteration to only train for a subset instead of the whole dataset.

    Args:
        weights: the current weights of the model
        bias: the current bias
        lr: the learning rate
        X: the datapoints of our dataset
        y: the classes of our dataset
        batch_size: the size of the subset we train on
        steps: the number of iterations

    Returns: trained weights and bias

    """
    for _ in range(steps):
        batch_idx = np.random.choice(np.arange(len(X)), batch_size, replace=False)
        X_b = X[batch_idx]
        y_b = y[batch_idx]

        for idx, x_i in enumerate(X_b):
            if y_b[idx] * (np.dot(x_i, weights) - bias) < 1:
                bias += lr * y_b[idx]
                weights += lr * (y_b[idx] * x_i)

    return weights, bias
