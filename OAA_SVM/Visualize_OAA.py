import matplotlib.pyplot as plt
import numpy as np


def draw_line(x, w, b, offset):
    """
    simple function to determine the y-values for x-values in the hyperplane such that np.dot(w,x)+b=0
    Args:
        x: the x-value
        w: the weights
        b: the bias
        offset:

    Returns: the y-value for the respective x-value.
    Remark: since we draw a simple line we only need a single point to return, as we can simply draw the line that
    connects both points with respect to the trained weights and bias (those 2 points will be the most left and right)

    """
    return (-w[0] * x + b + offset) / w[1]


def visualize_oaa_svm(X, y, weights, bias):
    """
    a function to visualize the svm.
    First we define the most left and most right point to draw the lines from.
    Args:
        X: The Datapoints to visualize
        y: the classes to color them
        weights: a list of weights, one entry for each model we trained
        bias: a list of biases, one entry for each model we trained

    Returns: a plot that shows the hyperplane of our svm with the datapoints.

    """
    plt.style.use('seaborn')

    # Define color map for classes
    cmap = plt.cm.get_cmap('cividis')

    # create the subplot and plot the data points
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], marker="o", c=y, cmap=cmap)

    # get the minimal and maximal values for the X and Y axis such that we can work with them
    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])

    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    for w, b in zip(weights, bias):
        # Get the hyperplane value for the most left and most right point
        hyperplane_l = draw_line(x_min, w, b, 0)
        hyperplane_r = draw_line(x_max, w, b, 0)

        # Plot all of that
        ax.plot([x_min, x_max], [hyperplane_l, hyperplane_r], "r", alpha=0.5)

    ax.set_ylim([y_min, y_max])

    ax.grid(c='white')

    plt.show()
