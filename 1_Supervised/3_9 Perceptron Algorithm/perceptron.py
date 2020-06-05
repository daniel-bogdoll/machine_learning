import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.


def perceptronStep(X, y, W, b, learn_rate=0.01):
    for point, label in zip(X, y):
        myPrediction = prediction(point, W, b)
        if myPrediction != label:
            if myPrediction == 0:  # y-y_hat = -1
                W += np.array(learn_rate * point).reshape(2, 1)
                b += learn_rate
            elif myPrediction == 1:  # y-y_hat = 1
                W -= np.array(learn_rate * point).reshape(2, 1)
                b -= learn_rate
    return W, b


# def perceptronStep(X, y, W, b, learn_rate = 0.01):
#    for i in range(len(X)):
#        y_hat = prediction(X[i],W,b)
#        print(y_hat)
#        if y[i]-y_hat == 1:
#            W[0] += X[i][0]*learn_rate
#            W[1] += X[i][1]*learn_rate
#            b += learn_rate
#        elif y[i]-y_hat == -1:
#            W[0] -= X[i][0]*learn_rate
#            W[1] -= X[i][1]*learn_rate
#            b -= learn_rate
#    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


def main():
    train_data = pd.read_csv('data.csv', header=None)
    # Python does !!not!! slice inclusive of the ending index
    X = train_data.iloc[:, 0:2]
    X = X.values

    X_values = X[:, 0]
    Y_values = X[:, 1]
    y = train_data.iloc[:, 2].values  # label data
    colors = ['red', 'blue']
    plt.scatter(X_values, Y_values, c=y,
                cmap=matplotlib.colors.ListedColormap(colors))

    trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25)
    # plt.show()


if __name__ == '__main__':
    main()
