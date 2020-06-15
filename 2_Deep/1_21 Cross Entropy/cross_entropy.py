import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.


def cross_entropy(Y, P):
    crossEntropy = 0
    for y, p in zip(Y, P):
        entry = y * np.log(p) + (1-y) * np.log(1-p)
        crossEntropy += entry

    crossEntropy = crossEntropy * (-1)
    return crossEntropy


testY = [1, 1, 0]
testP = [0.8, 0.7, 0.1]
result = cross_entropy(testY, testP)
print(result)
