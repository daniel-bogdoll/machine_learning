import numpy as np
import math

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    propabilities = np.empty(len(L))
    denominator = 0

    #Compute sum
    for score in L:
        denominator += pow(math.e,score)
    print (denominator)

    #Compute probabilities
    i = 0
    for score in L:
        probability = pow(math.e,score) / denominator
        propabilities[i] = probability
        i += 1

    return propabilities

test = [5,6,7]
result = softmax(test)
print(result)