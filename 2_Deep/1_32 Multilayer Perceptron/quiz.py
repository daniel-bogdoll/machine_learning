import math
import numpy as np


def sigmoid(x):
    sigmoid = 1/(1 + pow(math.e, -x))
    return sigmoid


x1 = 0.4
x2 = 0.6

model1 = [5, -2, -8]  # w1, w2, b
model2 = [7, -3, 1]
model3 = [7, 5, -6]

probability_model_1 = sigmoid(x1 * model1[0] + x2 * model1[1] + model1[2])
probability_model_2 = sigmoid(x1 * model2[0] + x2 * model2[1] + model2[2])

x1_new = probability_model_1
x2_new = probability_model_2

probability_combined = sigmoid(x1_new * model3[0] + x2_new * model3[1] + model3[2])

# w1 * x1 + w2 * 0.6 + b = 0.88
option1 = [2, 6, -2]
option2 = [3, 5, -2.2]
option3 = [5, 4, -3]

prob1 = sigmoid(x1 * option1[0] + x2 * option1[1] + option1[2])
prob2 = sigmoid(x1 * option2[0] + x2 * option2[1] + option2[2])
prob3 = sigmoid(x1 * option3[0] + x2 * option3[1] + option3[2])
print(prob1)
print(prob2)
print(prob3)
