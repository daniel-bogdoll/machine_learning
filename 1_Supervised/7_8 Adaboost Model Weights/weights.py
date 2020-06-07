import numpy as np

def accuracy(right,total):
    return right/total

one_r = 7   #right
one_w = 1   #wrong

two_r = 4
two_w = 4

three_r = 2
three_w = 6

one_acc = accuracy(one_r, 8)
two_acc = accuracy(two_r, 8)
three_acc = accuracy(three_r, 8)

one_weight = np.log(one_acc/(1-one_acc))    #solution is 1.945 --> Udacity wants 1.95 (2 significant digits)
one_weight = np.log(one_r/one_w)            #Same result
print(one_weight)

