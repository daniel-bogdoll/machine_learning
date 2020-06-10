import math

#point = [1,1]
#point = [2,4]
#point = [5,-5]
point = [-4,5]

score = 4*point[0] + 5*point[1] - 9

sigmoid = 1/(1 + pow(math.e,(score*-1)))    #sigmoid is 0.5 if score is 0
print(sigmoid)