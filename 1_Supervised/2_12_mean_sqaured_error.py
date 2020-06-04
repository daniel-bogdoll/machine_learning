import numpy as np

# Mean squared error
# line: y = 1.2x + 2
# line: y = w1x + w2
# points: (2, -2), (5, 6), (-4, -4), (-7, 1), (8, 14)

line_w1 = 1.2
line_w2 = 2
points = np.array([[2, -2], [5, 6], [-4, -4], [-7, 1], [8, 14]])
numberOfPoints = len(points) 

sumOfErrors = 0
for point in points:
    x_point = point[0]
    y_point = point[1]
    print (x_point , " " , y_point)
    y_line = line_w1 * x_point + line_w2
    error = pow((y_point - y_line),2)
    print (error)
    sumOfErrors += error

print ("Sum of errors ", sumOfErrors)
meanAbsoluteError = 1/(2*numberOfPoints) * sumOfErrors
print("Mean Absolute Error ", meanAbsoluteError)
