from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

import numpy as np
import matplotlib.pyplot as plt

# Load the data from the boston house-prices dataset 
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
# TODO: Fit the model and assign it to the model variable
model = LinearRegression()
model.fit(x,y)

# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house

numberOfDataPoints = len(x[0])

print("Number of data points:", numberOfDataPoints)
W = model.coef_
b = model.intercept_

#y =wx + b;

print("W",W)
print("Sample",sample_house)
prediction = np.matmul(W, sample_house[0]) + b
print("Price:",prediction)