# https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

import operator

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# extend the predictor feature column into multiple columns
# with polynomial features.

# Assign the data to predictor and outcome variables
# TODO: Load the data
training_data = pd.DataFrame(pd.read_csv('data.csv'))
#X = training_data['Var_X'].to_numpy()
#y = training_data['Var_Y'].to_numpy()

# to_numpy() too new for Udacity
X = np.asarray(training_data['Var_X'].values.tolist())
y = np.asarray(training_data['Var_Y'].values.tolist())

X = X.reshape((len(X), 1))

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature

# Play around with different degrees of polynomial
# and test to see what fits best

degreeRange = 50    #Udacity solution was degree = 4
                    #My solution says degree = 18, but that is probably bad because of overfitting!
results = []

color_idx = np.linspace(0, 1, degreeRange)
poly_model = LinearRegression()

for degree in range(1, degreeRange):
    index = degree-1
    myColor = color_idx[index]
    poly_feat = PolynomialFeatures(degree)
    X_poly = poly_feat.fit_transform(X)

    # Make and fit the polynomial regression model
    # TODO: Create a LinearRegression object and fit it to the polynomial predictor
    # features
    poly_model.fit(X_poly, y)
    y_poly_pred = poly_model.predict(X_poly)

    # Once you've completed all of the steps, select Test Run to see your model
    # predictions against the data, or select Submit Answer to check if the degree
    # of the polynomial features is the same as ours!
    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)
    results.append([degree, rmse, r2])
    print("Degree", degree, "| RMSE:", rmse, "| R2:", r2)

    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    useColor = plt.cm.cool(myColor)
    myWidth = 2
    myzorder = 1
    if degree == 4:
        useColor="green"
        myWidth = 10
        myzorder = 98
    elif degree == 18:
        useColor="red"
        myWidth = 10
        myzorder = 97
    plt.plot(x, y_poly_pred, color=useColor, lineWidth = myWidth, zorder = myzorder)

bestFit = np.argmin(results, axis=0)
bestrmseDegree = bestFit[1]
print("Best fit is degree", bestrmseDegree)

#Recompute for Udacity solution check
degree = bestrmseDegree
poly_feat = PolynomialFeatures(degree)
X_poly = poly_feat.fit_transform(X)
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

plt.scatter(X, y, s=100,color="black",zorder=99)
plt.show()
