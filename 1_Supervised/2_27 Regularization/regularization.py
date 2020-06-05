#In this assignment's data.csv, you'll find data for a bunch of points
# including six predictor variables and one outcome variable.
# Use sklearn's Lasso class to fit a linear regression model to the data,
# while also using L1 regularization to control for model complexity.

# TODO: Add import statements
import numpy as np
import pandas as pd

from sklearn import linear_model


# Assign the data to predictor and outcome variables
# TODO: Load the data
# Split the data so that the six predictor features (first six columns) are stored in X,
# and the outcome feature (last column) is stored in y.

train_data_csv = pd.read_csv('data.csv')    #Does not read header because of names! Creates small error!
X_me = train_data_csv.values[:, [0,1,2,3,4,5]]
y_me = train_data_csv.values[:, [6]]

#Solution does the import like that:
train_data = pd.read_csv('data.csv', header = None)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = linear_model.Lasso()
lasso_reg_me = linear_model.Lasso()

# TODO: Fit the model.
lasso_reg.fit(X,y)
lasso_reg_me.fit(X_me,y_me)


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
reg_coef_me = lasso_reg_me.coef_

print(reg_coef)
print(reg_coef_me)