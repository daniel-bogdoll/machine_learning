#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

# TODO: Add import statements
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

#Solution does the import like that:
train_data = pd.read_csv('data.csv', header = None)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]


# TODO: Create the standardization scaling object.
scaler = StandardScaler()
scaler.fit(X)

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = linear_model.Lasso()

# TODO: Fit the model.
lasso_reg.fit(X_scaled,y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
