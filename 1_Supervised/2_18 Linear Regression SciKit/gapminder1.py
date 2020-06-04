#DataPrep: https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
#Plotting: https://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot

# TODO: Add import statements
from sklearn.linear_model import LinearRegression


import numpy as np  #Deal with math
import pandas as pd #Deal with data
import matplotlib.pyplot as plt

# Assign the dataframe to this variable.
# TODO: Load the data
csv = pd.read_csv('bmi_and_life_expectancy.csv')
bmi_life_data = pd.DataFrame(csv)

#Plot input data
bmi_life_data.plot(x='BMI', y='Life expectancy', style='o')  
plt.title('BMI vs Life expectancy')  
plt.xlabel('BMI')  
plt.ylabel('Life expectancy')  

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931

X = bmi_life_data['BMI'].values.reshape(-1,1)
y = bmi_life_data['Life expectancy'].values.reshape(-1,1)

bmi_life_model.fit(X,y)
bmi_predict = 21.07931

#y = wx + b
w = bmi_life_model.coef_[0][0]
b = bmi_life_model.intercept_[0]
laos_life_exp = w * bmi_predict + b
print("Laos Life expectancy", laos_life_exp)

#Plot results
x_min = X.min()
x_max = X.max()
y_min = w * x_min + b
y_max = w * x_max + b
plt.plot([x_min, x_max], [y_min, y_max])
plt.plot(bmi_predict, laos_life_exp, color='green', marker='o')
plt.show()
