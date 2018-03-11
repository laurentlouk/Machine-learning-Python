# Decision Tree Regression

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Datas/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

print(dataset)

# Fitting the Decision tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results (High resolution)
X_gird = np.arange(min(X), max(X), 0.01)
X_gird = X_gird.reshape(len(X_gird), 1)
plt.scatter(X, Y, color = 'red')
plt.plot(X_gird, regressor.predict(X_gird), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()