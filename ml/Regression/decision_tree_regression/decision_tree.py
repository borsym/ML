import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
reg_1 = DecisionTreeRegressor(max_depth=2, random_state=0)
reg_2 = DecisionTreeRegressor(max_depth=5, random_state=0)

reg_1.fit(X,Y)
reg_2.fit(X,Y)

Y_pred_1 = reg_1.predict([[6.5]])
Y_pred_2 = reg_2.predict([[6.5]])

print(Y_pred_1, Y_pred_2, sep = "\n")





X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, reg_2.predict(X_grid), color = 'blue')
plt.plot(X_grid, reg_1.predict(X_grid), color = 'yellow')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()