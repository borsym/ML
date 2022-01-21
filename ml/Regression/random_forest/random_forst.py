import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
# how many tree?  10 
reg_2 = RandomForestRegressor(n_estimators=10, random_state=0)

reg_2.fit(X,Y)
Y_pred_2 = reg_2.predict([[6.5]])
print(Y_pred_2)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, reg_2.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()