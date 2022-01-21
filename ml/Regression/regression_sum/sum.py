import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

# compare
compare = {}
#import
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
# print(X, Y, sep='\n')
#train/test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#### train ####
#Multi
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
Y_pred = lin_reg.predict(X_test)
compare[r2_score(Y_test, Y_pred)] = 'multi'
#np.set_printoptions(precision=0)
#print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#Poly
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
lin_reg.fit(X_poly, Y_train)
Y_pred = lin_reg.predict(poly_reg.transform(X_test)) 
compare[r2_score(Y_test, Y_pred)] = 'poly'
#SVR
#feuter scaling 
from sklearn.preprocessing import StandardScaler
Y_reshaped = Y.reshape(len(Y), 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_reshaped, test_size = 0.2, random_state = 0)
sc_X = StandardScaler()
sc_Y = StandardScaler()

X_train_SVR = sc_X.fit_transform(X_train)
Y_train_SVR = sc_Y.fit_transform(Y_train)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train_SVR, Y_train_SVR)

Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_test)))

compare[r2_score(Y_test, Y_pred)] = 'svr'
#Decision
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
compare[r2_score(Y_test, Y_pred)] = 'decision_tree'

#Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
compare[r2_score(Y_test, Y_pred)] = 'random_forest'



print("The best performance comes from: ", compare[max(compare)])
