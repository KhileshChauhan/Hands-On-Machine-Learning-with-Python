import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

x1 = np.random.normal(0, 10, 1000)
x2 = np.random.normal(10, 1, 1000)

X = np.zeros((1000, 2))
X[:,0] = x1
X[:,1] = x2

y = (0.5 * x1 ** 2) - (0.7 * x1 * x2) + 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
poly = PolynomialFeatures(degree=2)
X_train_transform = poly.fit_transform(X_train)

print(X_train_transform.shape)
X_test_transform = poly.fit_transform(X_test)

regr = linear_model.LinearRegression()
regr.fit(X_train_transform, y_train)
print(regr)

predictions = regr.predict(X_test_transform)
print(regr.coef_)
print(regr.intercept_)
print("W0: {}, w1: {}, w2: {}, w3: {}, w4: {}, w5 : {}, w6: {}".format(round(regr.intercept_), round(regr.coef_[0]), round(regr.coef_[1]), round(regr.coef_[2]), round(regr.coef_[3]), round(regr.coef_[4]), round(regr.coef_[5])))
print("Mean squared error is: ", round(mean_squared_error(predictions, y_test)))


# Regularization

import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

data = load_boston()
df_x = pd.DataFrame(data.data, columns=data.feature_names)
df_targets = pd.DataFrame(data.target)

X_train, X_test, y_train, y_test = train_test_split(df_x, df_targets, test_size=0.30, random_state=42)

poly = PolynomialFeatures(degree=2)
X_transform = poly.fit_transform(X_train)
regr = linear_model.LinearRegression()
regr.fit(X_transform, y_train)
print(regr)


X_test_transform = poly.fit_transform(X_test)
pred = regr.predict(X_test_transform)

lasso = Lasso(alpha= 0.5, fit_intercept = True)
lasso.fit(X_transform, y_train)

p = lasso.predict(X_test_transform)

ridge = Ridge(alpha=10, fit_intercept = True)
ridge.fit(X_transform, y_train)

r = ridge.predict(X_test_transform)



# print(regr.coef_)
# print(regr.intercept_)
print("Mean squared error is: ", mean_squared_error(pred, y_test))
print("Mean squared error for Lasso is: ", mean_squared_error(p, y_test))
print("Mean squared error for Ridge is: ", mean_squared_error(r, y_test))


