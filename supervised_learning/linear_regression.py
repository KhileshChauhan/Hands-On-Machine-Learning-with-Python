# Exercise 7.4

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
response = np.arange(100) + 10.01 + np.random.normal(10, 20.0, 100)
ax.scatter(np.arange(100), response)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Scatter plot of data points")
plt.show()


# Two points drawn at random
y11 = np.random.randint(0, 100, size=(1))
y12 = np.random.randint(0, 100, size=(1))
print(y11, y12)
x11 = response[y1]
x12 = response[y2]
ax.plot([0, 99], [y12, y12], 'g--')

y21 = np.random.randint(0, 100, size=(1))
y22 = np.random.randint(0, 100, size=(1))
print(y21, y22)
x21 = response[y1]
x22 = response[y2]
ax.plot([0, 99], [y21, y22], 'g--')


# Activity 1 -- about linear regression
import numpy as np
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split

x1 = np.random.normal(0, 1, 10000)
x2 = np.random.normal(5, 10, 10000)
x3 = np.random.normal(1, 10, 10000)
x4 = np.random.normal(15, 15, 10000)

X = np.zeros((10000, 4))

X[:,0] = x1
X[:,1] = x2
X[:,2] = x3
X[:,3] = x4

y = (0.5 * x1) - (0.7 * x2) + (10 * x3) - (0.1 * x4) + 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


start_time = time.time()
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr)
y_pred = regr.predict(X_test)

end_time = time.time()
print("Time taken to predict: {} seconds".format(end_time - start_time))
print("W0: {}, w1: {}, w2: {}, w3: {}, w4: {}".format(regr.intercept_, regr.coef_[0], regr.coef_[1], regr.coef_[2], regr.coef_[3]))
print("Mean squared error is: ", mean_squared_error(y_pred, y_test))

start_time = time.time()
clf = linear_model.SGDRegressor(max_iter=100, tol=1e-5)
clf.fit(X_train, y_train)
print("gradient descent regressor", clf)
pred = clf.predict(X_test)
end_time = time.time()
print("Time taken to predict: {} seconds".format(end_time - start_time))
print("W0: {}, w1: {}, w2: {}, w3: {}, w4: {}".format(clf.intercept_, clf.coef_[0], clf.coef_[1], clf.coef_[2], clf.coef_[3]))
print("Mean squared error is: ", mean_squared_error(pred, y_test))

