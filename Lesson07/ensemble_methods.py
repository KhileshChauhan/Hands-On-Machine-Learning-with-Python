# Code snippet 1

import random
import numpy as np

S = [2,4,6,8,10,12,14,16,18,20]
# Approach 1:
sample1 = random.choice(S)
S.remove(sample1)
sample2 = random.choice(S)

# Approach 2: 
np.random.choice(S, 2)


# Adaboost implemnetation

from sklearn.datasets import load_boston
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

# print(X_test)

# print(data)

# data has 178 rows and 13 columns; 178 data points and 13 predictors 
df_x = pd.DataFrame(data.data, columns=data.feature_names)
df_targets = pd.DataFrame(data.target)
# print(df_x)
bdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), 
                         n_estimators=1000)

adaboost_model = bdt.fit(X_train, y_train)
adaboost_model.estimators_
pred = adaboost_model.predict(X_test)

np.sum(abs(pred - y_test))/len(pred)

