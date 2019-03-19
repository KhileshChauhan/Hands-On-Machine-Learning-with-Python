from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score



data = load_wine()
df_x = pd.DataFrame(data.data, columns=data.feature_names)
df_targets = pd.DataFrame(data.target)

X_train, X_test, y_train, y_test = train_test_split(df_x, df_targets, test_size=0.20, random_state=42)
# print(y_train)
clf = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(10, 10, 5), random_state=1)
clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test)
# print("Mean squared error is: ", round(mean_squared_error(pred, y_test)))
confusion_matrix(y_pred, y_test)
# score(clf, y_test, y_pred)
accuracy_score(y_pred, y_test)