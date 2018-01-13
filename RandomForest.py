import numpy as np
import pandas as pd
import sklearn.ensemble

train_data = pd.read_csv('X_train.txt')
test_data = pd.read_csv('X_test.txt')


X_train = train_data.drop('Activity', axis = 1)
y_train = pd.get_dummies(train_data.Activity)
X_test = test_data.drop('Activity', axis = 1)
y_test = pd.get_dummies(test_data.Activity)
rf = sklearn.ensemble.RandomForestClassifier(20)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf.score(X_test, y_test)
