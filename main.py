import pandas as pd
import os
import numpy as np
#
# # display pandas results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#
# # features
path_features = os.getcwd() + '/Data/features.txt'
# # calssifications labels
path_activity_labels = os.getcwd() + '/Data/activity_labels.txt'
# # Train Set + Text Set
path_x_train = os.getcwd() + '/Data/Train/X_train.txt'
path_y_train = os.getcwd() + '/Data/Train/y_train.txt'
path_x_test = os.getcwd() + '/Data/Test/X_test.txt'
path_y_test = os.getcwd() + '/Data/Test/y_test.txt'
#
# # read features
features = None
with open(path_features) as handle:
    features_raw = handle.readlines()
    features = list(map(lambda x: x.strip(), features_raw))
#
# # read classification labels
activity_labels = None
with open(path_activity_labels) as handle:
    activity_labels_raw = handle.readlines()
    activity_labels = list(map(lambda x: x.strip(), activity_labels_raw))

activity_df = pd.DataFrame(activity_labels)
activity_df = pd.DataFrame(activity_df[0].str.split(' ').tolist(), columns = ['activity_id', 'activity_label'])
# activity_df
#
# # reads train and test set

x_train = pd.read_table(path_x_train, header = None, sep = "\t", names = features)
x_train['timestamp'] = pd.to_numeric(x_train['timestamp'])
# print(x_train.iloc[:10, :10].head())
#
y_train = pd.read_table(path_y_train, header = None, sep = "\t", names = ['activity_id'])
# print(y_train.head())
#
x_test = pd.read_table(path_x_test, header = None, sep = "\t", names = features)
x_test['timestamp'] = pd.to_numeric(x_test['timestamp'])
y_test = pd.read_table(path_y_test, header = None, sep = "\t", names = ['activity_id'])
#
#
# # Builds learning model using Grid Search
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
#
# Cs = np.logspace(-6, 3, 10)
# parameters = [{'kernel': ['rbf'], 'C': Cs},
#               {'kernel': ['linear'], 'C': Cs}]
#
# svc = SVC()
#
# clf = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
# clf.fit(x_train.values, y_train.values.flatten())
#
# print (clf.best_params_)
# print (clf.best_score_)
#
#

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

epochs = 30
batch_size = 16
n_hidden = 32

timesteps = len(x_train)
input_dim = len(features_raw)
n_classes = len(activity_labels)

model = Sequential()
model.add(LSTM(n_hidden, return_sequences=True, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          validation_data=(x_test, y_test),
          epochs=epochs)

# from sklearn.neighbors import  KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=5);
# neigh.fit(x_train.values,y_train.values.flatten())
# print(neigh.score(x_test.values,y_test.values.flatten()))

# Rando Forest Model
# from sklearn import ensemble
#
# rf = ensemble.RandomForestClassifier(20)
#
# rf.fit(x_train.values, y_train.values.flatten())
# y_pred = rf.predict(x_test)
# score = rf.score(x_test, y_test)
# print(score)