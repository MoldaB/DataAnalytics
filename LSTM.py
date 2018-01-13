import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
import pandas as pd
import os

ACTIVITIES = {
    0: 'Forehand',
    1: 'Backhand',

}

def confusion_matrix(Y_true, Y_pred):
    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])

    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])
# for reproducibility
# https://github.com/fchollet/keras/issues/2280
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
import pandas as pd



SIGNALS = [
    "accumulatedYawRotation",
    "peakRate",
    "yawThreshold.bool",
    "rawThreshold.bool",

]

# # Train Set + Text Set
path_x_train = os.getcwd() + '/Data/Train/X_train.txt'
path_y_train = os.getcwd() + '/Data/Train/y_train.txt'
path_x_test = os.getcwd() + '/Data/Test/X_test.txt'
path_y_test = os.getcwd() + '/Data/Test/y_test.txt'

def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        filename = path_x_train if subset == "train" else path_x_test
        signals_data.append(
            _read_csv(filename).as_matrix()
        )


    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):

    filename = path_y_train if subset == "train" else path_y_test
    y = _read_csv(filename)[0]

    return pd.get_dummies(y).as_matrix()

def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train = load_signals('train')
    X_test = load_signals('test')
    y_train = load_y('train')
    y_test = load_y('test')
    return X_train,X_test,y_train,y_test

from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

epochs = 30
batch_size = 16
n_hidden = 32

def _count_classes(y):
    return len(set([tuple(category) for category in y]))

X_train, X_test, Y_train, Y_test = load_data()

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=epochs)

# Evaluate
print(confusion_matrix(Y_test, model.predict(X_test)))