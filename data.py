import pandas as pd
import numpy as np
import os

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
        filename = '{DATADIR}/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(
            _read_csv(filename).as_matrix()
        )


    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):

    filename = '{DATADIR}/{subset}/y_{subset}.txt'
    y = _read_csv(filename)[0]

    return pd.get_dummies(y).as_matrix()

def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_y('train'), load_y('test')

    return X_train, X_test, y_train, y_test