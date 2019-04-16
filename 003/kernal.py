import sklearn.neighbors.kde as kde
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, train_test_split


def read_csc(path):
    data = []
    with open(path) as f:
        line = f.readline()
        while line:
            T = [float(num) for num in line.split(',')]
            data.append(T)
            line = f.readline()
    data = np.array(data)
    X = data[:, 0:4]
    Y = data[:, -1]
    return X, Y


if __name__ == '__main__':
    X, Y = read_csc("HWData3.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 0)

    pattern1 = kde.KernelDensity(kernel='gaussain', bandwidth=0.7).fit(X_train)

    pattern1.score()




