import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit


def read_csc(path):
    data = []
    with open(path) as f:
        line = f.readline()
        while line:
            T = [float(num) for num in line.split(',')]
            data.append(T)
            line = f.readline()
    data = np.array(data)
    X = data[:, 1:]
    Y = data[:, 0]
    return X, Y


def cal_p_Yey(Y, y):
    """cal P(Y=y)"""
    y_index = np.where(Y == y, 1, 0).nonzero()
    return len(y_index) / len(Y)


def cal_p_Xex(X, x):
    """cal P(X=x) x is an array"""
    row = X.shape[0]
    count = 0
    for i in range(row):
        if np.equal(X[i], x).all():
            count += 1
    return count / row


def cal_p_Xx_when_Yy(X, x, Y, y):

    """cal P(X=x|Y=y) """
    y_index = np.where(Y == y, Y, 0).nonzero()
    x_model = X[y_index]
    return cal_p_Xex(x_model, x)



def cal_p_Yy_when_Xx(X, x, Y, y):
    """cal P(Y=y|X=x)"""
    P_Xx_when_Yy = cal_p_Xx_when_Yy(X, x, Y, y)
    P_Yy = cal_p_Yey(Y, y)
    P_Xx = cal_p_Xex(X, x)
    return P_Xx_when_Yy * P_Yy / P_Xx



def predict(X, x, Y):
    """predict y when X=y"""
    y = Y[0]
    P_Yy_whenXx = 0
    for c_k in set(Y):
        P = cal_p_Yy_when_Xx(X, x, Y, c_k)
        if P_Yy_whenXx < P:
            y = c_k
            P_Yy_whenXx = P
    return y





if __name__ == '__main__':
    X, Y = read_csc("wine.data")
    # 函数本身能保证分层采样
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
    Y_predicted = []
    for x in X_test:
        y = predict(X_train, x, Y_train)
        Y_predicted.append(y)
    Y_predicted = np.array(Y_predicted)

    # 代价敏感
    w = 1

    e = Y_predicted - Y_test
    err = len(np.array(e).nonzero()) * w / len(Y_test)
    print(err)






