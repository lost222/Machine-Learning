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


def cal_m(X_mat):
    return np.mean(X_mat, axis=0)

def cal_cov(x, y):
    m_x = np.mean(x)
    m_y = np.mean(y)
    return np.sum((x - m_x) * (y - m_y)) / len(x)

def cal_gaussian_P(x, mu, sq_sigma):
    a = -(x - mu)**2 / (2*sq_sigma)
    return np.e**a / (np.sqrt(2*np.pi)*np.sqrt(sq_sigma))

class Bayes:
    mu_list = []
    sq_sigma_list = []

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        (row, col) = X.shape
        for i in range(col):
            x_i = X[:, i]
            self.mu_list.append(cal_m(x_i))
            self.sq_sigma_list.append(cal_cov(x_i, x_i))

    def cal_p_Yey(self, y):
        """cal P(Y=y)"""
        a = np.sum(self.Y == y) / self.Y.shape[0]
        return a

    def cal_p_Xex(self, x):
        """cal P(X=x) x is an array"""
        P_list = []
        for i in range(len(x)):
            P_list.append(cal_gaussian_P(x[i], self.mu_list[i], self.sq_sigma_list[i]))
        P_list = np.log(P_list)
        return np.e**np.sum(P_list)


def cal_p_Xx_when_Yy(X, x, Y, y):

    """cal P(X=x|Y=y) """
    y_index = np.where(Y == y, Y, 0).nonzero()
    n_m = Bayes(X[y_index], Y[y_index])
    return n_m.cal_p_Xex(x)



def cal_p_Yy_when_Xx(X, x, Y, y):
    """cal P(Y=y|X=x)"""
    P_Xx_when_Yy = cal_p_Xx_when_Yy(X, x, Y, y)
    ba = Bayes(X, Y)
    P_Yy = ba.cal_p_Yey(y)
    P_Xx = ba.cal_p_Xex(x)
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
    print(e)
    err = np.sum(e != 0) * w / len(e)
    print(err)






