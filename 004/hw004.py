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
    a = -(x - mu) ** 2 / (2 * sq_sigma)
    return np.e ** a / (np.sqrt(2 * np.pi) * np.sqrt(sq_sigma))


class Bayes:
    def __init__(self, X, Y):
        self.mu_list = []
        self.sq_sigma_list = []
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
        return np.log(a)

    def cal_p_Xex(self, x):
        """cal P(X=x) x is an array"""
        P_list = []
        for i in range(len(x)):
            P_list.append(cal_gaussian_P(x[i], self.mu_list[i], self.sq_sigma_list[i]))
        P_list = np.log(P_list)
        return np.sum(P_list)


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
    # P_Xx = ba.cal_p_Xex(x)
    return P_Xx_when_Yy + P_Yy


def predict(X, x, Y):
    """predict y when X=y"""
    y = Y[0]
    P_Yy_whenXx = []
    for c_k in range(3):
        P = cal_p_Yy_when_Xx(X, x, Y, c_k + 1)
        P_Yy_whenXx.append(P)
    return np.argmax(P_Yy_whenXx) + 1


def predict_threshold(X, x, Y, c, t):
    P = cal_p_Yy_when_Xx(X, x, Y, c)
    return t < P


if __name__ == '__main__':
    X, Y = read_csc("wine.data")
    # 函数本身能保证分层采样
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

    # Y_predicted = []
    # for x in X_test:
    #     y = predict(X_train, x, Y_train)
    #     Y_predicted.append(y)
    # Y_predicted = np.array(Y_predicted)
    #
    # # 代价敏感
    # w = 1
    #
    # e = Y_predicted - Y_test
    # print(Y_predicted)
    # print(Y_test)
    # err = np.sum(e != 0) * w / len(e)
    # print(err)
    # # cal Confusion_Matrix
    #
    # for to in [1, 2, 3]:
    #     CM = np.zeros((2, 2))
    #     for i in range(len(Y_test)):
    #         T = Y_test[i]
    #         P = Y_predicted[i]
    #         if T == to:
    #             if P == to:
    #                 CM[0, 0] += 1
    #             else:
    #                 CM[0, 1] += 1
    #         else:
    #             if P == to:
    #                 CM[1, 0] += 1
    #             else:
    #                 CM[1, 1] += 1
    #     print("Confusion_Matrix of class " + str(to))
    #     print(CM)

    # ROC
    cla = 3

    threadH = []

    for x in X_test:
        p = cal_p_Yy_when_Xx(X_train, x, Y_train, cla)
        threadH.append(p)




    Y_test_u = np.where(Y_test == cla, True, False)
    FPRs = []
    TPRs = []
    mi = int(np.min(threadH))
    ma = int(np.max(threadH))
    step = int((ma - mi) / 10)
    for t in range(mi, ma):
        th = t
        Y_predicted = []
        for x in X_test:
            p = predict_threshold(X_train, x, Y_train, cla, th)
            Y_predicted.append(p)
        # CM
        CM = np.zeros((2, 2))
        for i in range(len(Y_test_u)):
            T = Y_test_u[i]
            P = Y_predicted[i]
            if T:
                if P:
                    CM[0, 0] += 1
                else:
                    CM[0, 1] += 1
            else:
                if P:
                    CM[1, 0] += 1
                else:
                    CM[1, 1] += 1
        TPR = CM[0, 0] / (CM[0, 0] + CM[1, 1])
        FPR = CM[1, 0] / (CM[0, 1] + CM[1, 0])
        TPRs.append(TPR)
        FPRs.append(FPR)

    print(FPRs)
    print(TPRs)
    plt.plot(FPRs, TPRs, '^-')
    plt.show()