import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold




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


def cal_m(X_mat):
    return np.mean(X_mat, axis=0)

def cal_cov(x, y):
    m_x = np.mean(x)
    m_y = np.mean(y)
    return np.sum((x - m_x) * (y - m_y)) / len(x)

def cal_sigma(X_mat):
    col = np.shape(X_mat)[1]
    sigma = np.zeros((col, col))
    for i in range(col):
        for j in range(i, col):
            sigma[i, j] = cal_cov(X_mat[:, i], X_mat[:, j])

    for i in range(col):
        for j in range(0, i):
            sigma[i, j] = sigma[j, i]

    return sigma


def cal_P(x, sigma, mu):
    d = len(x)
    a0 = (2*np.pi)**(d/2) * np.sqrt(np.linalg.det(sigma))
    a1 = np.dot(np.transpose(x-mu), np.linalg.inv(sigma))
    a2 = np.dot(a1, (x - mu))
    return np.e**(-0.5*a2) / a0





if __name__ == '__main__':
    pa = "HWData3.csv"
    X, Y = read_csc(pa)
    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)

    err_rate_list = []


    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        # 分三类分别计算 mu 和 sigma
        first_index = np.where(Y_train==1, Y_train, 0).nonzero()
        sec_index = np.where(Y_train==2, Y_train, 0).nonzero()
        third_index = np.where(Y_train==3, Y_train, 0).nonzero()

        # X_1
        X_1 = X_train[first_index]
        mu_X_1 = cal_m(X_1)
        sigma_X_1 = cal_sigma(X_1)

        # X_2
        X_2 = X_train[sec_index]
        mu_X_2 = cal_m(X_2)
        sigma_X_2 = cal_sigma(X_2)

        # X_3
        X_3 = X_train[third_index]
        mu_X_3 = cal_m(X_3)
        sigma_X_3 = cal_sigma(X_3)

        # 对test里的每一个X， 计算三类的概率， 概率最大的， 我们就判断他属于那一类
        # 计算错误率
        num_test = len(Y_test)
        count_acc = 0
        for i in range(num_test):
            x = X_test[i]
            y = Y_test[i]
            p_1 = cal_P(x,sigma_X_1,mu_X_1)
            p_2 = cal_P(x,sigma_X_2,mu_X_2)
            p_3 = cal_P(x,sigma_X_3,mu_X_3)
            predict_x = np.argmax([p_1, p_2, p_3]) + 1
            if predict_x == y:
                count_acc += 1
        err_rate_list.append(1 - count_acc/num_test)

    print("err rate =", np.mean(err_rate_list))






