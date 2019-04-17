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
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 0)

    bandwidth = []
    err_rate = []
    for h in range(1, 200, 10):
        d = h / 100
        bandwidth.append(d)
        kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
        err_rate_list = []
        for train_index, test_index in kf.split(X):
            X_train = X[train_index]
            Y_train = Y[train_index]
            X_test = X[test_index]
            Y_test = Y[test_index]

            # 分三类分别计算 mu 和 sigma
            first_index = np.where(Y_train == 1, Y_train, 0).nonzero()
            sec_index = np.where(Y_train == 2, Y_train, 0).nonzero()
            third_index = np.where(Y_train == 3, Y_train, 0).nonzero()

            # X_1
            X_1 = X_train[first_index]
            pattern1 = kde.KernelDensity(kernel='gaussian', bandwidth=d).fit(X_1)


            # X_2
            X_2 = X_train[sec_index]
            pattern2 = kde.KernelDensity(kernel='gaussian', bandwidth=d).fit(X_2)

            # X_3
            X_3 = X_train[third_index]
            pattern3 = kde.KernelDensity(kernel='gaussian', bandwidth=d).fit(X_3)

            # 对test里的每一个X， 计算三类的概率， 概率最大的， 我们就判断他属于那一类
            # 计算错误率
            num_test = len(Y_test)
            count_acc = 0


            for i in range(num_test):
                x = X_test[i]
                y = Y_test[i]
                p_1 = pattern1.score_samples(np.mat(x))
                p_2 = pattern2.score_samples(np.mat(x))
                p_3 = pattern3.score_samples(np.mat(x))
                predict_x = np.argmax([p_1, p_2, p_3]) + 1
                if predict_x == y:
                    count_acc += 1
            err_rate_list.append(1 - count_acc / num_test)

        print("err rate =", np.mean(err_rate_list))
        err_rate.append(np.mean(err_rate_list))

    plt.style.use('ggplot')
    plt.plot(bandwidth, err_rate, 'o-')
    plt.xlabel("Bandwidth")
    plt.ylabel("Error Rate")
    plt.title("Kernel Density Estimation")
    plt.show()
    # plt.savefig("Kernel Density Estimation.png")






