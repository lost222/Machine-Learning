import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy import stats
from sklearn.model_selection import RepeatedKFold
# Y 在是 [] 形式的时候天生不对



def find_k_nearest(x_point, x_train, k):
    ":return K args for nearest"
    dis_vec = [distance.euclidean(x_point,  x_train[it]) for it in range(x_train.shape[0])]
    arr = np.argsort(dis_vec)
    return arr[: k]


def knn_predict(y_train, k_point_arr):
    near_class = y_train[k_point_arr]
    return stats.mode(near_class)[0][0]

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
    dataMatrix = []
    pa = "HWData3.csv"
    X, Y = read_csc(pa)

    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)

    err_rate_list = np.zeros(10)
    err_rate_list_sk = np.zeros(10)

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]



        acc_arr = []
        acc_my_arr = []


        for k in range(2, 12):
            clf = neighbors.KNeighborsClassifier(k)
            clf.fit(X_train, Y_train)
            acc = clf.score(X_test, Y_test)
            acc_arr.append(acc)

            trueNum = 0

            for i in range(X_test.shape[0]):
                point = X_test[i]
                trueValue = Y_test[i]
                # prediction = clf.predict([point])[0]
                arr = find_k_nearest(point, X_train, k)
                prediction = knn_predict(Y_train, arr)
                if trueValue == prediction:
                    trueNum = trueNum + 1
            acc_my = trueNum / X_test.shape[0]
            acc_my_arr.append(acc_my)

            # print("acc in sklearn when k=" + str(k) + " is " + str(acc))
            # print("acc in my model when k=" + str(k) + " is " + str(acc_my))

        acc_arr = 1 - np.array(acc_arr)
        acc_my_arr = 1 - np.array(acc_my_arr)

        err_rate_list = err_rate_list + acc_my_arr
        err_rate_list_sk = err_rate_list_sk + acc_arr

    err_rate_list /= 50
    err_rate_list_sk /=50

    X_arr = [i for i in range(2, 12)]

    print(err_rate_list_sk)
    print(err_rate_list)

    # plt.style.use('ggplot')
    # plt.plot(X_arr, err_rate_list_sk, 'o-', label="sklearn")
    # plt.plot(X_arr, err_rate_list, '+-', label="myKNN")
    # plt.xlabel("K")
    # plt.ylabel("Error Rate")
    # plt.title("Compare My KNN  with sklearn")
    # plt.legend(loc='best')
    # plt.savefig("compare_X_train.png")


