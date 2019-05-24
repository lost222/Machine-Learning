 import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy import stats
# Y 在是 [] 形式的时候天生不对



def find_k_nearest(x_point, x_train, k):
    ":return K args for nearest"
    dis_vec = [distance.euclidean(x_point,  x_train[it]) for it in range(x_train.shape[0])]
    arr = np.argsort(dis_vec)
    return arr[: k]


def knn_predict(y_train, k_point_arr):
    near_class = y_train[k_point_arr]
    return stats.mode(near_class)[0][0]


dataMatrix = []

with open("semeion_train.csv") as file:
    for line in file:
        point = [float(x) for x in line.split()]
        dataMatrix.append(point)

dataMatrix = np.array(dataMatrix)

X = dataMatrix[:, :16*16]
Y = dataMatrix[:, 16*16:]
# 改变Y的形式试一试
newY = []
for line in Y:
    for i in range(len(line)):
        if line[i] > 0:
            newY.append(i)
            break

newY = np.array(newY)

Y = newY

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# X_train = X
# Y_train = Y






X_arr = []
acc_arr = []
acc_my_arr = []


for k in range(1, 10):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    X_arr.append(k)
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

    print("acc in sklearn when k=" + str(k) + " is " + str(acc))
    print("acc in my model when k=" + str(k) + " is " + str(acc_my))

acc_arr = 1 - np.array(acc_arr)
acc_my_arr = 1 - np.array(acc_my_arr)

plt.style.use('ggplot')
plt.plot(X_arr, acc_arr, 'o-', label="sklearn")
plt.plot(X_arr, acc_my_arr, '+-', label="myKNN")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.title("Compare My KNN  with sklearn")
plt.legend(loc='best')
plt.savefig("compare_X_train.png")
# plt.show()
# trueNum = 0
# my acc calculate
# for i in range(X_test.shape[0]):
#     point = X_test[i]
#     trueValue = Y_test[i]
#     prediction = clf.predict([point])[0]
#     if trueValue == prediction:
#         trueNum = trueNum + 1
#
# print(trueNum / X_test.shape[0])

