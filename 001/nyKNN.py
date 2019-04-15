import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from scipy import stats
import cv2
from sklearn import neighbors
from matplotlib import pyplot as plt
from scipy.spatial import distance

def find_k_nearest(x_point, x_train, k):
    ":return K args for nearest"
    dis_vec = [distance.euclidean(x_point,  x_train[it]) for it in range(x_train.shape[0])]
    arr = np.argsort(dis_vec)
    return arr[: k]


def knn_predict(y_train, k_point_arr):
    # k_point_arr 是k个距离最近的点的array下标
    # near_class  是这k个点的class
    near_class = y_train[k_point_arr]
    # scipy 函数， 获得array众数
    return stats.mode(near_class)[0][0]

# Y 在是 [] 形式的时候天生不对

def show_pic(pic_vec):
    pic_mat =np.uint8(pic_vec.reshape(16, 16))
    pic_white = np.uint8(np.ones(16*16).reshape(16, 16))
    print(pic_white)
    # emptyImage = np.zeros((400, 600), np.uint8)
    cv2.imwrite('messigray.png', pic_white)
    # cv2.imshow("EmptyImage", pic_mat)


def score_knn(x_train, y_train, x_test, y_test, k):
    truenum = 0
    for ii in range(x_test.shape[0]):
        # 对测试集里的每一个点
        x_point = x_test[ii]
        truevalue = y_test[ii]
        # 找到最近邻k个点
        arr = find_k_nearest(x_point, x_train, k)
        # 依据做出预测
        prediction = knn_predict(y_train, arr)
        # 记录预测正确数量
        if truevalue == prediction:
            truenum = truenum + 1
    return truenum / X_test.shape[0]


dataMatrix = []

with open("semeion_train.csv") as file:
    for line in file:
        point = [float(x) for x in line.split()]
        dataMatrix.append(point)

dataMatrix = np.array(dataMatrix)

# 前 256 维是特征向量
X = dataMatrix[:, :16*16]
# 后 10 维是类别
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

dataMatrixTest = []

with open("semeion_test.csv") as file:
    for line in file:
        point = [float(x) for x in line.split()]
        if len(point) != 266:
            print("err")
            exit(-1)
        dataMatrixTest.append(point)

dataMatrixTest = np.array(dataMatrixTest)
print(dataMatrixTest.shape)

X_test = dataMatrixTest[:, :16*16]
Y_test = dataMatrixTest[:, 16*16:]
# 改变Y的形式试一试
newY_test = []
for line in Y_test:
    for i in range(len(line)):
        if line[i] > 0:
            newY_test.append(i)
            break

newY_test = np.array(newY_test)
Y_test = newY_test

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train = X
Y_train = Y
X_test = X_test
Y_test = Y_test

# clf = neighbors.KNeighborsClassifier(5)
# clf.fit(X_train, Y_train)

# trueNum = 0
#
# for i in range(X_test.shape[0]):
#     point = X_test[i]
#     trueValue = Y_test[i]
#     # prediction = clf.predict([point])[0]
#     arr = find_k_nearest(point, X_train, 5)
#     prediction = knn_predict(Y_train, arr)
#     if trueValue == prediction:
#         trueNum = trueNum + 1
#
# print(trueNum / X_test.shape[0])

X_arr = []
acc_arr = []
acc_my_arr = []

for k in range(1, 10):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    X_arr.append(k)
    acc_arr.append(acc)
    acc_my = score_knn(X_train, Y_train, X_test, Y_test, k)
    acc_my_arr.append(acc_my)
    print("err in sklearn when k=" + str(k) + " is " + str(1 - acc))
    print("err in my model when k=" + str(k) + " is " + str(1 - acc_my))

acc_arr = 1 - np.array(acc_arr)
acc_my_arr = 1 - np.array(acc_my_arr)

plt.style.use('ggplot')
plt.plot(X_arr, acc_arr, 'o-', label="sklearn")
plt.plot(X_arr, acc_my_arr, '+-', label="myKNN")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.title("Compare My KNN  with sklearn")
plt.legend(loc='best')
plt.savefig("compare_train_test.png")
