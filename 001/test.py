import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance


def find_k_nearest(x_point, x_train, k):
    ":return K args for nearest"
    dis_vec = [distance.euclidean(x_point,  x_train[it]) for it in range(x_train.shape[0])]
    arr = np.argsort(dis_vec)
    return arr[: k]


X = [1, 2, 3, 4, 5]
Y = np.sin(X)
Z = np.cos(X)

# plt.style.use('ggplot')
plt.plot(X, Y, 'o-', label="sklearn")
plt.plot(X, Z, '+-', label="myKNN")
plt.xlabel("K")
plt.ylabel("accuracy")
plt.title("Compare My KNN  with sklearn")
plt.legend(loc='best')
# plt.savefig("compare.png"）
plt.show()

