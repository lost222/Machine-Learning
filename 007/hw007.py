import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss


def fsml(X_train, Y_train, alpha):
    """X is attrs * samples
    Y is samples * lables
    W is attrs * lables(has labled)"""
    (dim, n) = np.shape(X_train)
    label_num = np.shape(Y_train)[1]

    # 按照列加和
    labeled_id = np.nonzero(np.sum(Y_train, 1))
    labeled_num = np.shape(labeled_id[0])[0]
    W = np.random.rand(dim, label_num)

    iter = 1
    obji = 1

    H = np.eye(n) - 1 / n * np.ones((n, n))

    objective = []

    while 1:
        d = 0.5 / np.sqrt(np.sum(W * W, 1) + np.finfo(float).eps)
        D = np.diag(d)
        # W = (X_train * H * H * X_train' + 2 * alpha * D + eps) \ X_train * H * Y_train;
        W1 = (X_train.dot(H).dot(H).dot(np.transpose(X_train)) + 2 * alpha * D + np.finfo(float).eps)
        W2 = X_train.dot(H).dot(Y_train)
        W = np.linalg.inv(W1).dot(W2)
        # bt = 1/n * ( ones(n, 1)' * Y_train - ones(n, 1)' * X_train' * W);
        bt = 1 / n * (np.ones((1, n)).dot(Y_train) - np.ones((1, n)).dot(np.transpose(X_train)).dot(W))
        Ypred = np.transpose(X_train).dot(W) + np.ones((n, 1)).dot(bt)

        # # 手动保证有lable的在前没lable的在后
        for i in range(labeled_num, n):
            for j in range(0, label_num):
                if Ypred[i, j] <= 0:
                    Y_train[i, j] = 0
                elif Ypred[i, j] >= 1:
                    Y_train[i, j] = 1
                else:
                    Y_train[i, j] = Ypred[i, j]

        # print(Ypred[-1])
        # print(Y_train[-1])

        # (norm((X_train'*W + ones(n,1)*bt - Y_train), 'fro'))^2 + alpha * sum(sqrt(sum(W.*W,2)+eps));
        to_norm = np.transpose(X_train).dot(W) + np.ones((n, 1)).dot(bt) - Y_train
        temp = np.linalg.norm(to_norm, 'fro') ** 2 + alpha * np.sum(np.sqrt(np.sum(W * W, 1) + np.finfo(float).eps))

        # cver = abs((objective(iter) - obji)/obji)
        cver = np.abs((temp - obji) / obji)
        # obji = objective(iter);
        obji = temp
        objective.append(cver)

        iter = iter + 1

        # if (cver < 10^-3 && iter > 2) , break, end
        if cver < 0.001 and iter > 2:
            Y_train = np.uint8(Y_train + 0.5)
            break

    return W, bt, objective, Y_train


def pick_attrs(W, frac):
    select = []
    for i in range(W.shape[0]):
        m = W[i].reshape(1, -1)
        s = np.linalg.norm(m, 'fro')
        select.append(s)
    select_num = int(frac * W.shape[0])
    to_select_attr = np.argsort(select)[:select_num]
    return to_select_attr


def precision_recall_mul_lable(Y_true, Y_pre):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(Y_true.shape[0]):
        for j in range(Y_true.shape[1]):
            if Y_true[i][j] == 1 and Y_pre[i][j] == 1:
                TP += 1
            elif Y_true[i][j] == 1 and Y_pre[i][j] == 0:
                FN += 1
            elif Y_true[i][j] == 0 and Y_pre[i][j] == 1:
                FP += 1
            else:
                TN += 1

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    return precision, recall


def label_take(X, Y, rate):
    # 随机打乱 X 和 Y
    random_arry = np.random.choice(Y.shape[0], Y.shape[0], replace=False)
    X = X[random_arry]
    Y = Y[random_arry]

    # Y 的后部 rate 没标签
    no_label_num = int(Y.shape[0] * rate)

    pick_no_label = np.ones((Y.shape[0], 1))

    pick_no_label[Y.shape[0] - no_label_num:Y.shape[0], :] = np.zeros((no_label_num, 1))

    Y_s = np.uint8(Y * pick_no_label)

    return X, Y_s, Y


def fs_predict(W, bt, x):
    ans = x.dot(W) + bt
    ans = np.uint8(ans + 0.5)
    return ans


# if __name__ == '__main__':
mat_contents = sio.loadmat('07/emotions.mat')
# X for samples * attrs
X = mat_contents['data']
# X = np.transpose(X)
# Y for labels * samples
Y = mat_contents['target']
Y = np.transpose(Y)
# Y_ba = Y.copy()

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 拿走一半sample
rate = 0.1
X_train, Y_train_s, Y_train = label_take(X_train, Y_train, 1 - rate)

Y_train_s = np.asarray(Y_train_s, np.float)

# 在训练集中划分训练集和测试集

# X_1, X_2, Y_1, Y_2 = train_test_split(X_train, Y_train, test_size=0.2)

# 训练合适的 alpha

alpha_vec = [10e-6, 10e-4, 10e-2, 10e0, 10e2, 10e4, 10e6]

min_vec = []

alpha = 0.1
# X_1, Y_1 = label_take(X_1, Y_1, 0.5)

# 使用预测出来的Y_train
W, bt, obj, Y_train_s = fsml(np.transpose(X_train), Y_train_s, alpha)
precision_vec = []
ham_vec = []


Y_predict = fs_predict(W, bt, X_test)
ham = hamming_loss(Y_test, Y_predict)
print(ham)


# for i_ in range(1, 6):
#     pick_array = pick_attrs(W, i_ / 6)
#     X_train_picked = X_train[:, pick_array]
#     clf = OneVsRestClassifier(SVC(kernel='linear'))
#
#     # # 送进SVM
#     # clf.fit(X_train_picked, Y_train)
#     # Y_predict = clf.predict(X_test[:, pick_array])
#     # # precision, recall = precision_recall_mul_lable(Y_test, Y_predict)
#     # ham = hamming_loss(Y_test, Y_predict)
#     # # precision_vec.append(precision)
#     # ham_vec.append(ham)
#
#
# min_h = min(ham_vec)
# min_vec.append(min_h)
#
# map_x = list(range(1, 6))
# map_x = np.array(map_x) / 6 * X.shape[0]
#
# plt.clf()
# # plt.plot(map_x, precision_vec, '-o')
# plt.plot(map_x, ham_vec, '-d')
# plt.xlabel("choose attrs")
# plt.ylabel("hamming_loss")
# plt.title("label "+str(rate*100)+"% alpha = " + str(alpha))
# figname = "label "+str(rate*100)+"% alpha = " + str(alpha) + " hamming_loss.png"
# # plt.savefig(figname)
# print(min_h)
# plt.show()

