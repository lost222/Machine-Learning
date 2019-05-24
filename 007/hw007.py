import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



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

        # 手动保证有lable的在前没lable的在后
        for i in range(labeled_num, n):
            for j in range(0, label_num):
                if Ypred[i, j] <= 0:
                    Y_train[i, j] = 0
                else:
                    if Ypred[i, j] >= 1:
                        Y_train[i, j] = 1
                    else:
                        Y_train[i, j] = Ypred[i, j]

        # (norm((X_train'*W + ones(n,1)*bt - Y_train), 'fro'))^2 + alpha * sum(sqrt(sum(W.*W,2)+eps));
        to_norm = np.transpose(X_train).dot(W) + np.ones((n, 1)).dot(bt) - Y_train
        temp = np.linalg.norm(to_norm, 'fro') ** 2 + alpha * np.sum(np.sqrt(np.sum(W * W, 1) + np.finfo(float).eps))

        # cver = abs((objective(iter) - obji)/obji);
        cver = np.abs((temp - obji) / obji)
        # obji = objective(iter);
        obji = temp
        objective.append(cver)

        iter = iter + 1

        # if (cver < 10^-3 && iter > 2) , break, end
        if cver < 0.001 and iter > 2:
            break

    return W, bt, objective

def pick_attrs(W, frac):
    select = []
    for i in range(W.shape[0]):
        m = W[i].reshape(1, -1)
        s = np.linalg.norm(m, 'fro')
        select.append(s)
    select_num = int(frac * W.shape[0])
    to_select_attr = np.argsort(select)[:select_num]
    return to_select_attr

def precision_recall_mul_lable(Y_pre, Y_true):
    precision_vec = np.zeros(Y_true.shape[1])
    recall_vec = np.zeros(Y_true.shape[1])
    p_vec = np.zeros(Y_true.shape[1])
    for i in range(Y_true.shape[1]):
        precision, recall, _ = precision_recall_curve(Y_true[:, i], Y_pre[:, i])
        precision_vec[i] = precision
        recall_vec[i] = recall
        p_vec[i] = np.sum(Y_true[:, i]) / Y_true.shape[0]
    return np.sum(precision_vec*p_vec), np.sum(recall_vec*p_vec)




if __name__ == '__main__':
    mat_contents = sio.loadmat('07/emotions.mat')
    # X for samples * attrs
    X = mat_contents['data']
    # X = np.transpose(X)
    # Y for labels * samples
    Y = mat_contents['target']
    Y = np.transpose(Y)
    # Y_ba = Y.copy()

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # 在训练集中划分训练集和测试集

    X_1, X_2, Y_1, Y_2 = train_test_split(X_train, Y_train, test_size=0.2)

    # 训练合适的 alpha
    alpha = 0.01


    W, bt , obj = fsml(np.transpose(X_1), Y_1, alpha)

    pick_array = pick_attrs(W, 1/6)

    X_train_picked = X_train[:, pick_array]

    clf = OneVsRestClassifier(SVC(kernel='linear'))

    clf.fit(X_train_picked, Y_train)

    Y_predict = clf.predict(X_test[:, pick_array])

    precision_vec = np.zeros(Y_test.shape[1])
    recall_vec = np.zeros(Y_test.shape[1])
    p_vec = np.zeros(Y_test.shape[1])
    for i in range(Y_test.shape[1]):
        precision, recall, _ = precision_recall_curve(Y_test[:, i], Y_predict[:, i])
        precision_vec[i] = precision
        recall_vec[i] = recall
        p_vec[i] = np.sum(Y_test[:, i]) / Y_test.shape[0]

