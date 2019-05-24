import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def fsml(X_train, Y_train, alpha):
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
        objective.append(temp)

        iter = iter + 1

        # if (cver < 10^-3 && iter > 2) , break, end
        if cver < 0.001 and iter > 2:
            break

    return W, bt, objective

if __name__ == '__main__':
    mat_contents = sio.loadmat('07/emotions.mat')
    X = mat_contents['data']
    X = np.transpose(X)
    Y = mat_contents['target']
    Y = np.transpose(Y)
    Y_ba = Y.copy()

    W, bt , obj = fsml(X, Y, 0.01)

    print(W.shape)
    select = []
    for i in range(W.shape[0]):
        m = W[i].reshape(1, -1)
        s = np.linalg.norm(m, 'fro')
        select.append(s)

    frac = 1/3
    select_num = int(frac * W.shape[0])
    to_select_attr = np.argsort(select)[:select_num]

    selected_X = np.transpose(X)[:, to_select_attr]



    print(to_select_attr)
