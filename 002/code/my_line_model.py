import numpy as np
import matplotlib.pyplot as plt

## normalize
def norm(x):
    x = (x - np.mean(x, 0)) / (np.max(x, 0) - np.min(x, 0))
    return x


## predict
def predict(alpha, beta, x):
    arr = alpha * x
    return np.sum(arr) + beta

def predict_re(alpha, beta, x):
    arr = alpha * x
    return np.sum(arr) + beta + np.sum(alpha**2)

## Gradient descent
def gradient_descent(x, y, alpha, beta, learn_rate):
    # gradient_arr是整个 alpha 偏导数数组
    gradient_arr = np.zeros((1, x.shape[1]))
    gradient_beta = 0
    mean_s_err = 0
    for line in range(x.shape[0]):
        xline = x[line, :]
        yline = y[line]
        # err = y - (alpha X + beta)
        err = yline - predict(alpha, beta, xline)
        mean_s_err += err ** 2
        gradient_arr += err * xline
        gradient_beta += err

    # arr 是 alpha vector的梯度vec， 意思是 alpha0 是 arr[0]
    gradient_arr = gradient_arr * 2 / x.shape[0]
    gradient_beta = gradient_beta * 2 / x.shape[0]
    mean_s_err = mean_s_err / x.shape[0]

    alpha += np.reshape(gradient_arr, alpha.shape) * learn_rate
    beta += gradient_beta * learn_rate

    return alpha, beta, mean_s_err

def gradient_descent_random(x, y, alpha, beta, learn_rate):

    randomId = int(np.random.random_sample() * x.shape[0])
    x = x[randomId, :]
    y = y[randomId]
    gradient_arr = np.zeros(x.shape[0])
    gradient_beta = 0
    mean_s_err = 0


    err = y - predict(alpha, beta, x)
    mean_s_err += err ** 2
    gradient_arr += err * x
    gradient_beta += err

    # arr 是 alpha vector的梯度vec， 意思是 alpha0 是 arr[0]
    gradient_arr = gradient_arr * 2 / x.shape[0]
    gradient_beta = gradient_beta * 2 / x.shape[0]
    mean_s_err = mean_s_err / x.shape[0]

    alpha += np.reshape(gradient_arr, alpha.shape) * learn_rate
    beta += gradient_beta * learn_rate

    return alpha, beta, mean_s_err


# gradient_descent_regularized
def gradient_descent_regularized(x, y, alpha, beta, learn_rate, _lambda):
    # gradient_arr是整个 alpha 偏导数数组
    gradient_arr = np.zeros((1, x.shape[1]))
    gradient_beta = 0
    mean_s_err = 0
    for line in range(x.shape[0]):
        xline = x[line, :]
        yline = y[line]
        # err = y - (alpha X + beta)
        err = yline - predict_re(alpha, beta, xline)
        mean_s_err += err ** 2
        # 公式5 求和部分
        gradient_arr += err * xline
        gradient_beta += err

    # arr 是 alpha vector的梯度vec， 意思是 alpha0 是 arr[0]
    gradient_arr = gradient_arr * 2 / x.shape[0]

    # 加入正则化系数
    gradient_arr = gradient_arr - 2 * _lambda * alpha

    gradient_beta = gradient_beta * 2 / x.shape[0]
    mean_s_err = mean_s_err / x.shape[0]

    # 学习率 公式 7
    alpha += np.reshape(gradient_arr, alpha.shape) * learn_rate
    beta += gradient_beta * learn_rate

    return alpha, beta, mean_s_err


def train_model(x, y, learn_rate, loop_times):
    # random init alpha, beta
    alpha = np.random.random_sample(x.shape[1])
    beta = np.random.random_sample()


    err_vec = []
    for i in range(loop_times):
        alpha, beta, mean_s_err = gradient_descent_regularized(x, y, alpha, beta, learn_rate, 2)
        err_vec.append(mean_s_err)

    return alpha, beta

def cal_r(x, y):
    x1 = x - np.mean(x)
    y1 = y - np.mean(y)
    nume = np.mean(x1 * y1)
    return nume / (np.std(x, ddof=1) * np.std(y, ddof=1))




if __name__ == "__main__":
    data = np.loadtxt("housing.data")
    X, Y = data[:, :-1], data[:, -1]

    # 留一法
    X = norm(X)

    for learn_rate in [0.1, 0.2, 0.5, 0.7]:
        mean_s_err_vec = []
        x_loop = [i for i in range(5, 35, 10)]
        for loop in x_loop:
            mean_s_err = 0
            for i in range(X.shape[0]):
                X_test = X[i, :]
                Y_test = Y[i]
                X_train = np.delete(X, i, axis=0)
                Y_train = np.delete(Y, i)
                alpha, beta = train_model(X_train, Y_train, learn_rate, loop)
                err = Y_test - predict(alpha, beta, X_test)
                mean_s_err += err ** 2
            mean_s_err /= X.shape[0]
            mean_s_err_vec.append(np.sqrt(mean_s_err))

        plt.plot(x_loop, mean_s_err_vec, "-", label="learn_rate="+str(learn_rate))
        print("learn_rate="+str(learn_rate)+" done")
    plt.xlabel("loop time")
    plt.ylabel("RMSE")
    plt.title("RMSE of my liner model L2 regularized")
    plt.legend(loc="best")
    plt.savefig("fig4.png")
    #plt.show()