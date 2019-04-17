import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq  ##引入最小二乘法算法

## normalize
def norm(x):
    x = (x - np.mean(x, 0)) / (np.max(x, 0) - np.min(x, 0))
    return x


## predict
def predict(alpha, beta, x):
    arr = alpha * x
    return np.sum(arr) + beta

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
        err = yline - predict(alpha, beta, xline)
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
    if len(x.shape) < 2:
        alpha = np.random.random_sample
    else:
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

def func(p,x):
    k, b=p
    return k*x+b


def error(p,x,y):
    return func(p, x)-y




if __name__ == "__main__":
    data = np.loadtxt("housing.data")
    X, Y = data[:, :-1], data[:, -1]

    # 留一法
    X = norm(X)

    max_index = 0
    max_r = 0
    for i in range(X.shape[1]):
        factor = X[:, i]
        r = abs(cal_r(factor, Y))
        if max_r < r:
            max_index = i
            max_r = r

    p0 = np.array([1, 20])
    X = X[:, max_index]
    # 把error函数中除了p0以外的参数打包到args中(使用要求)
    Para = leastsq(error, p0, args=(X, Y))
    k, b = Para[0]

    pre_y = [x*k+b for x in X]

    plt.plot(X, Y, "d")
    plt.plot(X, pre_y, '-')
    xla = "X" + str(max_index)
    plt.xlabel(xla)
    plt.ylabel("Y")
    plt.title("leave one factor")
    plt.savefig("fig5.png")
    #plt.show()