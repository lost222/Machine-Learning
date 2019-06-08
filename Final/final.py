import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import hamming_loss
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score,average_precision_score


mat_contents = sio.loadmat('dataset/medical.mat')
# X for samples * attrs
X = mat_contents['data']
# X = np.transpose(X)
# Y for labels * samples
Y = mat_contents['target']
Y = np.transpose(Y)
# Y_ba = Y.copy()

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


loss_vec = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
penalty_vec = ['l1', 'l2', 'elasticnet']


for loo in loss_vec:
    for pe in penalty_vec:
        clf = OneVsRestClassifier(SGDClassifier(loss=loo, penalty=pe, max_iter=1000))
        # clf  = OneVsRestClassifier(SVC(kernel='linear')
        # 送进SGD
        print("")
        clf.fit(X_train, Y_train)
        Y_predict = clf.predict(X_test)
        pre1 = average_precision_score(Y_test, Y_predict, average='samples')
        pre2 = precision_score(Y_test, Y_predict, average='samples')
        # precision_vec.append(precision)
        states_str = "loss : " + loo + "\npenalty : " + pe
        log_str = "average_precision_score = " + str(pre1) + "\nprecision_score = " + str(pre2)
        print(states_str)
        print(log_str)
        print("")



