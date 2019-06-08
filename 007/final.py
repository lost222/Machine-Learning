import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import hamming_loss
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

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



clf = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l1", max_iter=5000))
# clf  = OneVsRestClassifier(SVC(kernel='linear'))
# 送进SGD
clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
# precision, recall = precision_recall_mul_lable(Y_test, Y_predict)
ham = hamming_loss(Y_test, Y_predict)
# precision_vec.append(precision)
print(ham)