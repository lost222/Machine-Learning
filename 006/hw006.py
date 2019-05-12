import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import codecs
from sklearn.metrics import accuracy_score
import graphviz

feature_dict = {"色泽": ["青绿", "乌黑", "浅白"],
                "根蒂": ["蜷缩", "稍蜷", "硬挺"],
                "敲声": ["浊响", "沉闷", "清脆"],
                "纹理": ["清晰", "稍糊", "模糊"]
                }

lable_list = ["否", "是"]
feature_list = ["色泽", "根蒂", "敲声", "纹理"]


def load_txt(path):
    ans = []
    with codecs.open(path, "r", "GBK") as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.rstrip("\n").split(',')
            print(d)
            re = []
            # 输入编号方便追踪
            re.append(int(d[0]))
            re.append(feature_dict.get("色泽").index(d[1]))
            re.append(feature_dict.get("根蒂").index(d[2]))
            re.append(feature_dict.get("敲声").index(d[3]))
            re.append(feature_dict.get("纹理").index(d[4]))
            re.append(lable_list.index(d[5]))
            ans.append(np.array(re))
            line = f.readline()
    return np.array(ans)


def ent(D):
    # D is a 1d np array which actually is Y
    s = 0
    for k in set(D):
        p_k = np.sum(np.where(D == k, 1, 0)) / np.shape(D)[0]
        if p_k == 0:
            # 此时Pklog2Pk 定义为 0
            continue
        s += p_k * np.log2(p_k)
    return -s


def gain(X, Y, attr):
    # X, Y 是numpy arrary attr是某个特征的index
    x_attr_col = X[:, attr]
    ent_Dv = []
    weight_Dv = []
    for x_v in set(x_attr_col):
        D_x_equal_v = np.where(x_attr_col == x_v)
        y_x_equal_v = Y[D_x_equal_v]
        ent_Dv.append(ent(y_x_equal_v))
        weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])

    return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))


class Node:
    def __init__(self, attr, label):
        # label == nan是非叶节点
        # attr == nan 是叶节点
        self.attr = attr
        self.label = label
        self.children = []

def dicision_tree(X, Y, attrs, root, purity_cal):
    # 递归基
    if len(set(Y)) == 1:
        root.attr = np.nan
        root.label = Y[0]
        return

    # @todo:
    # 什么是D中样本在A上取值相同
    if len(attrs) == 0 or len(set(Y[attrs])) == 1:
        root.attr = np.nan
        # Y 中出现次数最多的label设定为node的label
        root.label = np.argmax(np.bincount(Y))

    # 计算每个attr的划分收益
    purity_attrs = []
    for a in attrs:
        p = purity_cal(X, Y, a)
        purity_attrs.append(p)

    chosen_index = purity_attrs.index(max(purity_attrs))
    chosen_attr = attrs[chosen_index]
    print("chose", chosen_attr)
    del attrs[chosen_index]
    x_attr_col = X[:, chosen_attr]
    for x_v in set(X[:, chosen_attr]):
        n = Node(-1, -1)
        root.children.append(n)
        # 不可能Dv empty 要是empty压根不会在set里
        # 选出 X[attr] == x_v的行

        index_x_equal_v = np.where(x_attr_col == x_v)
        X_x_equal_v = X[index_x_equal_v]
        Y_x_equal_v = Y[index_x_equal_v]

        dicision_tree(X_x_equal_v, Y_x_equal_v, attrs, n, purity_cal)


    # @todo:
    # 假如某个选项在训练集上已经被另一个选项删除了P(AB) = 0 而且A选择在所有训练集数据上都更先发生， 那么现在的实现是没有B的分支的。






if __name__ == '__main__':
    ans = load_txt("第六次实验要求/Watermelon-train1.csv")
    X_train = ans[:, 1: -1]
    Y_train = ans[:, -1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    test_data = load_txt("第六次实验要求/Watermelon-test1.csv")
    X_test = test_data[:, 1:-1]
    Y_test = test_data[:, -1]

    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")

    r = Node(-1, -1)
    attrs = [0, 1, 2, 3]
    dicision_tree(X_train, Y_train, attrs, r, gain)