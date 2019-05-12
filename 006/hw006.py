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
            if len(d) > 6:
                re.append(float(d[5]))
            re.append(lable_list.index(d[-1]))
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


def Gini(D):
    # return Gini
    s = []
    for k in set(D):
        p_k = np.sum(np.where(D == k, 1, 0)) / np.shape(D)[0]
        s.append(p_k)
    s = np.array(s)
    return 1 - np.sum(s * s)


def gain(X, Y, attr, is_conti):
    # X, Y 是numpy arrary attr是某个特征的index
    x_attr_col = X[:, attr]
    ent_Dv = []
    weight_Dv = []
    # 离散值处理
    if not is_conti:
        for x_v in set(x_attr_col):
            index_x_equal_v = np.where(x_attr_col == x_v)
            y_x_equal_v = Y[index_x_equal_v]
            ent_Dv.append(ent(y_x_equal_v))
            weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])
    # 连续值处理
    else:
        half = (np.max(x_attr_col) + np.min(x_attr_col)) / 2
        index_x_less_half = np.where(x_attr_col < half)
        y_x_less_half = Y[index_x_less_half]
        ent_Dv.append(ent(y_x_less_half))
        weight_Dv.append(np.shape(y_x_less_half)[0] / np.shape(Y)[0])

        index_x_ge_half = np.where(x_attr_col >= half)
        y_x_ge_half = Y[index_x_ge_half]
        ent_Dv.append(ent(y_x_ge_half))
        weight_Dv.append(np.shape(y_x_ge_half)[0] / np.shape(Y)[0])

    return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))




def gini_index(X, Y, attr):
    # return -gini index to use argmax
    x_attr_col = X[:, attr]
    gini_Dv = []
    weight_Dv = []
    for x_v in set(x_attr_col):
        D_x_equal_v = np.where(x_attr_col == x_v)
        y_x_equal_v = Y[D_x_equal_v]
        gini_Dv.append(Gini(y_x_equal_v))
        weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])

    return -np.sum(np.array(gini_Dv) * np.array(weight_Dv))


class Node:
    def __init__(self, attr, label, v):
        # label == pi是非叶节点
        # attr == pi 是叶节点
        self.attr = attr
        self.label = label
        self.attr_v = v
        self.children = []


def is_same_on_attr(X, attrs):
    X_a = X[:, attrs]
    target = X_a[0]
    for r in range(X_a.shape[0]):
        row = X_a[r]
        if (row != target).any():
            return False
    return True


def dicision_tree_init(X, Y, attrs, is_con_array, root, purity_cal):
    # 递归基
    if len(set(Y)) == 1:
        root.attr = np.pi
        root.label = Y[0]
        return None

    # @todo:
    # 什么是D中样本在A上取值相同
    if len(attrs) == 0 or is_same_on_attr(X, attrs):
        root.attr = np.pi
        # Y 中出现次数最多的label设定为node的label
        root.label = np.argmax(np.bincount(Y))
        return None

    # 计算每个attr的划分收益
    purity_attrs = []
    for i, a in enumerate(attrs):
        is_con = is_con_array[i]
        p = purity_cal(X, Y, a, is_con)
        purity_attrs.append(p)

    chosen_index = purity_attrs.index(max(purity_attrs))
    chosen_attr = attrs[chosen_index]
    is_attr_conti = is_con_array[chosen_index]
    root.attr = chosen_attr
    root.label = np.pi
    print("chose", chosen_attr)

    del attrs[chosen_index]
    del is_con_array[chosen_index]

    x_attr_col = X[:, chosen_attr]
    # 离散数据处理
    if not is_attr_conti:
        for x_v in set(X[:, chosen_attr]):
            n = Node(-1, -1, x_v)
            root.children.append(n)
            # 不可能Dv empty 要是empty压根不会在set里
            # 选出 X[attr] == x_v的行

            index_x_equal_v = np.where(x_attr_col == x_v)
            X_x_equal_v = X[index_x_equal_v]
            Y_x_equal_v = Y[index_x_equal_v]
            dicision_tree_init(X_x_equal_v, Y_x_equal_v, attrs, is_con_array, n, purity_cal)
    else:
        half = (np.max(x_attr_col) + np.min(x_attr_col)) / 2
        n_l = Node(-1, -1, -np.inf)
        n_ge = Node(-1, -1, half)
        root.children.append(n_l)
        root.children.append(n_ge)

        index_x_less_half = np.where(x_attr_col < half)
        dicision_tree_init(X[index_x_less_half], Y[index_x_less_half], attrs, is_con_array, n_l, purity_cal)

        index_x_ge_half = np.where(x_attr_col >= half)
        dicision_tree_init(X[index_x_ge_half], Y[index_x_ge_half], attrs, is_con_array, n_ge, purity_cal)






    # @todo:
    # 假如某个选项在训练集上已经被另一个选项删除了P(AB) = 0 而且A选择在所有训练集数据上都更先发生， 那么现在的实现是没有B的分支的。


def dicision_tree_predict(x, tree_root, is_con_arry):
    if tree_root.label != np.pi:
        return tree_root.label

    # 决策
    if tree_root.label == np.pi and tree_root.attr == np.pi:
        print("err!")
        return None

    chose_attr = tree_root.attr
    is_attr_conti = is_con_arry[chose_attr]
    # 寻找自己应该进入哪个分支
    if not is_attr_conti:
        for child in tree_root.children:
            if child.attr_v == x[chose_attr]:
                return dicision_tree_predict(x, child, is_con_arry)
    else:
        attr_v_l = []
        for child in tree_root.children:
            attr_v_l.append(child.attr_v)
        attr_v_l = np.array(attr_v_l)
        child = None
        if np.sum(np.where(attr_v_l <= x[chose_attr], 1, 0)) == 1:
            child = tree_root.children[np.argmin(attr_v_l)]
        else:
            child = tree_root.children[np.argmax(attr_v_l)]
        return dicision_tree_predict(x, child, is_con_arry)





    # 因为构造的时候有点问题： 见todo， 有可能执行到这里， 这个时候应该报错
    print("err : need to fix bug in init")
    return None


if __name__ == '__main__':
    ans = load_txt("第六次实验要求/Watermelon-train2.csv")
    X_train = ans[:, 1: -1]
    Y_train = ans[:, -1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    test_data = load_txt("第六次实验要求/Watermelon-test2.csv")
    X_test = test_data[:, 1:-1]
    Y_test = test_data[:, -1]

    r = Node(-1, -1, -1)
    attrs = [0, 1, 2, 3, 4]
    is_contine = [False, False, False, False, True]
    dicision_tree_init(X_train, Y_train, attrs, is_contine, r, gain)

    y_predict = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        is_contine = [False, False, False, False, True]
        y_p = dicision_tree_predict(x, r, is_contine)
        y_predict.append(y_p)

    acc = accuracy_score(Y_test, y_predict)
    print(acc)
