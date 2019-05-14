import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import codecs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


def gain_continue(X, Y, attr, t):
    # get attr when attr is continue and t divide X
    x_attr_col = X[:, attr]
    ent_Dv = []
    weight_Dv = []

    index_x_less_t = np.where(x_attr_col < t)
    y_x_less_t = Y[index_x_less_t]
    ent_Dv.append(ent(y_x_less_t))
    weight_Dv.append(np.shape(y_x_less_t)[0] / np.shape(Y)[0])

    index_x_ge_t = np.where(x_attr_col >= t)
    y_x_ge_t = Y[index_x_ge_t]
    ent_Dv.append(ent(y_x_ge_t))
    weight_Dv.append(np.shape(y_x_ge_t)[0] / np.shape(Y)[0])
    return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))

def gain_cal_t(X, Y, attr):
    x_attr_col = X[:, attr]
    G = []
    T = []
    for i in range(len(x_attr_col) - 1):
        t = (x_attr_col[i] + x_attr_col[i+1]) / 2
        ga = gain_continue(X, Y, attr, t)
        T.append(t)
        G.append(ga)
    best_t_index = np.argmax(G)
    return T[best_t_index], G[best_t_index]







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
        return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))
    # 连续值处理
    else:
        return gain_cal_t(X, Y, attr)[1]






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
        half = gain_cal_t(X, Y, chosen_attr)[0]
        n_l = Node(-1, -1, -np.inf)
        n_ge = Node(-1, -1, half)
        root.children.append(n_l)
        root.children.append(n_ge)

        index_x_less_half = np.where(x_attr_col < half)
        dicision_tree_init(X[index_x_less_half], Y[index_x_less_half], attrs, is_con_array, n_l, purity_cal)

        index_x_ge_half = np.where(x_attr_col >= half)
        dicision_tree_init(X[index_x_ge_half], Y[index_x_ge_half], attrs, is_con_array, n_ge, purity_cal)

def cal_label(Y):
    count1 = np.sum(Y)
    count0 = Y.shape[0] - count1
    if count0 < count1:
        return 1
    else:
        return 0


def dicision_tree_init_pre_pru(X, Y, X_test, Y_test, attrs, is_con_array, root, real_root , purity_cal):
    # 递归基
    Y.astype(np.int64)
    if len(set(Y)) == 1:
        root.attr = np.pi
        root.label = Y[0]
        return None


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
    print("chose", chosen_attr)


    # 计算不展开的验证集精确度
    root.label = cal_label(Y)
    root.attr = np.pi
    y_predict = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        y_p = dicision_tree_predict(x, real_root, [False, False, False, False, True])
        y_predict.append(y_p)

    acc_no = accuracy_score(Y_test, y_predict)

    # 计算展开的验证集收益
    root.attr = chosen_attr
    root.label = np.pi
    x_attr_col = X[:, chosen_attr]

    if not is_attr_conti:
        for x_v in set(X[:, chosen_attr]):
            n = Node(-1, -1, x_v)
            # 不可能Dv empty 要是empty压根不会在set里
            # 选出 X[attr] == x_v的行
            index_x_equal_v = np.where(x_attr_col == x_v)
            X_x_equal_v = X[index_x_equal_v]
            Y_x_equal_v = Y[index_x_equal_v]
            n.attr = np.pi
            n.label = cal_label(Y_x_equal_v)
            root.children.append(n)
    else:
        half = gain_cal_t(X, Y, chosen_attr)[0]
        n_l = Node(-1, -1, -np.inf)
        n_ge = Node(-1, -1, half)


        index_x_less_half = np.where(x_attr_col < half)
        n_l.attr = np.pi
        n_l.label = cal_label(Y[index_x_less_half])

        index_x_ge_half = np.where(x_attr_col >= half)
        n_ge.attr = np.pi
        n_ge.label = cal_label(Y[index_x_ge_half])

        root.children.append(n_l)
        root.children.append(n_ge)

    y_predict_yes = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        y_p = dicision_tree_predict(x, real_root, [False, False, False, False, True])
        y_predict_yes.append(y_p)

    acc_yes = accuracy_score(Y_test, y_predict_yes)

    print("acc of expand=", acc_yes)
    print("acc of not expand=", acc_no)
    # 不展开
    if acc_yes < acc_no :
        print("do not expand")
        root.label = np.argmax(np.bincount(Y))
        root.attr = np.pi
        root.children.clear()

    # 展开
    else:

        root.attr = chosen_attr
        root.label = np.pi
        print("expand")

        del attrs[chosen_index]
        del is_con_array[chosen_index]


        # 离散数据处理
        if not is_attr_conti:
            for n in root.children:
                x_v = n.attr_v
                # 不可能Dv empty 要是empty压根不会在set里
                # 选出 X[attr] == x_v的行
                index_x_equal_v = np.where(x_attr_col == x_v)
                X_x_equal_v = X[index_x_equal_v]
                Y_x_equal_v = Y[index_x_equal_v]
                dicision_tree_init_pre_pru(X_x_equal_v, Y_x_equal_v, X_test, Y_test, attrs, is_con_array, n, real_root, purity_cal)
        else:
            half = gain_cal_t(X, Y, chosen_attr)[0]
            n_l = root.children[0]
            n_ge = root.children[1]

            index_x_less_half = np.where(x_attr_col < half)
            dicision_tree_init_pre_pru(X[index_x_less_half], Y[index_x_less_half], X_test, Y_test, attrs, is_con_array, n_l, real_root, purity_cal)

            index_x_ge_half = np.where(x_attr_col >= half)
            dicision_tree_init_pre_pru(X[index_x_ge_half], Y[index_x_ge_half], X_test, Y_test, attrs, is_con_array, n_ge , real_root, purity_cal)


def my_splite(X, Y, test_index_a):
    index_test = np.array(test_index_a) - 1
    index_all = range(X.shape[0])
    index_train = np.array(list(set(index_all) - set(index_test)))

    x_train = X[index_train]
    y_train = Y[index_train]
    x_test = X[index_test]
    y_test = Y[index_test]
    return x_train, x_test, y_train, y_test


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
    Y_train.astype(np.int64)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    test_data = load_txt("第六次实验要求/Watermelon-test2.csv")
    X_test = test_data[:, 1:-1]
    Y_test = test_data[:, -1]

    r = Node(-1, -1, -1)
    attrs = [0, 1, 2, 3, 4]
    is_contine = [False, False, False, False, True]

    test_index = [1, 3, 5, 7]

    x_tra, x_te, y_tra, y_te = my_splite(X_train, Y_train, test_index)
    # dicision_tree_init(X_train, Y_train, attrs, is_contine, r, gain)
    dicision_tree_init_pre_pru(x_tra, y_tra, x_te, y_te, attrs, is_contine, r, r, gain)

    y_predict = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        is_contine = [False, False, False, False, True]
        y_p = dicision_tree_predict(x, r, is_contine)
        y_predict.append(y_p)

    acc = accuracy_score(Y_test, y_predict)
    print(acc)
