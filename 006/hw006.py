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


def load_txt(path):
    ans = []
    with codecs.open(path, "r", "GBK") as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.rstrip("\n").split(',')
            print(d)
            re = []
            re.append(feature_dict.get("色泽").index(d[1]))
            re.append(feature_dict.get("根蒂").index(d[2]))
            re.append(feature_dict.get("敲声").index(d[3]))
            re.append(feature_dict.get("纹理").index(d[4]))
            re.append(lable_list.index(d[5]))
            ans.append(np.array(re))
            line = f.readline()
    return np.array(ans)


if __name__ == '__main__':
    ans = load_txt("第六次实验要求/Watermelon-train1.csv")
    X_train = ans[:, : -1]
    Y_train = ans[:, -1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    test_data = load_txt("第六次实验要求/Watermelon-test1.csv")
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]

    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")






