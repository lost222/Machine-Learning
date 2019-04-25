import numpy as np
# import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances



class my_cluster:
    distence_matrix = np.array([[]])

    def __init__(self, dis_mat):
        self.point_list = []
        self.distence_matrix = dis_mat

def single_linkage_distence(one_cluster, other_cluster):
    T = []
    for item_a in one_cluster.point_list:
        for item_b in other_cluster.point_list:
            T.append(one_cluster.distence_matrix[item_a, item_b])
    return np.min(T)


def complete_linkage_distence(one_cluster, other_cluster):
    T = []
    for item_a in one_cluster.point_list:
        for item_b in other_cluster.point_list:
            T.append(one_cluster.distence_matrix[item_a, item_b])
    return np.max(T)


def average_linkage_distence(one_cluster, other_cluster):
    T = []
    for item_a in one_cluster.point_list:
        for item_b in other_cluster.point_list:
            T.append(one_cluster.distence_matrix[item_a, item_b])
    return np.mean(T)


def hierarchical_clustering(itemlist, item_dis_mat, dis_func, num):
    """item_dis_mat[i,j] is distence of item i and j"""

    clusterlist = []
    for item in itemlist:
        n_clu = my_cluster(item_dis_mat)
        n_clu.point_list.append(item)
        clusterlist.append(n_clu)
    # 类的成员函数， 共享
    # clusterlist[0].distence_matrix = item_dis_mat



    while len(clusterlist) > num:
        cluster_dis_mat = np.zeros((len(clusterlist), len(clusterlist)))
        for i in range(len(clusterlist)):
            for j in range(i, len(clusterlist)):
                cluster_dis_mat[i, j] = dis_func(clusterlist[i], clusterlist[j])
                cluster_dis_mat[j, i] = float('inf')

        all_clu_set = set(range(len(clusterlist)))
        used_clu_set = set([])

        merged_cluster = []
        while all_clu_set != used_clu_set:
            # 处理奇数个类
            if len(all_clu_set) % 2 == 1 and len(all_clu_set) - len(used_clu_set) == 1:
                l = list(all_clu_set - used_clu_set)[0]
                merged_cluster.append(clusterlist[l])
                break

            # 加速 只剩 两个 情况
            if len(all_clu_set) - len(used_clu_set) == 2:
                l = list(all_clu_set - used_clu_set)
                for item in clusterlist[l[0]].point_list:
                    clusterlist[l[1]].point_list.append(item)
                merged_cluster.append(clusterlist[l[1]])
                break

            dis_min = np.min(cluster_dis_mat)
            min_pla = np.where(cluster_dis_mat == dis_min)
            for i in range(len(min_pla[0])):
                row = min_pla[0][i]
                col = min_pla[1][i]
                cluster_dis_mat[row, col] = float('inf')
                if (row not in used_clu_set) and (col not in used_clu_set):
                    # 可以合并了
                    used_clu_set.add(row)
                    used_clu_set.add(col)
                    # 合并
                    for item in clusterlist[row].point_list:
                        clusterlist[col].point_list.append(item)
                    merged_cluster.append(clusterlist[col])
        # print("len of clusters = ", len(clusterlist))
        clusterlist = merged_cluster

    return clusterlist

def main(func):
    plt.clf()
    clusters = hierarchical_clustering(item_list, item_dis_mat, dis_func=func, num=4)
    for i, c in enumerate(clusters):
        x_p = []
        y_p = []
        for point in c.point_list:
            x = X[point][0]
            y = X[point][1]
            x_p.append(x)
            y_p.append(y)
        plt.plot(x_p, y_p, 'o', label="cluster" + str(i))
    plt.legend(loc="best")
    figname = "single_linkage_distence_cluster.png"
    plt.savefig(figname)
    real_clusters = []
    for i in set(labels):
        real_clusters.append([])
    for i in range(len(labels)):
        real_clusters[labels[i]].append(i)

    print("in "+str(func))
    for i in set(labels):
        real = set(real_clusters[i])
        recalls = []
        precistions = []
        for c in clusters:
            predict = set(c.point_list)
            r = len(real & predict) / len(real)
            p = len(real&predict) / len(predict)
            recalls.append(r)
            precistions.append(p)
        recall = max(recalls)
        precistion = max(precistions)
        t1 = recalls.index(recall)
        t2 = precistions.index(precistion)
        if t1 != t2:
            print("err match cluster")
        print("recall of cluster"+str(i)+"=", recall)
        print("pricision of cluster"+str(i)+"=", precistion)





if __name__ == '__main__':
    # read data -> item_list
    X = np.load("X.npy")
    labels = np.load("Y.npy")
    # cal item_dis_mat , different distance func
    item_dis_mat = pairwise_distances(X, metric="euclidean")
    item_list = [x for x in range(X.shape[0])]
    main(single_linkage_distence)
    main(complete_linkage_distence)
    main(average_linkage_distence)


