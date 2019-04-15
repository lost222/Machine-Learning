# 机器学习实验1 — KNN分类器

## 基本要求

### 实验结果

![compare_train_test](/Users/acat/PycharmProjects/ML/001/ML/compare_train_test.png)

如图， 我的KNN分类器的错误值为

| K      | 1     | 3     | 5     |
| ------ | ----- | ----- | ----- |
| 错误率 | 14.4% | 16.1% | 16.7% |

由图可以看出， 当K在1到9内变化， 我的KNN分类器的错误率和sklearn机器学习库的KNN分类器的错误率基本接近， 相差没有超过百分之一。 我的KNN分类器的正确性得到了证明



## 中级要求

d

划分一个交叉验证集， 利用交叉验证选取k值

### 实验结果

![](/Users/acat/PycharmProjects/ML/001/ML/compare_X_train.png)

k

错误率最低的时候是k = 7的时候， 我的模型这时错误率仅有 7.5%。可以看到我的模型和sklearn模型的错误率差别一直不大



### 交叉验证





## 高级要求

我探索了距离加权 和 不同距离计算方法对KNN分类的影响。 



### 距离加权

距离加权基于这样一个想法， 选出距离待预测点最近的k个点，这样如果一个数据集里某类点数量比较多， 那么他们就在投票中占优势。 改进的方法是让距离待预测点越近的点越重要。 具体而言， 每个点投票的时候， 计算的不再是一个点， 而是 $\frac{1}{dis}$ 。 

![](/Users/acat/PycharmProjects/ML/001/ML/compare_weight_train_text.png)

可以看见， 距离加权后错误率显著下降。 这最明显发生在K=2时， 分析原因， 因为K=2时，在离待预测点最近的两个点中随机选一个， 正确率肯定不如直接选取最近的那一个。可以看见K为偶数的时候， 错误率下降得都比较大

### 不同距离计算方法 曼哈顿距离与欧式距离比较

距离计算方法可以不同。 这里比较了两种距离公式的比较

没有距离加权

![](/Users/acat/PycharmProjects/ML/001/ML/compare_dis_cal_train_text.png)

距离加权

![](/Users/acat/PycharmProjects/ML/001/ML/compare_dis_cal_train_text_dis.png)

结论是区别不大



## 实验理论

### KNN分类器

###机器学习实验流程

### 实验现象分析和KNN参数





## 代码分析

### 模型核心代码

基本要求下的KNN预测模型比较简单。 我的实现核心代码一共没有超过十行

#### 获取K个邻居

整个函数基于核心数据结构`numpy.array` , 由于我已经比较熟悉`numpy`接口， 代码力求简洁

~~~python
def find_k_nearest(x_point, x_train, k):
    ":return K args for nearest"
    # 对训练集里的每一个数据点， 都计算它和当前点的距离 
    dis_vec = [distance.euclidean(x_point,  x_train[it]) for it in range(x_train.shape[0])]
    # 获得距离从小到大的排序
    arr = np.argsort(dis_vec)
    # 返回前k个
    return arr[: k]
~~~

需要特别提及的是`np.argsort`方法。该方法返回排序后下标的值， 简单而言， 有

~~~python
a = np.argsort(array)
b = np.sort(array)
print(b == array[a])

==>true
~~~

#### 投票获得类别

~~~python
def knn_predict(y_train, k_point_arr):
    # k_point_arr 是k个距离最近的点的array下标
    # near_class  是这k个点的class
    near_class = y_train[k_point_arr]
    # scipy 函数， 获得array众数，即是需要预测点的坐标
    return stats.mode(near_class)[0][0]
~~~

值得一提的同样是`stats.mode`， 在有多个众数的时候我取到的是列表里第一个。



### 数据的输入和预处理

数据集和测试集的数据格式一致， 前256位是 16 $$\times$$ 16的手写字迹识别图片向量。 值得一提的是由于向量的值只有0和1两种取值， 所以不需要归一化数据（不同于一般的灰度图片处理）。后10位代表数字的类别， 分别从0到9

### 读取

~~~python
dataMatrix = []

with open("semeion_train.csv") as file:
    for line in file:
        point = [float(x) for x in line.split()]
        dataMatrix.append(point)

dataMatrix = np.array(dataMatrix)

# 前 256 维是特征向量
X = dataMatrix[:, :16*16]
# 后 10 维是类别
Y = dataMatrix[:, 16*16:]
~~~

X和Y就分别是训练集里的数据和标签（类别）了。获取X和Y的过程使用了`numpy.array`的高维切片。这些特性的确能让代码更简洁 

### 分类标签处理

Y的分类是由一个向量而非单个数字给出， 在这里换一下， 改成数字表示， 主要有两个目的

* 方便人阅读
* 符合sklearn库的标签输入。 

~~~python
newY = []
for line in Y:
    for i in range(len(line)):
        if line[i] > 0:
            newY.append(i)
            break

newY = np.array(newY)

Y = newY
~~~





### 预处理

一般来说， 在跑任何一个机器学习模型之前都需要归一化， 通常来说归一化能够提高性能，归一化的代码如下

```python
delta = np.max(X) - np.min(X)
X = X / delta
```

由于这次数据的形式特殊， 归一化做不做不对性能造成影响。 



### 正确率计算

要求输出的是错误率， 
$$
error \space rate = 1 - accuracy
$$


~~~python
def score_knn(x_train, y_train, x_test, y_test, k):
    truenum = 0
    for ii in range(x_test.shape[0]):
        # 对测试集里的每一个点
        x_point = x_test[ii]
        truevalue = y_test[ii]
        # 找到最近邻k个点
        arr = find_k_nearest(x_point, x_train, k)
        # 依据做出预测
        prediction = knn_predict(y_train, arr)
        # 记录预测正确数量
        if truevalue == prediction:
            truenum = truenum + 1
    return truenum / X_test.shape[0]
~~~

在基本要求里， 在给出的测试集上训练， 在给出的测试集上测试， 分别读入就可以



















































































































































































































































 