# Author:yifan
from numpy import *
import time
import matplotlib.pyplot as plt
import numpy as np

# euclDistance函数计算两个向量之间的欧氏距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))
    # initCentroids选取任意数据集中任意样本点作为初始均值点
    # dataSet: 数据集， k: 人为设定的聚类簇数目
    # centroids： 随机选取的初始均值点
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))  #k行dim的0矩阵
    for i in range(k):
        index = int(random.uniform(0, numSamples))   #从0到numSamples中随机选取一个数字
        centroids[i, :] = dataSet[index, :]           #随机选出k个数字，即为本函数的目的
        # print(centroids)
    return centroids
#定义每个点到其他点的距离和。步骤四的更新簇的均值点时会用到
def costsum( vector1,matrix1):
    sum = 0
    for i in range(matrix1.shape[0]):
        sum += euclDistance(matrix1[i,:], vector1)
    return sum
# kmediod: k-mediod聚类功能主函数
# 输入：dataSet-数据集，k-人为设定的聚类簇数目
# 输出：centroids-k个聚类簇的均值点，clusterAssment－聚类簇中的数据点
def kmediod(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples, 2)))
    # clusterAssment第一列存储当前点所在的簇
    # clusterAssment第二列存储点与质心点的距离
    clusterChanged = True   #用于遍历的标记
    ## 步骤一: 初始化均值点
    centroids = initCentroids(dataSet, k)
    while clusterChanged:
        clusterChanged = False
        ## 遍历每一个样本点
        for i in range(numSamples):
            minDist  = 100000.0    # minDist：最近距离，初始定一个较大的值
            minIndex = 0       # minIndex：最近的均值点编号
            ## 步骤二: 寻找最近的均值点
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])  #每个点和中心点的距离，共有k个值
                if distance < minDist:
                    #循环去找最小的那个
                    minDist  = distance
                    minIndex = j
            ## 步骤三: 更新所属簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2  #记录序号和点与质心点的距离
            ## 步骤四: 更新簇核心点
            for j in range(k):
                pointsInCluster = dataSet[nonzero(clusterAssment[:, 0] == j)[0]]  #当前属于j类的序号
                mincostsum = costsum(centroids[j,:],pointsInCluster)
                for point in range(pointsInCluster.shape[0]):
                    cost = costsum( pointsInCluster[point, :],pointsInCluster)
                    if cost < mincostsum:
                        mincostsum = cost
                        centroids[j, :] = pointsInCluster[point, :]
        print ('Congratulations, cluster complete!')
        return centroids, clusterAssment

# showCluster利用pyplot绘图显示聚类结果（二维平面）
# 输入:dataSet-数据集，k-聚类簇数目，centroids-聚类簇的均值点，clusterAssment－聚类簇中数据点
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print ("Sorry, the dimension of your data is not 2!")
        return 1
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        return 1
    # 画出所有的样本点
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 标记簇的质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
    plt.show()

## step 1: 构造数据
matrix1=np.random.random((12,2))
matrix2=np.random.random((12,2))
matrix3=np.random.random((12,2))
matrix4=np.random.random((12,2))
for i in range(12):
    matrix2[i,0] = matrix2[i,0]+2
    matrix3[i,1] = matrix3[i,1]+2
    matrix4[i,:] = matrix4[i,:]+2
dataSet = np.vstack((matrix1,matrix2,matrix3,matrix4))

dataSet1 = [[1.658985, 4.285136], [-3.453687, 3.424321], [4.838138, -1.151539], [-5.379713, -3.362104], [0.972564, 2.924086], [-3.567919, 1.531611], [0.450614, -3.302219], [-3.487105, -1.724432], [2.668759, 1.594842], [-3.156485, 3.191137], [3.165506, -3.999838], [-2.786837, -3.099354], [4.208187, 2.984927], [-2.123337, 2.943366], [0.704199, -0.479481], [-0.39237, -3.963704], [2.831667, 1.574018], [-0.790153, 3.343144], [2.943496, -3.357075], [-3.195883, -2.283926], [2.336445, 2.875106], [-1.786345, 2.554248], [2.190101, -1.90602], [-3.403367, -2.778288], [1.778124, 3.880832], [-1.688346, 2.230267], [2.592976, -2.054368], [-4.007257, -3.207066], [2.257734, 3.387564], [-2.679011, 0.785119], [0.939512, -4.023563], [-3.674424, -2.261084], [2.046259, 2.735279], [-3.18947, 1.780269], [4.372646, -0.822248], [-2.579316, -3.497576], [1.889034, 5.1904], [-0.798747, 2.185588], [2.83652, -2.658556], [-3.837877, -3.253815], [2.096701, 3.886007], [-2.709034, 2.923887], [3.367037, -3.184789], [-2.121479, -4.232586], [2.329546, 3.179764], [-3.284816, 3.273099], [3.091414, -3.815232], [-3.762093, -2.432191], [3.542056, 2.778832], [-1.736822, 4.241041], [2.127073, -2.98368], [-4.323818, -3.938116], [3.792121, 5.135768], [-4.786473, 3.358547], [2.624081, -3.260715], [-4.009299, -2.978115], [2.493525, 1.96371], [-2.513661, 2.642162], [1.864375, -3.176309], [-3.171184, -3.572452], [2.89422, 2.489128], [-2.562539, 2.884438], [3.491078, -3.947487], [-2.565729, -2.012114], [3.332948, 3.983102], [-1.616805, 3.573188], [2.280615, -2.559444], [-2.651229, -3.103198], [2.321395, 3.154987], [-1.685703, 2.939697], [3.031012, -3.620252], [-4.599622, -2.185829], [4.196223, 1.126677], [-2.133863, 3.093686], [4.668892, -2.562705], [-2.793241, -2.149706], [2.884105, 3.043438], [-2.967647, 2.848696], [4.479332, -1.764772], [-4.905566, -2.91107]]

## step 2: 开始聚类...
dataSet1 = mat(dataSet1)

print(103, dataSet, dataSet1)
k = 4
centroids, clusterAssment = kmediod(dataSet1, k)
## step 3: 显示聚类结果
showCluster(dataSet1, k, centroids, clusterAssment)