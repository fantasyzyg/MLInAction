# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 00:56:42 2020

@author: Fantasy
"""


from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([
            [1.0, 1.1],
            [1.0, 1.0],
            [0, 0],
            [0, 0.1]
            ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 0：代表列    1：代表行

### 实施KNN算法   inX 输入需要预测到的向量
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 矩阵行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()   ## 位置进行排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    ## print(classCount)
    '''
    sort 与 sorted 区别：
    sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
    list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，
    而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
    '''
    # print(classCount.items())
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    
    # print(type(sortedClassCount))
    return sortedClassCount[0][0]
    
    
group, labels = createDataSet()
### 测试预测结果
print(classify0([0, 0], group, labels, 3))
print(classify0([0, 1], group, labels, 3))
print(classify0([1, 1], group, labels, 3))

## 测试已知结果

print(classify0([1.0, 1.1], group, labels, 3))
print(classify0([1, 1], group, labels, 3))
print(classify0([0, 0.1], group, labels, 3))
print(classify0([0, 0], group, labels, 3))



#在约会网站上使用 KNN 算法
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    
    return returnMat, classLabelVector

# 1.每年获得飞行常客里程数
# 2.玩视频游戏所耗时间百分比
# 3.每周消费的冰淇淋公升数
datingDataMat, classLabelVector = file2matrix('D:\PySpyder\MLInAction\ch2\datingTestSet2.txt')

    
import matplotlib
import matplotlib.pyplot as plt

### 道理就是影响类别的特征组合可能产生十分不同的结果，有一些特征组合会产生十分明显的分类明显效果
fig = plt.figure()
ax = fig.add_subplot(111) 
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(classLabelVector), 15.0*array(classLabelVector))
plt.show()


## 数据归一化
## 有一些特征数据之间的差异十分大，但是我们希望这些特征都是同等重要的，我们需要做归一化处理
## 数值归一化通常就是将取值范围处理为[0, 1]或者[-1, -1]
## 公式为： newValue = (oldValue - min) / (max - min)

def autoNorm(dataSet):
    minVals = dataSet.min(0)   ## 0代表从列的维度出发计算  1代表从行维度出发计算
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    ## tile 函数复制矩阵
    normDataSet = dataSet - tile(minVals, (dataSet.shape[0], 1))
    normDataSet = normDataSet / tile(ranges, (dataSet.shape[0], 1))
    return normDataSet, ranges, minVals

normDataSet, ranges, minVals = autoNorm(datingDataMat)


### 测试该方法的一个预测效果
def datingClassTest(k=3):
    hoRatio = 0.10     # 测试比率
    datingDataMat, classLabelVector = file2matrix('D:\PySpyder\MLInAction\ch2\datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat) # 归一化
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)
    print(numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        ### k近邻前三最大出现次数
        classifierResult = classify0(normDataSet[i, :], normDataSet[numTestVecs:m, :], classLabelVector[numTestVecs:m], k)
        print("The classifier came back with: {}, the real answer is: {}".format(classifierResult, classLabelVector[i]))
        
        if (classifierResult != classLabelVector[i]):
            errorCount += 1
          
    ratio = errorCount / float(numTestVecs)
    print("The total error rate is: {}".format(ratio))
    
    return ratio
  
#### 测试不同的k不同的比率
#ratios = []
#for k in range(1, 21):
#    ratios.append(datingClassTest(k))
#    
#print(min(ratios))
#plt.figure()
#plt.plot(range(1, 21), ratios)
#plt.show()
    

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("Percentage of time spent playing video games?"))
    ffMiles = float(input("Frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    
    datingDataMat, classLabelVector = file2matrix('D:\PySpyder\MLInAction\ch2\datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat) # 归一化
    inX = (array([ffMiles,percentTats,iceCream])-minVals) / ranges
    classifierResult = classify0(inX, normDataSet, classLabelVector, 3)
    print("You will probably like this person: {}".format(resultList[classifierResult-1]))
    
    
    
### 手写识别系统
### 图像转化为矩阵信息保存
def img2vector(filename):
    returnVect = zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
                
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits/')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        filename = trainingFileList[i]
        trainingMat[i, :] = img2vector('digits/trainingDigits/'+filename)
        hwLabels.append(int((filename.split('.')[0]).split('_')[0]))
        
    testFileList = listdir('digits/testDigits/')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        filename = testFileList[i]
        inX = img2vector('digits/testDigits/'+filename)
        classNum = int((filename.split('.')[0]).split('_')[0])
        classifierResult = classify0(inX, trainingMat, hwLabels, 3)
        print("The classifier came back with: {}, the real answer is: {}".format(classifierResult, classNum))
        print(filename)
        if classifierResult != classNum:
            errorCount += 1.0
            
    ratio = errorCount / float(mTest)
    print("The total error rate is: {}".format(ratio))