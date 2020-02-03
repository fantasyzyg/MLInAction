# -*- coding: utf-8 -*-

### 决策树

### 数据准备：树构造算法只适用于标称型数据，因此数值型数据必须离散化。 ???

### 构造决策树需要解决的第一个问题就是：当前数据集在哪一个特征在划分数据分类时起到决定性作用。

### 信息增益：在划分数据集之前之后信息的发生的变化称为信息增益。

### 获取信息增益最高的特征就是最好的选择

### 熵：定义为信息的期望值   (entropy)
### 目的就是找到信息增益最大的那一个分类


### 本章使用的算法是ID3算法，一般拿来处理标称型数据集，匹配选项太多可能会产生过拟合
### 即overfitting，我们可以采取剪枝来减少这种过拟合，即合并信息增益很少的那些节点

from math import log
import operator

## 计算熵
## 熵越高则表明混合的数据越多
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
        
    return shannonEnt

def createDataSet():
    dataSet = [
            [1, 1, "yes"],
            [1, 1, "yes"],
            [1, 0, "no"],
            [0, 1, "no"],
            [0, 1, "no"]
            ]
    labels = ["no surfacing", "fippers"]
    return dataSet, labels

dataSet, labels = createDataSet()
print(calcShannonEnt(dataSet))


## 划分数据集, 根据第axis个特征对应特定的value值来划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            
            retDataSet.append(reduceFeatVec)
    
    return retDataSet


## print(splitDataSet(dataSet,0,1))
    
### 信息增益就是熵的减少或者数据无序度的减少
def chooseBestFeatureToSplit(dataSet):
    baseEntropy = calcShannonEnt(dataSet)   # 不选择任何特征时的熵
    numFeatures = len(dataSet[0])-1   # 特征数量
    bestInfoGain = 0.0
    bestFeature = -1
    
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature =  i
            
    return bestFeature
  
print(chooseBestFeatureToSplit(dataSet))


### 构造决策树
### 若只剩一个结果label列的话找最多的那一个label作为主要label
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0)+1
        
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    ## 是否已经是完全分好类了，即是说全部label类别都是一样了
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    ### 若只剩一个结果label列的话找最多的那一个label作为主要label
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    
    ## 删除该label
    del(labels[bestFeature])
    
    for value in uniqueVals:
        subLabels = labels[:]  ### 深复制
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature, value), subLabels)
        
    return myTree
    
tree = createTree(dataSet, labels[:])


### 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    classLabel = None
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

print(classify(tree, labels[:], [1, 0]))
print(classify(tree, labels[:], [1, 1]))


### 现在已经构造好分类器，我们可以将分类器存储起来


### 存储决策树
### 第二章的k-近邻算法就无法持久化分类器了。
def storeTree(inputTree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)
        f.close()
        
def grabTree(filename):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)
    
storeTree(tree, "classifierStrorage.txt")
newTree = grabTree("classifierStrorage.txt")
print(newTree)



#### 预测隐形眼镜类型
with open("lenses.txt") as f:
    lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]
    lensesTree = createTree(lenses, lensesLabels)
    
    print(lensesTree)