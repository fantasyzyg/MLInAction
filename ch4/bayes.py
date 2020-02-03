# -*- coding: utf-8 -*-


'''
  朴素贝叶斯
    两个假设
    1.特征之间是相互独立的，这正是朴素的含义(naive)
    2.每一个特征之间是同等重要
    
    
决策的缺点：
    同时它也是相对容易被攻击的分类器[3]。
    这里的攻击是指人为的改变一些特征，使得分类器判断错误。
    常见于垃圾邮件躲避检测中。因为决策树最终在底层判断是基于单个条件的，攻击者往往只需要改变很少的特征就可以逃过监测。

朴素贝叶斯算法基于概率更好一点。条件概率和全概率公式。
'''

from numpy import *
import matplotlib.pyplot as plt

## 创建数据集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


### 根据数据集创建文档向量，该向量所有的单词都有了
def createVocabList(dataSet):
    vocabSet = set()
    for doucument in dataSet:
        vocabSet = vocabSet | set(doucument)
        
    return list(vocabSet)

postingList,classVec = loadDataSet()

## 单词向量
vocabSet = createVocabList(postingList)


### 根据输入文档向量转化为文档库的向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: {} is not in my vocabulary!".format(word))
            
    return returnVec

## print(setOfWords2Vec(vocabSet, postingList[0]))




### 朴素贝叶斯分类器训练函数
### 统计所有词汇在不同分类下的出现频率
### 对于每一个单词在所有文档出现的频次是一样的
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(len(trainCategory))  # 带侮辱词汇的文章
    
#    p0Num = zeros(numWords)
#    p1Num = zeros(numWords)
    ## 需要有初始值
    p0Num = ones(numWords)
    p1Num = ones(numWords)
#    p0Denom = 0.0
#    p1Denom = 0.0
    p0Denom = 2.0
    p1Denom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])   ## 这是统计了所有单词的
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])   ## 这是统计了所有单词的
    
    ## 防止过多小数相乘变为0
    p0Vect = log(p0Num/p0Denom)
    p1Vect = log(p1Num/p1Denom)
    return p0Vect, p1Vect, pAbusive

### 构造词汇矩阵
trainMatrix = []
for postinDoc in postingList:
    trainMatrix.append(setOfWords2Vec(vocabSet, postinDoc))
    
# print(trainMatrix)
    
p0Vect,p1Vect,pAbusive = trainNB0(trainMatrix, classVec)

## 绘制图形

def drawPicture(x, y1, y2):   
    plt.plot(x, y1, color="r", linestyle="-", marker="^", linewidth=1)
    plt.plot(x, y2, color="b", linestyle="-", marker="s", linewidth=1)
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.show()

### 可以看到不同分类下的出现的词汇概率差别还是很分明的
# drawPicture(range(len(p0Vect)), p0Vect, p1Vect)

def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify*p1Vect) + log(pClass1)
    p0 = sum(vec2Classify*p0Vect) + log(1-pClass1)
    
    # 如果p1==p0 return 0
    if p1 > p0:
        return 1
    else:
        return 0
    
#testEntry = ["love", "my", "dalmation"]
#print(classifyNB(setOfWords2Vec(vocabSet, testEntry),p0Vect,p1Vect,pAbusive))
#testEntry = ["hello", "world"]
#print(classifyNB(setOfWords2Vec(vocabSet, testEntry),p0Vect,p1Vect,pAbusive))
        
'''
将每个词出现与否作为一个特征，这可以被描述为词集模型(set of word model)
将每个词出现次数作为一个特征，这可以被描述为词集模型(bag of word model)
'''

# 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("The word: {} is not in my vocabulary!".format(word))
            
    return returnVec

# 使用朴素贝叶斯进行交叉验证
# 什么是交叉认证?    一部分数据作为训练集，另一部分作为测试集。留存交叉验证(hold-out cross validation)
# 文件解析以及完整的垃圾邮件测试函数
def textParse(text):
    import re
    listOfTokes = re.split(r'\W+', text)
    return [token.lower() for token in listOfTokes if len(token) > 2]

# print(textParse(open("email/ham/1.txt").read()))
    
def spamTest():
    docList = []
    classList = []
    fullText = []
    
    for i in range(1, 26):
        docList.append(textParse(open("email/ham/{}.txt".format(i), encoding='gb18030',errors='ignore').read()))
        classList.append(0)
        docList.append(textParse(open("email/spam/{}.txt".format(i), encoding='gb18030',errors='ignore').read()))
        classList.append(1)
        
    vocabList = createVocabList(docList)
    
    # 划分测试集和训练集
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        del(trainingSet[randIndex])
        testSet.append(randIndex)
        
    trainMat = []
    trainClass = []
    for index in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[index]))
        trainClass.append(classList[index])
        
    p0V, p1V, pSpam = trainNB0(trainMat, trainClass)
    
    errorCount = 0.0
    for index in testSet:
        if classifyNB(setOfWords2Vec(vocabList,docList[index]), p0V, p1V, pSpam) != classList[index]:
            print(docList[index])
            errorCount += 1
        
    print("The error rate is: {}".format(errorCount/len(testSet)))
    
spamTest()

# 使用朴素贝叶斯分类器从个人广告中获取区域倾向
# ......
# 对于分类而言，使用概率有时候要比使用硬规则更为有效。
