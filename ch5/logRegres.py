# -*- coding: utf-8 -*-


'''

适用情景：
	LR同样是很多分类算法的基础组件，它的好处是输出值自然地落在0到1之间，并且有概率意义。
	因为它本质上是一个线性的分类器，所以处理不好特征之间相关的情况。
	虽然效果一般，却胜在模型清晰，背后的概率学经得住推敲。
	它拟合出来的参数就代表了每一个特征(feature)对结果的影响,也是一个理解数据的好工具。

LR 基于概率的函数的，求解参数使用最大似然估计
'''
from numpy import *
import matplotlib.pyplot as plt
from functools import wraps
import time


# 计算耗时装饰器
def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer


def loadDataSet():
	dataMat = []
	labelMat = []
	
	with open('testSet.txt') as f:
		for line in f.readlines():
			lineArr = line.strip().split('\t')
			dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
			labelMat.append(int(lineArr[2]))
			
	return dataMat, labelMat


def sigmoid(inX):
	return 1.0 / (1+exp(-inX))


@func_timer
def gradAscent(dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	
	m,n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n, 1))
	
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat-h)
		## 这里的公式需要进行一定的推导获得
		weights = weights + alpha * dataMatrix.transpose() * error
	
	return weights

dataMat, labelMat = loadDataSet()
wei = gradAscent(dataMat, labelMat)


## 画出二分图
def plotBestFit(wei):
	
	weights = wei.getA()
	
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if labelMat[i] == 1:
			xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
		
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0]-weights[1]*x) / weights[2]
	
	ax.plot(x, y)
	plt.xlabel('X1'); plt.ylabel('X2');
	plt.show()
	
	
plotBestFit(wei)


# 上面梯度上升算法在每一次更新回归系数的时候都需要遍历整个数据集，计算复杂度很高 （批处理batch）
# 随机梯度上升算法是一次仅用一个样本点来更新回归系数，增量式更新，是一个在线学习算法

@func_timer
def stocGradAscent(dataMatrix, classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	
	y1=[]
	y2=[]
	y3=[]
	
	# 训练m次即可，因为m个样本
	for k in range(m):
		h = sigmoid(sum(dataMatrix[k]*weights))
		error = classLabels[k]-h
		## 这里的公式需要进行一定的推导获得
		weights = weights + alpha * error * array(dataMatrix[k]) 
		
		y1.append(weights[0])
		y2.append(weights[1])
		y3.append(weights[2])
		
	# 观察参数变化趋势
	x = range(len(y1))
	plt.plot(x, y1, color='r')
	plt.plot(x, y2, color='b')
	plt.plot(x, y3, color='g')
	plt.show()
	
	return mat(weights).transpose()

wei1 = stocGradAscent(dataMat, labelMat)
plotBestFit(wei1)


@func_timer
def stocGradAscent1(dataMatrix, classLabels, numberIter=150):
	m,n = shape(dataMatrix)
	weights = ones(n)
	
	for j in range(numberIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+i+j) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex]-h
			weights = weights + alpha * error * array(dataMatrix[randIndex])
			del(dataIndex[randIndex])
			
	return mat(weights).transpose()

wei2 = stocGradAscent1(dataMat, labelMat)
plotBestFit(wei2)


'''
处理缺失值的方法：
	1.使用可用特征的均值来填补缺失值
	2.使用特殊值填补缺失值
	3.忽略有缺失值的样本
	4.使用相似样本的均值填补缺失值
	5.使用另外的机器学习算法预测缺失值
'''

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX*weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0
	
	
def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	
	trainingSet = []; trainingLabel = []
	for line in frTrain.readlines():
		currentLine = line.strip().split('\t')
		lineArr = []
		trainingSet.append([float(currentLine[i]) for i in range(21)])
		trainingLabel.append(float(currentLine[21]))
		
	trainWeights = stocGradAscent1(trainingSet, trainingLabel, 500)
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1
		currentLine = line.strip().split('\t')
		inX = [float(currentLine[i]) for i in range(21)]
		if int(classifyVector(inX, trainWeights)) != int(currentLine[21]):
			errorCount += 1
	
	errorRate = (float(errorCount)/numTestVec)
	print('The error rate of this test is: {}'.format(errorRate))
	
	return errorRate

# colicTest()


def multiTest():
	numTests = 10; errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
		
	print('afer {} iterations the average error rate is: {}'.format(numTests, errorSum/float(numTests)))
	
multiTest()