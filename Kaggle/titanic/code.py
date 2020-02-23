# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:27:52 2020

@author: Fantasy
"""

import warnings
warnings.filterwarnings('ignore')  # 忽略警告
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='white',context='notebook',palette='muted')
import matplotlib.pyplot as plt

#导入数据
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#将实验数据和预测数据合并
full=train.append(test,ignore_index=True)

# Embarked 和  Embarked 之间的关系  登录港口
# 特征和结果的关系
sns.barplot(x='Embarked', y='Survived', data=train)

print('Embarked为"S"的乘客，其生存率为%.2f'%train['Survived'][train['Embarked']=='S'].value_counts(normalize=True)[1])
print('Embarked为"C"的乘客，其生存率为%.2f'%train['Survived'][train['Embarked']=='C'].value_counts(normalize=True)[1])
print('Embarked为"Q"的乘客，其生存率为%.2f'%train['Survived'][train['Embarked']=='Q'].value_counts(normalize=True)[1])

# 特征之间的数据统计
sns.factorplot('Pclass', col='Embarked', data=train, kind='count', size=3)

# Pclass 与  Survived 之间的关系
sns.barplot(x='Pclass', y='Survived', data=train)


# 离散型和数值型的处理方法不一样
#创建坐标轴  kde 随机变量的分布密度函数
ageFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
#作图，选择图形类型
ageFacet.map(sns.kdeplot,'Age',shade=True)
#其他信息：坐标轴范围、标签等
ageFacet.set(xlim=(0,train['Age'].max()))
ageFacet.add_legend()

# 票价小于 18 的生还率低很多
ageFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
#作图，选择图形类型
ageFacet.map(sns.kdeplot,'Fare',shade=True)
#其他信息：坐标轴范围、标签等
ageFacet.set(xlim=(0,150))
ageFacet.add_legend()

# 查看Fare分布
#farePlot = sns.distplot(full['Fare'][full['Fare'].notnull()], label='skewness:%.2f'%(full['Fare'].skew()))
#farePlot.legend(loc='best')

#对数化处理fare值
full['Fare']=full[full['Fare'].notnull()]['Fare'].map(lambda x: np.log(x) if x>0 else 0)

# 缺失值填补
full['Cabin'] = full['Cabin'].fillna('U')
full['Embarked'] = full['Embarked'].fillna('S')

#查看缺失数据情况，该乘客乘坐3等舱，登船港口为法国，舱位未知
full['Fare'] = full['Fare'].fillna(full[(full['Pclass'] == 3) & (full['Embarked'] == 'C') & (full['Cabin'] == 'U')]['Fare'].mean())

# 在理解原数据特征的基础上，特征工程通过对原有数据进行整合处理，得到新特征以反映数据更多信息。

# 构造新特征  乘客 Title
#将title信息进行整合
TitleDict={}
TitleDict['Mr']='Mr'
TitleDict['Mlle']='Miss'
TitleDict['Miss']='Miss'
TitleDict['Master']='Master'
TitleDict['Jonkheer']='Master'
TitleDict['Mme']='Mrs'
TitleDict['Ms']='Mrs'
TitleDict['Mrs']='Mrs'
TitleDict['Don']='Royalty'
TitleDict['Sir']='Royalty'
TitleDict['the Countess']='Royalty'
TitleDict['Dona']='Royalty'
TitleDict['Lady']='Royalty'
TitleDict['Capt']='Officer'
TitleDict['Col']='Officer'
TitleDict['Major']='Officer'
TitleDict['Dr']='Officer'
TitleDict['Rev']='Officer'

#构造新特征Title
full['Title']=full['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
full['Title']=full['Title'].map(lambda x: TitleDict[x])

# 增加家庭成员字段
full['familyNum'] = full['Parch'] + full['SibSp'] + 1

#我们按照家庭成员人数多少，将家庭规模分为“小、中、大”三类：
def familySize(familyNum):
	if familyNum <= 1:
		return 0
	elif (familyNum >= 2) and (familyNum <= 4):
		return 1
	else:
		return 2
	
full['familySize'] = full['familyNum'].map(familySize)

# Cabin字段的首字母代表客舱的类型，也反映不同乘客群体的特点，可能也与乘客的生存率相关。
full['Deck'] = full['Cabin'].map(lambda x: x[0])

# 票号  同一票号的乘客数量可能不同，可能也与乘客生存率有关系。
TickCountDict=full['Ticket'].value_counts()
full['TickCot'] = full['Ticket'].map(TickCountDict)

#按照TickCot大小，将TickGroup分为三类。 大小适中时存活率较高
def TickCountGroup(num):
    if (num>=2)&(num<=4):
        return 0
    elif (num==1)|((num>=5)&(num<=8)):
        return 1
    else :
        return 2
#得到各位乘客TickGroup的类别
full['TickGroup']=full['TickCot'].map(TickCountGroup)

# 并不是一昧地选择特征进行数据拟合而是选择有代表性的数据特征进行数据拟合
# 查看Age与Parch、Pclass、Sex、SibSp、Title、familyNum、familySize、Deck、TickCot、TickGroup等变量的相关系数大小，筛选出相关性较高的变量构建预测模型。

#筛选数据集
# Title 不是数字类型所以需要被特征处理
AgePre=full[['Age','Parch','Pclass','SibSp','Title','familyNum','TickCot']]
#进行one-hot编码
AgePre=pd.get_dummies(AgePre)
ParAge=pd.get_dummies(AgePre['Parch'],prefix='Parch')
SibAge=pd.get_dummies(AgePre['SibSp'],prefix='SibSp')
PclAge=pd.get_dummies(AgePre['Pclass'],prefix='Pclass')

AgeCorrDf=AgePre.corr()

#拼接数据
AgePre=pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)

#拆分实验集和预测集
AgeKnown=AgePre[AgePre['Age'].notnull()]
AgeUnKnown=AgePre[AgePre['Age'].isnull()]

#生成实验数据的特征和标签
AgeKnown_X=AgeKnown.drop(['Age'],axis=1)
AgeKnown_y=AgeKnown['Age']
#生成预测数据的特征
AgeUnKnown_X=AgeUnKnown.drop(['Age'],axis=1)

# 利用随机森林来进行预测
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)
rfr.fit(AgeKnown_X,AgeKnown_y)

# 模型得分
print(rfr.score(AgeKnown_X, AgeKnown_y))

# 预测年龄
AgeUnKnown_y = rfr.predict(AgeUnKnown_X)
# 数据填充回去
full.loc[full['Age'].isnull(),['Age']] = AgeUnKnown_y


## 同组识别 识别出具有共同特征的一组用户
# 我们知道性别和年龄其实是很主要的特征，因为女性和年龄小于12岁以下的乘客获救率更高
# 虽然这两个特征会被算法学习到，但是我们就真的能够完全让这个特征大行其道了吗？
# 但是我们知道男性中也会有生还者，女性中也会有死亡的，我们是否可以将它找出来然后再做特征纠正呢?
# 在Titancic案例中，我们主要探究相同姓氏的乘客是否存在明显的同组效应。 即同一个家族下

# 12岁以上男性：找出男性中同姓氏均获救的部分；
# 女性以及年龄在12岁以下儿童：找出女性及儿童中同姓氏均遇难的部分。


full['Surname'] = full['Name'].map(lambda x: x.split(',')[0].strip())
SurNameDict = full['Surname'].value_counts()
full['SurnameNum'] =  full['Surname'].map(SurNameDict)

# 将数据分成两组
MaleDf=full[(full['Survived'].notnull())&(full['Sex']=='male')&(full['Age']>12)&(full['familyNum']>=2)]
FemChildDf=full[(full['Survived'].notnull()) & ((full['Sex']=='female')|(full['Age']<=12))&(full['familyNum']>=2)]

#分析男性同组效应
MSurNamDf=MaleDf['Survived'].groupby(MaleDf['Surname']).mean()


print(MSurNamDf.value_counts())  # 共 19 个家族姓氏是会全部生还的 

# 找出来这些姓氏
MSurNamDict = MSurNamDf[MSurNamDf.values == 1.0].index

# full[(full['Name'].map(lambda x: x.split(',')[0].strip() in MSurNamDict)) & (full['Sex'] == 'male') & (full['Survived'].isnull())]


# 女性及儿童分析
FSurNameDf = FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
print(FSurNameDf.value_counts())   # 有 27个家族的人会全部死亡

# 找出这些姓氏
FCSurNamDict=FSurNameDf[FSurNameDf.values==0].index


# 其实我不是很明白为什么作者要 这要字修改需要预测的那一部分数据的特性 
#对数据集中这些姓氏的男性数据进行修正：1、性别改为女；2、年龄改为5。
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Age']=5
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Sex']='female'

#对数据集中这些姓氏的女性及儿童的数据进行修正：1、性别改为男；2、年龄改为60。
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Age']=60
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Sex']='male'

#人工筛选
fullSel=full.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)

corrDf=fullSel.corr()


fullSel=fullSel.drop(['familyNum','SibSp','TickCot','Parch'],axis=1)
#one-hot编码
fullSel=pd.get_dummies(fullSel)
PclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
TickGroupDf=pd.get_dummies(full['TickGroup'],prefix='TickGroup')
familySizeDf=pd.get_dummies(full['familySize'],prefix='familySize')

fullSel=pd.concat([fullSel,PclassDf,TickGroupDf,familySizeDf],axis=1)




#拆分实验数据与预测数据
experData=fullSel[fullSel['Survived'].notnull()]
preData=fullSel[fullSel['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)

#导入机器学习算法库
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)

#汇总不同模型算法
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())


#不同机器学习交叉验证结果汇总
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))


#求出模型得分的均值和标准差
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
#汇总数据
cvResDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})

	

	
#GradientBoostingClassifier模型
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(experData_X,experData_y)

#LogisticRegression模型
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_y)


#查看模型ROC曲线
#求出测试数据模型的预测值
modelgsGBCtestpre_y=modelgsGBC.predict(experData_X).astype(int)
#画图
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(experData_y, modelgsGBCtestpre_y) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic GradientBoostingClassifier Model')
plt.legend(loc="lower right")
plt.show()




#TitanicGBSmodle
GBCpreData_y=modelgsGBC.predict(preData_X)
GBCpreData_y=GBCpreData_y.astype(int)
#导出预测结果
GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['PassengerId']=full['PassengerId'][full['Survived'].isnull()]
GBCpreResultDf['Survived']=GBCpreData_y
#将预测结果导出为csv文件
GBCpreResultDf.to_csv('TitanicGBSmodle.csv',index=False)