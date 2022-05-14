# -*-codeing=utf-8-*-
# @Time：2021/9/24 19:38
# @Autor:李薇
# File: model.PY
# @Software: PyCharm
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.utils import class_weight

data = pd.read_csv('data_processed.csv')
X = data.iloc[:,:26]
#肌肉方式，右上直肌和右下直肌的手术方式只有无操作，故预测就没有意义
y_r1 = data.iloc[:,32]  #右外直肌
# y_r2 = data.iloc[:,26]  #右内直肌
# y_r3 = data.iloc[:,28]  #右上直肌
# y_r4 = data.iloc[:,30]  #右下直肌
y_r5 = data.iloc[:,28]  #右上斜肌
y_r6 = data.iloc[:,30]  #右下斜肌
#左下直肌的手术方式只有无操作
y_l1 = data.iloc[:,32]  #左外直肌
y_l2 = data.iloc[:,34]  #左内直肌
y_l3 = data.iloc[:,36]  #左上直肌
# y_l4 = data.iloc[:,30]  #左下直肌
y_l5 = data.iloc[:,38]  #左上斜肌无任何方式
y_l6 = data.iloc[:,40]  #左下斜肌

#19	0	1	1	-25	100	200	-25	-90	-90	-45	-45	0	0	0	0	0	0	0	0	0	0	0	0	0	16
X_test=[19,0,1,1,-25,100,200,-25,-90,-90,-45,-45,0,0,0,0,0,0,0,0,0,0,0,0,0,8]
X_test=np.array(X_test).reshape(1,-1)
RANDOM_SEED = 42
# 模型
clf1 = KNeighborsClassifier()
clf2 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True,
                        bootstrap_features=False, n_jobs=1, random_state=1)
clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
lr = LogisticRegression()
# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],  # 第一层分类器
                            meta_classifier=lr,   # 第二层分类器
                            random_state=RANDOM_SEED)
#集成学习器stack
classifier = OneVsRestClassifier(sclf)

# y_rv1=data.iloc[:,27]
# gbr_model = GradientBoostingRegressor() # 梯度增强回归模型对象
# lr_model = LinearRegression()
# br_model = BayesianRidge()
#
# print(lr_model.fit(X,y_rv1).predict(X_test))
# print(br_model.fit(X,y_rv1).predict(X_test))
#print(gbr_model.fit(X,y_rv1).predict(X_test))


def way_model(X_test):
    ways=[]
    for i in range(0,12):
        y=data.iloc[:,26+i*2]
        #上采样
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        ways.append(int(classifier.fit(X_resampled,y_resampled).predict(X_test)))
    return ways

def value_model(X_test):
    values=[]
    for i in range(0,12):
        y = data.iloc[:, 27+i*2]
        gbr_model = GradientBoostingRegressor()  #
        values.append(int(gbr_model.fit(X, y).predict(X_test)))
    return values

# print(way_model(X_test))
# print(value_model(X_test))
