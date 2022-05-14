# example of stratified k-fold cross-validation with an imbalanced dataset
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

# import dataset
data = pd.read_csv('data_processed.csv')
X = data.iloc[:,:24]
y = data.iloc[:,26]
#StratifiedKFold建模过程
skf = StratifiedKFold(n_splits=3,random_state=None)
rf = RandomForestClassifier(n_estimators=500,random_state=10)
print('StratifiedKFold结果:')
for index in skf.split(X,y): #因为是3等分，即3折交叉，一共循环5次
    X_train = X.loc[index[0],:]
    y_train = y[index[0]]
    X_test = X.loc[index[1],:]
    y_test = y[index[1]]
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    print(rf.score(X_test,y_test))
# 利用cross_validate建模
cv_results = cross_validate(RandomForestClassifier(n_estimators=10, random_state=10),
                            X, y, cv=3, scoring='accuracy')
print('cross_validate结果：', cv_results['test_score'])

# 利用GridSearchCV建模
param_grid = {'n_estimators': [10]}
gs = GridSearchCV(RandomForestClassifier(random_state=10), param_grid,
                  scoring='accuracy', cv=3, n_jobs=-1)  # 设置参数verbose=1,可以查看训练进程

gs.fit(X, y)
print('GridSearchCV结果：')
pd.DataFrame(gs.cv_results_)[['param_n_estimators', 'split0_test_score', 'split1_test_score', 'split2_test_score']]