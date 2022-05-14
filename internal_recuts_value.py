import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

data = pd.read_csv('data_processed.csv')
train_x = data.iloc[:,:24]
train_y = data.iloc[:,27]
# 划分数据集
#train_x, X_test, train_y, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
#----------------决策回归树算法---------------------------
from sklearn.tree import DecisionTreeRegressor

 # 参数优化--交叉检验
tuned_parameters = {
                    'criterion':['mse'],
                    'min_samples_split':[2,10,20],
                    'max_depth':[2,10,20,40],
                    'min_samples_leaf':[1,5,10],
                    'max_leaf_nodes':[2,10,20,40],
                    }
clf = DecisionTreeRegressor()
clf = GridSearchCV(data,tuned_parameters,cv=5)

clf.fit(train_x,train_y)
clf.best_params_
for params,mean_score,scores in clf.grid_score_:
    print('$0.3f(+/-%0.03f) for %r'
          %(mean_score,scores.std()*2,params))
 #可视化----模型结果展示----变量重要性显示
important_features = pd.Series(data = clf.best_estimator_.feature_importances_,index = train_x.columns).sort_values(ascending = False)
plt.figure(figsize = (20,10))
important_features.plot(king = 'bar')

 #效果评估
from sklearn.metrics import mean_squared_error
pred_y = clf.predict(train_x)
np.sqrt(mean_squared_error(np.expml(train_y),np.expml(pred_y)))

plt.figure(figsize = (20,10))
plt.scatter(x = np.expml(train_y),y = np.expml(pred_y))