import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# 读入数据集
data = pd.read_csv('data_processed.csv')
x = data.iloc[:,:24]
y = data.iloc[:,24]
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

# 度量单个决策树的准确性
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
# Decision tree train/test accuracies 1.000/0.942

#KNN

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
knn_train = accuracy_score(y_train, y_train_pred)
knn_test = accuracy_score(y_test, y_test_pred)
print('KNN train/test accuracies %.3f/%.3f' % (knn_train, knn_test))
print('KNN precision/Recall %.3f/%.3f' % (precision_score(y_test_pred, y_test, average='micro'), recall_score(y_test_pred, y_test, average='micro')))
print('KNN f1_score %.3f' % f1_score(y_test, y_test_pred, average='micro'))

# 度量bagging分类器的准确性
# 生成500个决策树
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                        bootstrap_features=False, n_jobs=1, random_state=1)
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))


# 随机森林，bagging思想
rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True,
                            n_jobs=1, random_state=1)
# 度量随机森林的准确性
rf = rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

# Boosting分类器准确性
ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

# GBDT
gbdt = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
gbdt = gbdt.fit(X_train, y_train)
y_train_pred = gbdt.predict(X_train)
y_test_pred = gbdt.predict(X_test)
gbdt_train = accuracy_score(y_train, y_train_pred)
gbdt_test = accuracy_score(y_test, y_test_pred)
print('GBDT train/test accuracies %.3f/%.3f' % (gbdt_train, gbdt_test))