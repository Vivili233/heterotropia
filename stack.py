from itertools import cycle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

data = pd.read_csv('data_processed.csv')
X = data.iloc[:,:26]
y = data.iloc[:,48]
print("原始数据集左下斜肌分类分布")                            # 输出原始数据集样本分类分布
from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("上采样后的样本分类分布")
print(Counter(y_resampled))


# 1. 简单堆叠3折CV分类
RANDOM_SEED = 42
# 设置交叉验证次数
cvkFold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
clf1 = KNeighborsClassifier()
clf2 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                        bootstrap_features=False, n_jobs=1, random_state=1)
clf3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
lr = LogisticRegression()

# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],  # 第一层分类器
                            meta_classifier=lr,   # 第二层分类器
                            random_state=RANDOM_SEED)
# 参数stratify： 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0, test_size=0.3,stratify=y_resampled)
cv_score_list = []
pre_y_list = []
print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], ['KNN', 'bagging', 'GBDT','Stacking']):
    scores = cross_val_score(clf, X_resampled, y_resampled, cv=cvkFold , scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    # 将交叉检验结果存入结果列表
    cv_score_list.append(scores)
    # 将训练中得到的预测y存入列表
    pre_y_list.append(clf.fit(X_train, y_train).predict(X_test))
### 模型效果指标评估 ###
model_names = ['KNN', 'bagging', 'GBDT','Stacking']
# 评估指标对象列表
model_metrics_name = [accuracy_score,mean_squared_error]
model_metrics_name1 = [precision_score,recall_score,f1_score]
# 回归评估指标列表
model_metrics_list = []
# 循环每个模型的预测结果
for pre_y in pre_y_list:
    # 临时结果列表
    tmp_list = []
    # 循环每个指标对象
    for mdl in model_metrics_name:
        # 计算每个回归指标结果
        tmp_score = mdl(y_test, pre_y)
        # 将结果存入临时列表
        tmp_list.append(tmp_score)
    for mdl in model_metrics_name1:
            # 计算每个分类指标结果
            tmp_score = mdl(y_test, pre_y,average='weighted')
            # 将结果存入临时列表
            tmp_list.append(tmp_score)
    # 将结果存入回归评估列表
    model_metrics_list.append(tmp_list)
df_score = pd.DataFrame(cv_score_list, index=model_names)
df_met = pd.DataFrame(model_metrics_list, index=model_names, columns=['AC', 'MSE','P','R','F1'])
# 各个交叉验证的结果
print (df_score)
print("模型效果指标评估：")
print (df_met)
# Binarize the output
y_resampled = label_binarize(y_resampled, classes=[0, 1, 2])
n_classes = y_resampled.shape[1]
# 参数stratify： 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0, test_size=0.3,stratify=y_resampled)
#集成学习器stack
classifier = OneVsRestClassifier(sclf)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
lw=1
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()