import pandas as pd
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
data = pd.read_csv('data_processed.csv')
X = data.iloc[:,:24]
y = data.iloc[:,24]
print("原始数据集样本分类分布")                            # 输出原始数据集样本分类分布
from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(y))
from imblearn.over_sampling import SMOTE
X_resampled,y_resampled = SMOTE().fit_resample(X, y)
print("上采样后的样本维度")
print(Counter(y_resampled))