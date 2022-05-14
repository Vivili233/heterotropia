# -*-codeing=utf-8-*-
# @Time：2021/10/9 10:48
# @Autor:李薇
# File: tiaoyong.PY
# @Software: PyCharm
import numpy as np

import model

X_test=[19,0,1,1,-25,100,200,-25,-90,-90,-45,-45,0,0,0,0,0,0,0,0,0,0,0,0,0,8]
X_test=np.array(X_test).reshape(1,-1)
ways = model.way_model(X_test)
values=model.value_model(X_test)
for i in range(10):
    if ways[i] == 0:
        values[i] = 0
print(ways)
print(values)
