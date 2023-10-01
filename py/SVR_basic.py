#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  SVR_basic.py
#  
#  Copyright 2023 ZhibinLi <ZhibinLi@DESKTOP-7D5TDA0>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
#from sklearn.metrics import accuracy_score

#gridsearchcv可以调优参数以及交叉

excel_file = "血常规数据整合.xlsx"
data = pd.read_excel(excel_file)

#def string_to_float(str):
#	return float(str)

# 假设Excel文件的列为 ['特征1', '特征2', '标签']
X = data[['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']].values
y = data['result'].values

# 将数据集分为训练集和测试集
# test_size=0.2代表train:test=8:2
# random_state=x则是不同的随机数组，可以填随意的数字用以验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# 创建SVR模型
#svr_model = SVR(kernel='linear', C=1.0)
#svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

pip = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=10, gamma=0.01, epsilon=.1))])
#寻找最佳参数

# 拟合（训练）模型
#pip.fit(X_train, y_train)
pip.fit(X_train, y_train)  # X_train是训练特征集，y_train是训练目标值
y_pred = pip.predict(X_test)  # X_test是测试特征集

#输出
acc = pip.score(X_test,y_test)
print(acc)

#获取参数
arr = pip.get_params()
print(arr)

param_gs = {"svr__C":[0.1,1,10,100,1000],"svr__kernel":["poly","rbf","sigmoid"],"svr__gamma":[1e-7,1e-4,1e-3,1e-2]}
gs=GridSearchCV(pip,param_gs,n_jobs=-1,verbose=1)
gs.fit(X_train,y_train)

param_arr=gs.best_estimator_
print(param_arr)

# 进行预测
#features = ['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']
#predictions = svr_model.predict([features])

#pre = float(predictions)

# 打印预测结果
#string_to_float(predictions)
