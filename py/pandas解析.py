import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#anemia(贫血)_1
#bac_infect(细菌感染)_2

# 读取Excel文件
excel_file = "血常规数据整合.xlsx"
data = pd.read_excel(excel_file)

# 假设Excel文件的列为 ['特征1', '特征2', '标签']
features = data[['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']].values
labels = data['result'].values

#astype
#input.astype(np.float32)

# 数据预处理
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 创建Keras模型
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
X_train = X_train.astype('float64') 
Y_train = Y_train.astype('float64') 
X_test = X_test.astype('float64') 
X_test = X_test.astype('float64') 
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))
