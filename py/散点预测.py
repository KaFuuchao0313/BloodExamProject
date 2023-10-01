import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

excel_file = "血常规数据整合.xlsx"
data = pd.read_excel(excel_file)

X = data[['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']].values
Y = data['result'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

svr_model = SVR(kernel='rbf',C=10,gamma=0.01,epsilon=.1)
svr_model.fit(X_train,Y_train)

Y_pre = svr_model.predict(X_test)

print(Y_pre)
print(Y_test)

