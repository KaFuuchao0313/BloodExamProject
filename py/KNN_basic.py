import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

excel_file = "血常规数据整合.xlsx"
data = pd.read_excel(excel_file)

X = data[['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']].values
Y = data['result'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

k = 10
clsf = KNeighborsClassifier(n_neighbors=k)
clsf.fit(X_train,Y_train)

Y_pred = clsf.predict(X_test)
print(Y_pred)
print(Y_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test,Y_pred)
print(acc)
