import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

excel_file = "血常规数据整合.xlsx"
data = pd.read_excel(excel_file)

# 假设Excel文件的列为 ['特征1', '特征2', '标签']
X = data[['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']].values
y = data['result'].values

# 将数据集分为训练集和测试集
# test_size=0.2代表train:test=8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建一个SVM分类器
svm_classifier = SVC(kernel='linear', C=1.0)

# 在训练集上训练分类器
svm_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm_classifier.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率：{accuracy:.2f}")

# 第一次训练了0.62准确率XD
