import pandas as pd
import numpy as np
import torch
from torch import optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset 

#x_train = torch.randn(150,14)
#y_train = torch.randn(0,3,(150))

excel_file = "血常规数据整合.xlsx"
data = pd.read_excel(excel_file)

X = data[['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']].values
Y = data['result'].values
#Y = data['result'].values

#print(X.shape)
#print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#print(X_train,Y_train)

X_train=torch.tensor(X_train)
Y_train=torch.tensor(Y_train)
train_dataset = TensorDataset(X_train,Y_train)
train_loader = DataLoader(train_dataset,batch_size=60,shuffle=True)
#batch_size即每次训练的量(从train_dataset中挑选)，shuffle意为随机

X_test=torch.tensor(X_test)
Y_test=torch.tensor(Y_test)
test_dataset = TensorDataset(X_test,Y_test)
test_loader = DataLoader(test_dataset,batch_size=60,shuffle=False)

class MLP(nn.Module):
	def __init__(self, **kwargs):
		#初始化模型的参数
		super(MLP, self).__init__(**kwargs)
		#父类构造函数正确初始化
		self.hidden = nn.Linear(14,32)#hidden隐藏层
		self.output = nn.Linear(32,3)#output输出层
		self.act = nn.ReLU()#ReLU激活函数
	
	def forward(self, x):
		x = self.hidden(x)
		x = self.act(x)
		x = self.output(x)
		return x
		#执行步骤
		

model = MLP()


criterion = nn.CrossEntropyLoss()
#损失函数min差距，计算损失更新权重
optimizer = optim.Adam(model.parameters(),lr=0.01)
#Adam优化算法，0.01学习率

for epoch in range(10):#进行10个周期
	for inputs,labels in train_loader:
		inputs = inputs.to(torch.float32)
		optimizer.zero_grad()
		#优化器清零（默认相加，故在循环中每个周期开始时需要return0）
		outputs = model(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		#回传数据，更新参数
		optimizer.step()
		#依据以上更新参数，每次循环进行迭代
		
total = 0
correct = 0
		
with torch.no_grad():#喜报，把backward禁了
	for inputs,labels in test_loader:
		inputs = inputs.to(torch.float32)
		outputs = model(inputs)
		#print(inputs)
		#print(outputs)
		_,predicted = torch.max(outputs.data,1)
		total += labels.size(0)
		correct += int((predicted == labels).sum())
		#print(type(predicted))
		#<class 'torch.Tensor'>
		arrpre = predicted.int()
		#转换作numpy数组/int
		#strpre = np.array_str(arrpre)
		#numpy至str
		#print(strpre)
		

length = len(arrpre)
for i in arrpre:
	#print(type(i))
	if i == 1:
		print('贫血')
	elif i == 0:
		print('病毒感染')
	elif i == 2:
		print('细菌感染')

print(f"准确率：{100 * correct / total}%")
		
