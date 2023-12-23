import pandas as pd
import numpy as np
import torch
from torch import optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset 

excel_file = "血常规数据整合.xlsx"
data = pd.read_excel(excel_file)

X = data[['WBC', 'LY','GR','MO','RBC','Hgb','HCT','MCV','MCH','RDW','PLT','PCT','MPV','PDW']].values
Y = data['result'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train=torch.tensor(X_train)
Y_train=torch.tensor(Y_train)
train_dataset = TensorDataset(X_train,Y_train)
train_loader = DataLoader(train_dataset,batch_size=60,shuffle=True)

X_test=torch.tensor(X_test)
Y_test=torch.tensor(Y_test)
test_dataset = TensorDataset(X_test,Y_test)
test_loader = DataLoader(test_dataset,batch_size=60,shuffle=False)

class DCNNmodel(nn.Module):
	def __init__(self):
		super(DCNNmodel,self).__init__()
		self.cv1=nn.Conv1d(in_channels=14,out_channels=64,kernel_size=3,stride=1,padding=1)
		self.rl1=nn.ReLU()
		self.pl1=nn.MaxPool1d(kernel_size=3,stride=1)
		
		self.fc1=nn.Linear(1024,128)
		self.rl2=nn.ReLU()
		self.fc2=nn.Linear(128,3)
		
	def forward(self,x):
		print(x.shape)
		x=self.pl1(self.rl1(self.cv1(x)))
		x=(-1,1024)
		#卷积层至展平为一维
		x=self.fc2(self.rl2(self.fc1(x)))
		return x

model=DCNNmodel()

print(model)

criterion = nn.CrossEntropyLoss()
#损失函数min差距，计算损失更新权重
optimizer = optim.Adam(model.parameters(),lr=0.01)
#Adam优化算法，0.01学习率

for epoch in range(10):#进行10个周期
	for inputs,labels in train_loader:
		inputs = inputs.to(torch.float32)
		#print(inputs.shape)
		inputs=inputs.unsqueeze(-1)
		print(inputs.shape)
		optimizer.zero_grad()
		#优化器清零（默认相加，故在循环中每个周期开始时需要return0）
		outputs = model(inputs)
		#print(outputs.shape)
		loss = criterion(outputs,labels)
		loss.backward()
		#回传数据，更新参数
		optimizer.step()
		#依据以上更新参数，每次循环进行迭代
		
total = 0
correct = 0

with torch.no_grad():
	for inputs,labels in test_loader:
		inputs = inputs.to(torch.float32)
		inputs=inputs.unsqueeze(-1)
		outputs = model(inputs)
		_,predicted = torch.max(outputs.data,1)
		total += labels.size(0)
		correct += int((predicted == labels).sum())
		arrpre = predicted.int()
		
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

