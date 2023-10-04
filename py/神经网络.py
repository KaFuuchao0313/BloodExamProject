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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train,Y_train)

train_dataset = TensorDataset(X_train,Y_train)
train_loader = DataLoader(train_dataset,batch_size=60,shuffle=True)
#batch_size即每次训练的量(从train_dataset中挑选)，shuffle意为随机

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
		optimizer.zero_grad()
		#优化器清零（默认相加，故在循环中每个周期开始时需要return0）
		outputs = model(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		#回传数据，更新参数
		optimizer.step()
		#依据以上更新参数，每次循环进行迭代
		
with torch.no_grad():#喜报，把backward禁了
	for inputs,labels in test_loader:
		outputs = model(inputs)
		print(inputs)
		print(outputs)
