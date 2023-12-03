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
#X_train=X_train.unsqueeze(-1)
#X_train=X_train.unsqueeze(-1)
#X_train=X_train.repeat(1,1,8,8)
#Y_train=Y_train.unsqueeze(-1)
#Y_train=Y_train.unsqueeze(-1)
#Y_train=Y_train.repeat(1,1,8,8)
#print(X_train.shape)
#shape=[239,14,8,8]

X_test=torch.tensor(X_test)
Y_test=torch.tensor(Y_test)
test_dataset = TensorDataset(X_test,Y_test)
test_loader = DataLoader(test_dataset,batch_size=60,shuffle=False)
#X_test=X_test.unsqueeze(-1)
#X_test=X_test.unsqueeze(-1)
#X_test=X_test.repeat(1,1,8,8)
#Y_test=Y_test.unsqueeze(-1)
#Y_test=Y_test.unsqueeze(-1)
#Y_test=Y_test.repeat(1,1,8,8)

class SimpleCNN(nn.Module):
	def __init__(self,num_classes=3):
		#num_classes即最终输出的类别数（默认10）
		super(SimpleCNN,self).__init__()
		self.cv1=nn.Conv2d(in_channels=14,out_channels=32,kernel_size=3,stride=1,padding=1)
		#输入14，输出32，卷积核为3*3（后面调试）
		#padding=1:保证输入输出大小一致
		self.relu1=nn.ReLU()
		self.p1=nn.MaxPool2d(kernel_size=5,stride=1)
		#窗口越大步长越小指向全局而忽略细节，反之同理
		
		self.cv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
		self.relu2=nn.ReLU()	
		self.p2=nn.MaxPool2d(kernel_size=5,stride=1)	
		
		self.input=nn.Linear(1024,128)
		#3(上层卷积输出)*9*9（池化数据，后续调试）
		self.relu3=nn.ReLU()
		self.output=nn.Linear(128,num_classes)

	def forward(self,x):
		x=self.p1(self.relu1(self.cv1(x)))
		x=self.p2(self.relu2(self.cv2(x)))
		#print(x.shape)
		x=x.view(-1,1024)
		x=self.input(x)
		x=self.relu3(x)
		x=self.output(x)
		#x=x.repeat(60,1)
		return x
		
model = SimpleCNN()

#print(model)

criterion = nn.CrossEntropyLoss()
#损失函数min差距，计算损失更新权重
optimizer = optim.Adam(model.parameters(),lr=0.01)
#Adam优化算法，0.01学习率

for epoch in range(10):#进行10个周期
	for inputs,labels in train_loader:
		inputs = inputs.to(torch.float32)
		inputs=inputs.unsqueeze(-1)
		inputs=inputs.unsqueeze(-1)
		inputs=inputs.repeat(1,1,8,8)
		#print(inputs.shape)
		#print(labels.shape)
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
		
with torch.no_grad():#喜报，把backward禁了
	for inputs,labels in test_loader:
		inputs = inputs.to(torch.float32)
		inputs=inputs.unsqueeze(-1)
		inputs=inputs.unsqueeze(-1)
		inputs=inputs.repeat(1,1,8,8)
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
