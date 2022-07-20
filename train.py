
"""# 导入python包"""

import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")

def per_data(train_data):
    train_data = list(train_data)
    train = []
    for i in range(len(train_data)):
        train_data1 = train_data[i]
        train1 = []
        for c in train_data1:
            x = np.array(c)
            x1 = np.real(x)  # 实数
            x2 = np.imag(x)  # 虚数
            train1.append(x1)
            train1.append(x2)
        train.append(np.array(train1))
    train = np.array(train)
    return train

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
eng.cd('./data')
all = eng.data(10000)


all_data =np.array(all[0])
all_label =np.array(all[1])

train_data = all_data[:6000,:]
train_label1 = all_label[:6000,:]#训练

test_data = all_data[6000:8000,:]
test_label1 =all_label[6000:8000,:]#验证

last_test_data = all_data[8000:10000,:]###最终测试部分

train_number = len(train_data)
test_number = len(test_data)

train = per_data(train_data)
train_label = per_data(train_label1)

test = per_data(test_data)
test_label = per_data(test_label1)


train = torch.tensor(train).float()
train_label = torch.tensor(train_label).float()

test = torch.tensor(test).float()
test_label = torch.tensor(test_label).float()

train_ds = TensorDataset(train, train_label)
val_ds = TensorDataset(test, test_label)

bs = 50

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False)
val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('use:',device)

"""# V1 构建模型 普通模型"""
class Aotuencoder(nn.Module):
    def __init__(self, ):
        super(Aotuencoder, self).__init__()
        self.encoder = nn.Sequential(
          nn.Linear(184, 400),
          nn.ReLU(),
          nn.Linear(400, 800),
          nn.ReLU()
        )
        self.decoder = nn.Sequential(
          nn.Linear(800, 400),
          nn.ReLU(),
          nn.Linear(400, 184)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


aotuencoder =Aotuencoder()
aotuencoder = aotuencoder.to(device)


"""## 分配损失函数和优化器"""

optimzer = torch.optim.Adamax(aotuencoder.parameters(), lr=0.001)
loss_func = nn.MSELoss().to(device)

"""## 定义训练循环"""

# model, opt = get_model()
epochs = 100
best_loss = 100

train_angle_err = []
test_angle_err = []

for epoch in range(epochs):
    running_loss = 0.0
    # for xb, yb in train_dl:
    for i, data in enumerate(train_dl):
        xb = data[0]
        xb = xb.cuda()
        label = data[1]
        label = label.cuda()
        # print(xb.shape,yb.shape)
        pred = aotuencoder(xb)
        
        # print(pred.shape)
        loss = loss_func(pred, label)
        # print(loss,pred[:5],yb[:5])
        loss.backward()
        optimzer.step()
        optimzer.zero_grad()

        running_loss += loss.item()
    print(f'[{epoch + 1}] train_loss: {running_loss / train_number:.5f}')
    train_angle_err.append(running_loss / train_number)

        # break
    aotuencoder.eval()
    with torch.no_grad():
        test_running_loss = 0.0
        for i, data1 in enumerate(val_dl):
            xb1 = data1[0]
            xb1 = xb1.to(device)
            label1 = data1[1]
            label1 = label1.cuda()
            # print(xb.shape,yb.shape)
            pred = aotuencoder(xb1)
            # print(pred.shape)
            loss1 = loss_func(pred, label1)
            test_running_loss+= loss1.item()
        val_loss =test_running_loss/test_number
        print(f'[{epoch + 1}] test_loss: {test_running_loss / test_number :.5f}')
        test_angle_err.append(test_running_loss / test_number)

    if val_loss<best_loss:
      best_loss = val_loss
      PATH ='./model/best_model.pth'
      # aotuencoder.state_dict()
      torch.save(aotuencoder.state_dict(), PATH)
      print('save best model!')

plt.plot(train_angle_err, label='train_loss')
plt.plot(test_angle_err, label='validate_loss')
plt.title('model train loss and validate loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('out.jpg', dpi=200)
plt.show()
plt.close()
print('training done')



###最终测试部分
last_test_data = per_data(last_test_data)
last_test_data_ds = torch.tensor(last_test_data).float()
bs1 = test_number

val_dl = DataLoader(last_test_data_ds, batch_size=bs1)
aotuencoder.eval()
aotuencoder.load_state_dict(torch.load("./model/best_model.pth"))
with torch.no_grad():
    for i, data in enumerate(val_dl):
        xb = data
        xb = xb.to(device)
        pred = aotuencoder(xb)
        pred = pred.cpu().detach().numpy()
        pred1 = list(pred)

    pred_s = []
    for j in range(len(pred1)):
        p1 = pred1[j]
        pp1 = []
        for k in range(0, p1.shape[0], 2):
            if p1[k + 1] < 0:
                p2 = "%f%fi" % (p1[k], p1[k + 1])
            else:
                p2 = "%f+%fi" % (p1[k], p1[k + 1])
            pp1.append(p2)
        pred_s.append(pp1)
    pred_s = np.array(pred_s)
    print(pred_s)
    np.savetxt('./out/pred_s.txt', pred_s, fmt="%s")