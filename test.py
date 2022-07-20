
"""# 导入python包"""

import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
eng.cd('./data')
test_number = 1000
test_data = eng.data1(test_number)
print(test_data)
test_data = list(test_data)
test = []
for i in range(len(test_data)):
    test_data1 = test_data[i]
    test1 = []
    for c in test_data1:
        x = np.array(c)
        x1 = np.real(x)#实数
        x2 = np.imag(x)#虚数
        test1.append(x1)
        test1.append(x2)
    test.append(np.array(test1))
test = np.array(test)

val_ds = torch.tensor(test).float()
bs = test_number

val_dl = DataLoader(val_ds, batch_size=bs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('use:',device)

"""# V1 构建模型 普通模型"""


class Aotuencoder(nn.Module):
    def __init__(self, ):
        super(Aotuencoder, self).__init__()
        self.encoder = nn.Sequential(
          nn.Linear(184, 4000),
          nn.ReLU(),
          nn.Linear(4000, 8000),
          nn.ReLU()
        )
        self.decoder = nn.Sequential(
          nn.Linear(8000, 4000),
          nn.ReLU(),
          nn.Linear(4000, 184)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

loss_func = nn.MSELoss().to(device)

aotuencoder =Aotuencoder().to(device)
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
        p1=pred1[j]
        pp1=[]
        for k in range(0,p1.shape[0],2):
            if p1[k+1]<0:
                p2="%f%fi"%(p1[k],p1[k+1])
            else:
                p2 = "%f+%fi" % (p1[k], p1[k + 1])
            pp1.append(p2)
        pred_s.append(pp1)
    pred_s = np.array(pred_s)
    np.savetxt('./out/pred_s.txt',pred_s,fmt="%s")
