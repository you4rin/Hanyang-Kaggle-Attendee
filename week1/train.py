import torch
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("weight-height.csv")

X, y = data[['Height']].values, data[['Weight']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

net = nn.Linear(1, 1)
opt = optim.Adam(net.parameters(), lr = 1e-1, weight_decay = 1e-4)
criteria = nn.MSELoss()

for epoch in range(1, 5001):
    pred = net(X_train)
    opt.zero_grad()
    loss = criteria(pred, y_train)
    loss.backward()
    opt.step()
    
    print("epoch {}: loss {:.4f}".format(epoch, loss.item()))
    
    with torch.no_grad():
        pred = net(X_test)
        loss = torch.mean((y_test - pred) ** 2)
    
        print("test {}: loss {:.4f}".format(epoch, loss.item()))
        
torch.save(net.state_dict(), 'ckpt/checkpoint-%04d.pth' % epoch)