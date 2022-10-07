from tkinter import N
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(10,20)
        self.fc2 = nn.Linear(20,20)
        self.out = nn.Linear(20,10)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

input = torch.rand(10)
#instanciate model
net = Net()
output = net(input)
print(output)