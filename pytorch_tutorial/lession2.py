#making a singal neural network model 
import torch

def sigm(x):
    return 1/(1+torch.exp(-x))
    
torch.manual_seed(7)

inputs = torch.randn((1,10))
w1 = torch.randn_like(inputs)
b1 = torch.randn((1,1))

output = sigm(torch.matmul(inputs,w1.view(10,1))+b1)