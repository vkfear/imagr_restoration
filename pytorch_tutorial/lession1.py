import torch
import numpy as np
#creating tensor
tensor_array = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print("tensor_array: \n",tensor_array)

#creating 
torch_rand = torch.rand(3,3)
torch_zeros = torch.zeros(3,3)
torch_ones = torch.ones(3,3)

##printing one,zero,random matrix
# print("\n3 by 3 ones\n",torch_ones)
# print("\n3 by 3 zeros\n",torch_zeros)
# print("\n3 by 3 rand\n",torch_rand)

# #matrix multiplication
# torch_a1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# torch_a2 = torch.tensor([[2,3],[2,3],[2,3]])
# print(torch.mm(torch_a1,torch_a2))
# print("\n\n")

# #eliment_wise multiplicatin
# print(torch_a1*3)

#numpy array
np_a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])

torch_a1 = torch.from_numpy(np_a1)
print(torch_a1)
