import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def true_solution(x,y):
    return 

t_min = 0
t_max = 2
N_data = 10

# Create a NumPy array
t_data = np.linspace(t_min, t_max, N_data)

# PyTorch tensor conversion
data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1,1)      # .view() adjust the dimension, -1 means infer dimension automatically. 

def derivative(y,x):
    """
    Input: Calculatese gradient of tensor y, with respect to tensor x. 
    """
    return torch.autograd.grad(y,x,grad_outputs=torch.ones_like(y),create_graph=True)