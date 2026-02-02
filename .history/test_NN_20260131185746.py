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
    Input: Calculates gradient of tensor y, with respect to tensor x. 
    Output: A tuple containing the gradient tensor (dy_dx,)
    Access through: dy_dx = derivative(y, x)[0]
    """
    return torch.autograd.grad(y,x,grad_outputs=torch.ones_like(y),create_graph=True)

# PINNs

def physics_loss(model,t):
    """
    Compare predicted value with the known physics model. 
    """

    r_pred = model(t)
    dr_dt_pred = derivative(r_pred, t)

    r_true = None
    dr_dt_true = derivative(r_true, t)

    loss_phys = torch.mean((dr_dt_pred-dr_dt_true)**2)

    return loss_phys

def boundary_conditions_loss(model, t_data, h_data):
    """
    MSE between predicted r_pred(0) and r_pred(T) and the given boundary conditions. 
    """
    r_pred = model(t_data)