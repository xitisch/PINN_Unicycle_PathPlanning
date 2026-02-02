import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(0)
device = "cpu"

class PINN(nn.Module):
    def __init__(self, in_dim=1, out_dim=4, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        # Optional: small init helps stability in PINNs
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

# Network outputs: raw_xhat, raw_yhat, theta, v_raw
# We'll enforce x,y BCs via transformation; theta and v are free.
model = PINN(out_dim=4)



# Define BC
x0, y0 = 0.0, 0.0
yT, yT = 1.0, 1.0

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

def boundary_conditions_loss(model, t_data):
    """
    MSE between predicted r_pred(0) and r_pred(T) and the given boundary conditions. 
    """
    r_pred = model(t_data)
    return torch.mean((r_pred-r_data)**2)

# Training setup:
# # repeateldly adjusting the NN's parameters so that 
# its output trajectory satisfies physics, BC, and constraints by minimizing loss functions.

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# lr = eta, the factors that is multiplied with the gradient of the loss. 

lambda_phys = 1
lambda_ic = 1
lambda_bc = 1
lambda_optim = 0

num_epochs = 2000       # Num. of iterations of training
print_every = 200       # Print every 200 iterations

N = 100  # number of points
tau = torch.linspace(0.0, 1.0, N).view(-1, 1)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute losses
    L_phys = physics_loss(model, t_data)
    L_ic = boundary_conditions_loss(model, t_data)

def PINNs(t):
    """
    Input: time interval
    Output: array of x,y,theta,v,w at different discrete time steps. 
    """

    return x,y,theta,v,w