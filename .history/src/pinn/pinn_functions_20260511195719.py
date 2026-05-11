import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.manual_seed(0)
device = "cpu"

class PINN(nn.Module):
    def __init__(self, in_dim=1, out_dim=5, width=100, depth=6):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

# Network outputs: x_nn, y_nn, theta, v_nn, omega_nn
# We'll enforce x,y BCs via transformation; theta and v are free ().
model = PINN(out_dim=5)

def derivative(y,x):
    """
    Input: Calculates gradient of tensor y, with respect to tensor x. 
    Output: A tuple containing the gradient tensor (dy_dx,)
    Access through: dy_dx = derivative(y, x)[0]
    """
    return torch.autograd.grad(
        y,x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

# Loss functions

def hard_bc_transform(t, nn_data, T, BC):
    x_nn        = nn_data[:, 0:1]
    y_nn        = nn_data[:, 1:2]
    theta_nn    = nn_data[:, 2:3]
    v_nn        = nn_data[:, 3:4]
    omega_nn    = nn_data[:, 4:5]
    
    # Boundary Conditions
    x0 = BC[0]
    y0 = BC[1]
    xT = BC[2]
    yT = BC[3]

    x = (T - t) * x0 + t * xT + t * (T - t) * x_nn
    y = (T - t) * y0 + t * yT + t * (T - t) * y_nn

    return x, y, theta_nn, v_nn, omega_nn

def phyics_loss(model, t_list, T, BC):
    """
    Input: model, list of time, boundary conditions description (x0,y0,xT,yT)
    Ouptut: loss function value of current position. 
    """
    nn_input = model(t_list)
    x, y, theta, v, omega = hard_bc_transform(t_list, nn_input, T, BC)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)
    theta_t = derivative(theta, t_list)

    r_x = x_t - v * torch.cos(theta)
    r_y = y_t - v * torch.sin(theta)
    r_theta = (theta_t - omega)

    L_phy = (r_x**2) + (r_y**2) + (r_theta**2)
    return torch.trapz(L_phy.squeeze(), t_list.squeeze())

def circ_obs_loss(model, t_list, obs, T, BC):
    """
    Input: model, list of time, circular obstacle description (x,y,r)
    Ouptut: loss function value of current position. 
    """
    nn_input = model(t_list)
    x, y, theta, v, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_c = obs[0]
    y_c = obs[1]
    r = obs[2]

    safety = 0.03        # Buffer zone

    d = torch.sqrt((x - x_c)**2 + (y - y_c)**2)

    # Obstacle avoidance loss (positive within a certain range of the obstacle center)
    d_sdf = d - r  # this is phi(x,y) for circle, matching eq.\eqref{eq:circsdf}
    violation = F.softplus((safety - d_sdf), beta=40)
    return torch.trapz((violation**2).squeeze(), t_list.squeeze())

def rect_obs_loss(model, t_list, obs, T, BC):
    """
    Input: model, list of time, circular obstacle description (x,y,r)
    Ouptut: loss function value of current position. 
    """
    nn_input = model(t_list)
    x, y, theta, v, _ = hard_bc_transform(t_list, nn_input, T, BC)

    xmin = obs[0]
    xmax = obs[1]
    ymin = obs[2]
    ymax = obs[3]


    # Look-Ahead method
    T_L = 0.1
    x_L = x + v * T_L * torch.cos(theta)
    y_L = y + v * T_L * torch.sin(theta)

    d_sdf = rect_sdf(x_L, y_L, xmin, xmax, ymin, ymax)

    safety = 0.03        # Buffer zone

    # Obstacle avoidance loss
    violation = F.softplus((safety - d_sdf), beta=40)
    return torch.trapz((violation**2).squeeze(), t_list.squeeze())

def smooth_loss(model, t_list, T, BC):
    """
    Input: model, list of time, boundary conditions description (x0,y0,xT,yT)
    Ouptut: loss function value of the sum of omega. 
    """
    nn_input = model(t_list)
    _, _, _, v, omega = hard_bc_transform(t_list, nn_input, T, BC)

    return torch.trapz((omega**2).squeeze() + 0.01 * (v**2).squeeze(), t_list.squeeze())  #

def omega_loss(model, t_list, T, BC):
    """
    Input: model, list of time, boundary conditions description (x0,y0,xT,yT)
    Ouptut: loss function value of the sum of omega. 
    """
    nn_input = model(t_list)
    _, _, _, _, omega = hard_bc_transform(t_list, nn_input, T, BC)

    return torch.trapz((omega**2).squeeze(), t_list.squeeze())

def v_loss(model, t_list, T, BC):
    """
    Input: model, list of time, boundary conditions description (x0,y0,xT,yT)
    Ouptut: loss function value of the sum of omega. 
    """
    nn_input = model(t_list)
    _, _, _, v, _ = hard_bc_transform(t_list, nn_input, T, BC)

    return torch.trapz((v**2).squeeze(), t_list.squeeze())

def lse_max(x, y, k=5):
    sum_stack = torch.stack((k*x, k*y), dim=0)
    return torch.logsumexp(sum_stack, dim=0) / k

def lse_min(x, y, k=5):
    return -(lse_max(-x, -y, k))

def rect_sdf(x, y, xmin, xmax, ymin, ymax):
    x_c = 0.5 * (xmin + xmax)
    y_c = 0.5 * (ymin + ymax)
    bx = 0.5 * (xmax - xmin)  # half width
    by = 0.5 * (ymax - ymin)  # half height

    # distance to the wall
    qx = torch.abs(x - x_c) - bx
    qy = torch.abs(y - y_c) - by

    # outside distance
    ox = F.softplus(qx, beta=5)
    oy = F.softplus(qy, beta=5)
    # ox = torch.relu(qx)
    # oy = torch.relu(qy)
    outside = torch.sqrt(ox**2 + oy**2)

    # inside term (negative or 0)
    inside = lse_min(lse_max(qx, qy), torch.zeros_like(qx))

    return outside + inside  # signed distance
