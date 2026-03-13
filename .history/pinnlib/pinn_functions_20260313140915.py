import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.manual_seed(0)
device = "cpu"

class PINN(nn.Module):
    def __init__(self, in_dim=1, out_dim=5, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        """# Optional: small init helps stability in PINNs
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)"""

    def forward(self, t):
        return self.net(t)

# Network outputs: raw_xhat, raw_yhat, theta, v_raw
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
    raw_xhat = nn_data[:, 0:1]
    raw_yhat = nn_data[:, 1:2]
    theta    = nn_data[:, 2:3]
    v_raw    = nn_data[:, 3:4]
    omega_raw    = nn_data[:, 4:5]
    
    # Boundary Conditions
    x0 = BC[0]
    y0 = BC[1]
    xT = BC[2]
    yT = BC[3]

    x_lin = x0 * (1 - (t / T)) + (t / T) * xT
    y_lin = y0 * (1 - (t / T)) + (t / T) * yT

    f_theta = t * (T - t)
    x = x_lin + f_theta * raw_xhat
    y = y_lin + f_theta * raw_yhat

    # Bounding of velocity
    v = 10*torch.sigmoid(v_raw)
    # Bounding of angular velocity
    omega = 10*torch.sigmoid(omega_raw)


    return x, y, theta, v, omega

def physics_loss(model, t_list, T, BC):
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
    r_t = theta_t - omega

    # Physics loss (mean squared residuals)
    L_phys = (r_x**2).mean() + (r_y**2).mean() + (r_t**2).mean()

    L_phys = (r_x**2) + (r_y**2) + (r_t**2)
    return torch.trapz(L_phys.squeeze(), t_list.squeeze())

def circ_obs_loss(model, t_list, obs, T, BC):
    """
    Input: model, list of time, circular obstacle description (x,y,r)
    Ouptut: loss function value of current position. 
    """
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_c = obs[0]
    y_c = obs[1]
    r = obs[2]
    d = torch.sqrt((x - x_c)**2 + (y - y_c)**2)

    buffer = 0.05        # Buffer zone

    # Obstacle avoidance loss (positive within a certain range of the obstacle center)
    violation = F.softplus((r-d+buffer), beta=100)
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
    T_L = 0.05
    x_L = x + v * T_L * torch.cos(theta)
    y_L = y + v * T_L * torch.sin(theta)

    d_sdf = rect_sdf(x_L, y_L, xmin, xmax, ymin, ymax)

    buffer = 0.05        # Buffer zone

    # Obstacle avoidance loss (positive within a certain range of the obstacle center)
    violation = F.softplus((buffer-d_sdf), beta=100) 
    return torch.trapz((violation**2).squeeze(), t_list.squeeze())

def length_loss(model, t_list, T, BC):
    """
    Input: model, list of time, boundary conditions description (x0,y0,xT,yT)
    Ouptut: loss function value of the length. 
    """
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)

    sqrt = torch.sqrt(x_t**2 + y_t**2)

    return torch.trapz(sqrt.squeeze(), t_list.squeeze())

def smooth_loss(model, t_list, T, BC):
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

def lse_max(x, y, k=20):
    sum_stack = torch.stack((k*x, k*y), dim=0)
    return torch.logsumexp(sum_stack, dim=0) / k

def lse_min(x, y, k=20):
    return -(lse_max(-x, -y, k))

def rect_sdf(x, y, xmin, xmax, ymin, ymax):
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    bx = 0.5 * (xmax - xmin)  # half width
    by = 0.5 * (ymax - ymin)  # half height

    # distance to the wall
    qx = torch.abs(x - cx) - bx
    qy = torch.abs(y - cy) - by

    # outside distance
    ox = F.softplus(qx, beta=100)
    oy = F.softplus(qy, beta=100)
    # ox = torch.relu(qx)
    # oy = torch.relu(qy)
    outside = torch.sqrt(ox**2 + oy**2)

    # inside term (negative or 0)
    inside = lse_min(lse_max(qx, qy), torch.zeros_like(qx))

    return outside + inside  # signed distance
