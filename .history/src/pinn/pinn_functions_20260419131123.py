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

# Network outputs: x_nn, y_nn, theta, v_nn
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
    v0 = BC[4]
    v0 = 2
    theta0 = BC[5]
    theta0 = 0.0

    v_free = 5 * torch.tanh(v_nn)
    alpha = 5.0
    exp_term = torch.exp(-alpha * t)
    v = v0 + t * v_free

    theta_free = 5*torch.tanh(omega_nn)
    theta = theta0 + t * theta_free


    # Linear + quadratic terms
    x_lin = x0 + (v0 * torch.cos(torch.tensor(theta0))) * t
    y_lin = y0 + (v0 * torch.sin(torch.tensor(theta0))) * t

    x_quad = ((xT - x0 - v0 * torch.cos(torch.tensor(theta0)) * T) / (T**2)) * (t**2)
    y_quad = ((yT - y0 - v0 * torch.sin(torch.tensor(theta0)) * T) / (T**2)) * (t**2)

    # Neural network correction (IMPORTANT: t^2 term!)
    x_nn_term = (t**2) * (T - t) * x_nn
    y_nn_term = (t**2) * (T - t) * y_nn

    # Final positions
    x = x_lin + x_quad + x_nn_term
    y = y_lin + y_quad + y_nn_term

    x = (1 - t) * x0 + t * xT + t * (1 - t) * x_nn
    y = (1 - t) * y0 + t * yT + t * (1 - t) * y_nn

    return x, y, theta, v, omega_nn

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
    r_t = theta_t - omega

    # phyics loss (mean squared residuals)
    L_phy = (r_x**2).mean() + (r_y**2).mean() + (r_t**2).mean()

    L_phy = (r_x**2) + (r_y**2) + (r_t**2)
    return torch.trapz(L_phy.squeeze(), t_list.squeeze())

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
    violation = F.softplus((r-d+buffer), beta=5)
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
    T_L = 0.01
    x_L = x + v * T_L * torch.cos(theta)
    y_L = y + v * T_L * torch.sin(theta)

    d_sdf = rect_sdf(x_L, y_L, xmin, xmax, ymin, ymax)

    buffer = 0.05        # Buffer zone

    # Obstacle avoidance loss (positive within a certain range of the obstacle center)
    violation = F.softplus((buffer-d_sdf), beta=5) 
    return torch.trapz((violation**2).squeeze(), t_list.squeeze())

def smooth_loss(model, t_list, T, BC):
    """
    Input: model, list of time, boundary conditions description (x0,y0,xT,yT)
    Ouptut: loss function value of the sum of omega. 
    """
    nn_input = model(t_list)
    _, _, _, v, omega = hard_bc_transform(t_list, nn_input, T, BC)

    return torch.trapz((omega**2).squeeze() + (v**2).squeeze(), t_list.squeeze())  #  

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
    ox = F.softplus(qx, beta=50)
    oy = F.softplus(qy, beta=50)
    # ox = torch.relu(qx)
    # oy = torch.relu(qy)
    outside = torch.sqrt(ox**2 + oy**2)

    # inside term (negative or 0)
    inside = lse_min(lse_max(qx, qy), torch.zeros_like(qx))

    return outside + inside  # signed distance
