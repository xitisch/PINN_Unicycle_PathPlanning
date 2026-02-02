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

def hard_bc_transform(t, nn_data, BC):
    raw_xhat = nn_data[:, 0:1]
    raw_yhat = nn_data[:, 1:2]
    theta    = nn_data[:, 2:3]
    v_raw    = nn_data[:, 3:4]
    
    # IC
    x0 = BC[0,1]


    x_lin = x0 * (1 - (t / T)) + (t / T) * xT
    y_lin = y0 * (1 - (t / T)) + (t / T) * yT

    f_theta = t * (T - t)
    x = x_lin + f_theta * raw_xhat
    y = y_lin + f_theta * raw_yhat

    # Bounding of velocity
    # v = torch.tanh(v_raw)
    # Bounding of curvature
    # Kappa tbd
    return x, y, theta, v

def physics_loss(model, t_list):
    raw = model(t_list)
    x, y, theta, v = hard_bc_transform(t_list, raw)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)

    r_x = x_t - v * torch.cos(theta)
    r_y = y_t - v * torch.sin(theta)

    # Physics loss (mean squared residuals)
    L_phys = (r_x**2).mean() + (r_y**2).mean()

    # Optional regularization for smoother theta/v (helps training)
    """theta_t = derivative(theta, t_list)
    v_t = derivative(v, t_list)
    L_reg = 1e-3 * (theta_t**2).mean() + 1e-3 * (v_t**2).mean()"""

    return L_phys

# Training setup:
# Repeateldly adjusting the NN's parameters so that 
# its output trajectory satisfies physics and constraints by minimizing loss functions.

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# lr = eta, the factors that is multiplied with the gradient of the loss. 

lambda_phys = 1
lambda_ic = 1
lambda_optim = 0

num_epochs = 2000       # Num. of iterations of training
print_every = 200       # Print every 200 iterations

T = 1
N = 100

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

!!!!
# Define BC
x0, y0 = 0.0, 0.0
xT, yT = 1.0, 1.0

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute losses
    L_phys = physics_loss(model, t_list)

    loss = L_phys
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, loss.item())

t_eval = torch.linspace(0.0, T, 200, device=device).view(-1, 1)
with torch.no_grad():
    raw = model(t_eval)
    x, y, theta, v = hard_bc_transform(t_eval, raw)

    # Check boundary conditions (should be exact up to float precision)
    t0 = torch.tensor([[0.0]], device=device)
    tT = torch.tensor([[T]], device=device)
    x0p, y0p, *_ = hard_bc_transform(t0, model(t0))
    xTp, yTp, *_ = hard_bc_transform(tT, model(tT))
    print("\nBC check:")
    print(f"x(0)={x0p.item():.6f}, y(0)={y0p.item():.6f}")
    print(f"x(T)={xTp.item():.6f}, y(T)={yTp.item():.6f}")

plt.figure()
plt.plot(x.cpu().numpy(), y.cpu().numpy())
plt.scatter([x0, xT], [y0, yT])
plt.title("PINN unicycle path (hard x,y boundary conditions)")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
plt.show()