import torch
import numpy as np
import matplotlib.pyplot as plt
from PINNs_functions import *

def compute_curvature(model, t_list, T, BC):
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)
    x_tt = derivative(x_t, t_list)
    y_tt = derivative(y_t, t_list)

    safety = 0
    kappa = (x_t * y_tt - y_t * x_tt) / ((x_t**2 + y_t**2 + safety)**(3/2))

    return torch.max(torch.abs(kappa)).item()

radii = [0.05, 0.1, 0.15, 0.2]
x_positions = [0.3, 0.4, 0.5, 0.6]
T = 1
N = 100

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

# Define BC
x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
BC = [x0,y0,xT,yT]

results = []

for r in radii:
    for x_c in x_positions:

        model = PINN()
        # train model (reuse your training loop)

        max_kappa = compute_curvature(model, t_list, T, BC)

        results.append([r, x_c, max_kappa])


plt.plot(radii, kappas)
plt.xlabel("Obstacle radius")
plt.ylabel("Max curvature")