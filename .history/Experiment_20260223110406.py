import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PINNs_functions import *
from training import train_model

# Training setup:
# Repeateldly adjusting the NN's parameters so that 
# its output trajectory satisfies physics and constraints by minimizing loss functions.

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# lr = eta, the factors that is multiplied with the gradient of the loss. 

num_epochs = 2000       # Num. of iterations of training
print_every = 500       # Print every 200 iterations

T = 1
N = 100

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

# Define BC
x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
BC = [x0,y0,xT,yT]

lambda_phys = 1
lambda_circ_obs = 0.1
lambda_rect_obs = 1
lambda_optim = 0
lambda_length = 0

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute losses
    L_phys = physics_loss(model, t_list, T, BC)
    L_circ_obs = circ_obs_loss(model, t_list, obs_circ, T, BC)
    L_length = length_loss(model, t_list, T, BC)

    loss = lambda_phys * L_phys + lambda_circ_obs * L_circ_obs + lambda_length * L_length
    loss.backward()
    optimizer.step()

    if epoch % print_every == 0:
        print(epoch, loss.item())

t_eval = torch.linspace(0.0, T, 200, device=device).view(-1, 1)
with torch.no_grad():
    nn_input = model(t_eval)
    x, y, theta, v, omega = hard_bc_transform(t_eval, nn_input, T, BC)

    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    tT = torch.tensor([[T]], dtype=torch.float32, device=device)
    x0p, y0p, *_ = hard_bc_transform(t0, model(t0), T, BC)
    xTp, yTp, *_ = hard_bc_transform(tT, model(tT), T, BC)
    print("\nBC check:")
    print(f"x(0)={x0p.item():.6f}, y(0)={y0p.item():.6f}")
    print(f"x(T)={xTp.item():.6f}, y(T)={yTp.item():.6f}")


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

results = []

for r in radii:
    for x_c in x_positions:

        obs = [x_c, 0.1, r]

        model = train_model(
            T=1,
            BC=[0, 0, 1, 0],
            obs_circ=obs,
            epochs=1000
        )

        kappa_max = compute_curvature(model)

        results.append([r, x_c, kappa_max])

data = np.array(results)