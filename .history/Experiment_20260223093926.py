import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PINNs_functions import *


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

# Circle
x_c, y_c, r = 0.5, 0.1, 0.2
obs_circ = [x_c, y_c, r]

# Rectangle 
w, h = 1.5, 1.0 
xmin = x_c - w/2
xmax = x_c + w/2
ymin = y_c - h/2
ymax = y_c + h/2
obs_rect = [xmin, xmax, ymin, ymax]

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

# Plot of the trained path.
plt.figure()
plt.plot(x.cpu().numpy(), y.cpu().numpy())
plt.scatter([x0, xT], [y0, yT])
plt.title("PINN unicycle path w/ hard x,y BCs)")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")

ax = plt.gca()

obstacle_circle = plt.Circle((x_c, y_c), r, color='r', fill=True, alpha=0.3, label='Obstacle')
ax.add_patch(obstacle_circle)

"""
rect = patches.Rectangle((xmin, ymin), w, h)
ax.add_patch(rect)
"""

plt.legend()

plt.show()


# Plot theta(t) and v(t) vs time
plt.figure(figsize=(10, 4))

# Left subplot: angle (theta)
plt.subplot(1, 2, 1)
plt.plot(t_eval.cpu().numpy(), omega.cpu().numpy())
plt.xlabel("time t")
plt.ylabel("omega (angle/s)")
plt.title("Angularl velocity omega(t)")
plt.grid(True)

# Right subplot: velocity (v)
plt.subplot(1, 2, 2)
plt.plot(t_eval.cpu().numpy(), v.cpu().numpy())
plt.xlabel("time t")
plt.ylabel("v (velocity)")
plt.title("Velocity v(t)")
plt.grid(True)

plt.tight_layout()
plt.show()
