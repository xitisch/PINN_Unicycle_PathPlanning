# Vary obstacle x-position (longitudinal), keep the same radius.
# Saves:
#   figures/longitudinal/cicle/long_circ_k_vs_xc.png
#   figures/longitudinal/cicle/long_circ_5x1.png
#   figures/longitudinal/cicle/long_circ_5x2.png

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import LightSource

from src.pinn.pinn_functions import *
from src.pinn.train_pinn import train_model

output_folder = os.path.join("results", "3D", "circle")
os.makedirs(output_folder, exist_ok=True)

def compute_curvature(model, t_list, T, BC):
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_t  = derivative(x, t_list)
    y_t  = derivative(y, t_list)
    x_tt = derivative(x_t, t_list)
    y_tt = derivative(y_t, t_list)

    eps = 1e-6
    kappa = (x_t * y_tt - y_t * x_tt) / ((x_t**2 + y_t**2 + eps) ** (3/2))

    t_np = t_list.squeeze().detach().cpu().numpy()
    k_np = kappa.squeeze().detach().cpu().numpy()
    kappa_max = torch.max(torch.abs(kappa)).item()

    return t_np, k_np, kappa_max


def get_trajectory(model, t_list, T, BC):
    with torch.no_grad():
        nn_input = model(t_list)
        x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)
    return x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy()



# Experiment setup 
T = 1.0
N = 400
epochs = 3000

lambda_phy = 20
lambda_obs = 50
lambda_v = 0.2
lambda_omega = 2

x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
v0 = 2
theta0 = 0
BC = [x0,y0,xT,yT,v0,theta0]

# Fixed radius
r_fixed = 0.20

Delta = 0.10
y_c = r_fixed - Delta
if y_c < 0:
    raise ValueError(f"Delta={Delta} too large for r_fixed={r_fixed}: y_c would be negative.")

# Vary longitudinal obstacle position
x_positions = np.arange(0.3, 0.7 + 1e-9, 0.02)
Delta_values = np.arange(0.0, 0.2 + 1e-9, 0.01)


# Grids
t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)


# Run experiments
scenarios = []   # store trajectories for plotting

print(f"Fixed radius r = {r_fixed:.2f}")
print(f"x_c range: [{x_positions.min():.2f}, {x_positions.max():.2f}], step = {x_positions[1]-x_positions[0]:.2f}")
print(f"Delta range: [{Delta_values.min():.2f}, {Delta_values.max():.2f}], step = {Delta_values[1]-Delta_values[0]:.2f}")

K = np.zeros((len(Delta_values), len(x_positions)))

for i, Delta in enumerate(Delta_values):
    y_c = r_fixed - Delta

    if y_c < 0:
        K[i, :] = np.nan
        continue

    print(f"\n=== Delta = {Delta:.3f} → y_c = {y_c:.3f} ===")

    for j, x_c in enumerate(x_positions):
        total = len(Delta_values) * len(x_positions)
        counter = i * len(x_positions) + j + 1

        print(f"  [{counter}/{total}] Training: x_c = {x_c:.3f}")

        obs = [[x_c, y_c, r_fixed]]

        model = train_model(
            T=T,
            BC=BC,
            obs=obs,
            epochs=epochs,
            lambda_phy=lambda_phy,
            lambda_obs=lambda_obs,
            lambda_v=lambda_v,
            lambda_omega=lambda_omega,
            N=N
        )

        _, _, kappa_max = compute_curvature(
            model, t_list=t_list, T=T, BC=BC
        )

        K[i, j] = kappa_max


# Plot 1: curvature vs x_c

from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(x_positions, Delta_values)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

norm = plt.Normalize(np.nanmin(K), np.nanmax(K))

ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(K, cmap=cm.viridis, vert_exag=0.1, blend_mode='soft')

surf = ax.plot_surface(
    X, Y, K,
    facecolors=rgb,
    linewidth=0,
    antialiased=True,
    alpha=0.9
)

ax.set_xlabel(r"$x_c$")
ax.set_ylabel(r"$\Delta$")
ax.set_zlabel(r"$\kappa_{\max}$")
ax.set_title("3D surface of curvature")

ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

cbar = fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)
cbar.set_label(r"$\kappa_{\max}$")

plt.tight_layout()

path = os.path.join(output_folder, "surface_kappa_xc_delta.png")
plt.savefig(path, dpi=300)
plt.close()

print(f"Saved: {path}")