import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pinnlib.pinn_functions import *
from pinnlib.training_pinn import train_model


# Setup
T = 1.0
N = 100

lambda_phy = 1
lambda_obs = 10
lambda_omega = 0.0001

BC = [0, 0, 1, 0]

# Fixed obstacle center
x_c = 0.40
y_c = 0.10

# Sweep radii
radii = [0.10, 0.15, 0.20, 0.25, 0.30]

device = "cpu"

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

output_folder = "figures_1D_radius"
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


# -----------------------------
# Run experiment
# -----------------------------
scenarios = []

for r in radii:
    print(f"Training for radius r = {r:.2f}")
    obs = [[x_c, y_c, r]]

    model = train_model(
        T=T,
        BC=BC,
        obs=obs,
        epochs=2000,
        lambda_phy=lambda_phy,
        lambda_obs=lambda_obs,
        lambda_omega=lambda_omega,
        N=N
    )

    t_np, k_np, kappa_max = compute_curvature(model, t_list, T, BC)
    x_np, y_np = get_trajectory(model, t_list, T, BC)

    scenarios.append({
        "r": r,
        "x_c": x_c,
        "y_c": y_c,
        "kappa_max": kappa_max,
        "x": x_np,
        "y": y_np,
        "t": t_np,
        "kappa": k_np
    })



# Plot 1: kappa_max vs R
fig1, ax1 = plt.subplots(figsize=(7,4))

Rs = np.array([s["r"] for s in scenarios])
Ks = np.array([s["kappa_max"] for s in scenarios])

ax1.plot(Rs, Ks, marker="o")
ax1.set_xlabel("Obstacle radius R")
ax1.set_ylabel(r"Max curvature $\kappa_{\max}$")
ax1.set_title(rf"$\kappa_{{\max}}$ vs $R$ (fixed center at ({x_c:.2f}, {y_c:.2f}))")
ax1.grid(True)

fig1.savefig(os.path.join(output_folder, "kappa_vs_radius.png"), dpi=300)
plt.close(fig1)


# Plot 2: trajectories (1xN)
fig2, axes = plt.subplots(1, len(radii), figsize=(4.5*len(radii), 4.5), sharey=True)

if len(radii) == 1:
    axes = [axes]

for j, s in enumerate(scenarios):
    ax = axes[j]

    ax.plot(s["x"], s["y"], linewidth=2)

    # obstacle
    circle = patches.Circle((x_c, y_c), s["r"], fill=True, alpha=0.15)
    edge   = patches.Circle((x_c, y_c), s["r"], fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.add_patch(edge)

    ax.plot(x_c, y_c, "x")

    ax.set_title(f"r={s['r']:.2f}\nκmax={s['kappa_max']:.3g}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

fig2.suptitle("Trajectories for different radii (fixed center)")
fig2.tight_layout(rect=[0,0,1,0.92])

fig2.savefig(os.path.join(output_folder, "trajectories_radius_sweep.png"), dpi=300)
plt.close(fig2)

print("Experiment 1 finished.")