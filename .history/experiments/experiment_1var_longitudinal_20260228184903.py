# Vary obstacle x-position (longitudinal), keep the same radius.
# Saves:
#   figures_1D_longitudinal/curvature_vs_xc.png
#   figures_1D_longitudinal/trajectories_grid_1xN.png
#   figures_1D_longitudinal/scenario_results.csv

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PINNs_functions import *
from training_NN import train_model


def compute_curvature(model, t_list, T, BC):
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_t  = derivative(x, t_list)
    y_t  = derivative(y, t_list)
    x_tt = derivative(x_t, t_list)
    y_tt = derivative(y_t, t_list)

    eps = 1e-6
    kappa = (x_t * y_tt - y_t * x_tt) / ((x_t**2 + y_t**2 + eps) ** (3/2))
    return torch.max(torch.abs(kappa)).item()


def get_trajectory(model, t_list, T, BC):
    with torch.no_grad():
        nn_input = model(t_list)
        x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)
    return x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy()


# -----------------------------
# Experiment setup (edit these)
# -----------------------------
T = 1.0
N_train = 100
N_plot = 400

lambda_phys = 1
lambda_obs = 10
lambda_length = 0
lambda_omega = 0.0001

BC = [0, 0, 1, 0]  # Start (0,0) -> Goal (1,0)

# Fixed radius
r_fixed = 0.20

Delta = 0.10
y_c = r_fixed - Delta
if y_c < 0:
    raise ValueError(f"Delta={Delta} too large for r_fixed={r_fixed}: y_c would be negative.")


# Vary longitudinal obstacle position
x_positions = [0.25, 0.30, 0.35, 0.40, 0.45]


# -----------------------------
# Grids
# -----------------------------
t_train = torch.linspace(0.0, T, N_train, device=device).view(-1, 1)
t_train.requires_grad_(True)

t_plot = torch.linspace(0.0, T, N_plot, device=device).view(-1, 1)
t_plot.requires_grad_(True)

# -----------------------------
# Run experiments
# -----------------------------
results = []     # rows: [x_c, y_c, r, Delta, kappa_max]
scenarios = []   # store trajectories for plotting

print(f"Fixed radius r={r_fixed:.2f}")
print(f"Using y_c={y_c:.2f} (Delta={Delta:.2f} if Option A)")

for x_c in x_positions:
    print(f"Training for x_c = {x_c:.2f}")
    obs = [x_c, y_c, r_fixed]

    model = train_model(
        T=T,
        BC=BC,
        obs=obs,
        epochs=2000,
        lambda_phys=lambda_phys,
        lambda_obs=lambda_obs,
        lambda_length=lambda_length,
        lambda_omega=lambda_omega,
        N=N_train
    )

    kappa_max = compute_curvature(model, t_list=t_train, T=T, BC=BC)
    x_np, y_np = get_trajectory(model, t_list=t_plot, T=T, BC=BC)

    results.append([x_c, y_c, r_fixed, Delta, kappa_max])
    scenarios.append({
        "x_c": x_c, "y_c": y_c, "r": r_fixed, "Delta": Delta,
        "kappa_max": kappa_max, "x": x_np, "y": y_np
    })


# -----------------------------
# Plots: 
# -----------------------------
output_folder = "figures_1D_longitudinal"
os.makedirs(output_folder, exist_ok=True)

xs = np.array([row[0] for row in results], dtype=float)
ks = np.array([row[4] for row in results], dtype=float)

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(xs, ks, marker="o")
ax1.set_xlabel(r"Obstacle longitudinal position $x_c$")
ax1.set_ylabel(r"Max curvature $\kappa_{\max}$")
ax1.set_title(rf"$\kappa_{\max}$ vs $x_c$ (fixed $r={r_fixed:.2f}$, $y_c={y_c:.2f}$)")
ax1.grid(True, alpha=0.3)

path1 = os.path.join(output_folder, "curvature_vs_xc.png")
fig1.savefig(path1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {path1}")

# -----------------------------
# Plot 1: curvature vs x_c
# -----------------------------

# -----------------------------
# Plot 2: trajectories in 1×N grid (with obstacle)
# -----------------------------
# axis limits include obstacle extents so circles never get clipped
all_x = np.concatenate([s["x"] for s in scenarios] + [
    np.array([s["x_c"] - s["r"], s["x_c"] + s["r"]]) for s in scenarios
])
all_y = np.concatenate([s["y"] for s in scenarios] + [
    np.array([s["y_c"] - s["r"], s["y_c"] + s["r"]]) for s in scenarios
])

pad = 0.08
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)

fig2, axes = plt.subplots(1, len(x_positions), figsize=(4.2 * len(x_positions), 4.2), sharex=True, sharey=True)
if len(x_positions) == 1:
    axes = [axes]

for j, x_c in enumerate(x_positions):
    ax = axes[j]
    s = next(ss for ss in scenarios if abs(ss["x_c"] - x_c) < 1e-12)

    # trajectory
    ax.plot(s["x"], s["y"], linewidth=2)

    # reference straight line
    ax.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", linewidth=1, alpha=0.5)

    # obstacle: filled + outline
    fill = patches.Circle((s["x_c"], s["y_c"]), s["r"], fill=True, alpha=0.15, linewidth=0)
    edge = patches.Circle((s["x_c"], s["y_c"]), s["r"], fill=False, linewidth=2.5)
    ax.add_patch(fill)
    ax.add_patch(edge)

    # center marker
    ax.plot(s["x_c"], s["y_c"], marker="x", markersize=7, mew=2)

    # start/goal markers
    ax.plot([BC[0]], [BC[1]], marker="o")
    ax.plot([BC[2]], [BC[3]], marker="o")

    ax.set_title(f"x_c={x_c:.2f}\nκmax={s['kappa_max']:.3g}", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    ax.text(
        0.02, 0.06,
        f"c=({s['x_c']:.2f},{s['y_c']:.2f})\nr={s['r']:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left"
    )

fig2.suptitle(rf"Trajectories vs $x_c$ (fixed $r={r_fixed:.2f}$)", fontsize=14)
fig2.tight_layout(rect=[0, 0, 1, 0.92])

path2 = os.path.join(output_folder, "trajectories_grid_1xN.png")
fig2.savefig(path2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {path2}")