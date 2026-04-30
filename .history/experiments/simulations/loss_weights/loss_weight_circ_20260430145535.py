# Parameter sensitivity study: vary one loss weight at a time.
# Produces a 4x3 grid of trajectory plots:
#   rows = lambda_phy, lambda_obs, lambda_v, lambda_omega
#   cols = too low / baseline / too high
#
# Saves to:
#   results/loss_weight/sensitivity_4x3.png

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.pinn.pinn_functions import *
from src.pinn.train_pinn import train_model

# ------------------------------------------------------------------ #
#  Shared experiment settings
# ------------------------------------------------------------------ #
T      = 1.0
N      = 400
epochs = 2000          # qualitative study — reduced from 3000

x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
v0     = 2
theta0 = 0
BC     = [x0, y0, xT, yT, v0, theta0]

# One fixed circle obstacle used in every panel
x_c, y_c, r_obs = 0.3, 0.1, 0.2
obs = [[x_c, y_c, r_obs]]

# Baseline weights
BASE = dict(
    lambda_phy   = 20,
    lambda_obs   = 50,
    lambda_v     = 0.2,
    lambda_omega = 2,
)

# Sweep: (param_name, [low, baseline, high])
SWEEPS = [
    ("lambda_phy",   [2,    20,  200 ]),
    ("lambda_obs",   [5,    50,  500 ]),
    ("lambda_v",     [0.02, 0.2, 2.0 ]),
    ("lambda_omega", [0.2,  2,   20  ]),
]

COL_LABELS = ["Too low", "Baseline", "Too high"]

# ------------------------------------------------------------------ #
#  Helper: extract trajectory
# ------------------------------------------------------------------ #
def get_trajectory(model, T, BC, N):
    t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
    with torch.no_grad():
        nn_input = model(t_list)
        x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)
    return x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy()


# ------------------------------------------------------------------ #
#  Train all 12 models
# ------------------------------------------------------------------ #
# results[row][col] = (x_np, y_np, val)
results = []

total   = len(SWEEPS) * 3
counter = 0

for param_name, values in SWEEPS:
    row_results = []
    for val in values:
        counter += 1
        weights = {**BASE, param_name: val}
        print(f"[{counter}/{total}] {param_name} = {val}")

        model = train_model(
            T=T,
            BC=BC,
            obs=obs,
            epochs=epochs,
            lambda_phy   = weights["lambda_phy"],
            lambda_obs   = weights["lambda_obs"],
            lambda_v     = weights["lambda_v"],
            lambda_omega = weights["lambda_omega"],
            N=N,
        )

        x_np, y_np = get_trajectory(model, T, BC, N)
        row_results.append((x_np, y_np, val))

    results.append(row_results)


# ------------------------------------------------------------------ #
#  Compute shared axis limits from all trajectories + obstacle extents
# ------------------------------------------------------------------ #
all_x = np.concatenate(
    [x for row in results for x, _, _ in row] +
    [np.array([x_c - r_obs, x_c + r_obs])]
)
all_y = np.concatenate(
    [y for row in results for _, y, _ in row] +
    [np.array([y_c - r_obs, y_c + r_obs])]
)

pad  = 0.08
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)


# ------------------------------------------------------------------ #
#  Plot 4 × 3 grid
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(
    4, 3,
    figsize=(4.2 * 3, 4.2 * 4),
    sharex=True, sharey=True
)

for row_idx, ((param_name, values), row_results) in enumerate(zip(SWEEPS, results)):
    for col_idx, (x_np, y_np, val) in enumerate(row_results):
        ax = axes[row_idx, col_idx]

        # Trajectory
        ax.plot(x_np, y_np, linewidth=2.5)

        # Reference straight line
        ax.plot([BC[0], BC[2]], [BC[1], BC[3]],
                linestyle="--", linewidth=1, alpha=0.5)

        # Obstacle: filled + outline
        fill = patches.Circle((x_c, y_c), r_obs, fill=True, alpha=0.15, linewidth=0)
        edge = patches.Circle((x_c, y_c), r_obs, fill=False, linewidth=2.5)
        ax.add_patch(fill)
        ax.add_patch(edge)

        # Obstacle center marker
        ax.plot(x_c, y_c, marker="x", markersize=7, mew=2)

        # Start / goal markers
        ax.plot([BC[0]], [BC[1]], marker="o")
        ax.plot([BC[2]], [BC[3]], marker="o")

        # Title: column header on top row, parameter value below
        if row_idx == 0:
            ax.set_title(f"{COL_LABELS[col_idx]}\n{param_name}={val}", fontsize=14)
        else:
            ax.set_title(f"{param_name}={val}", fontsize=14)

        # Annotation: top-right, no bbox — matching longitudinal style
        ax.text(
            0.02, 0.06,
            f"c=({x_c:.2f},{y_c:.2f})\nr={r_obs:.2f}",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            ha="right"
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        if col_idx == 0:
            ax.set_ylabel("y")
        if row_idx == 3:
            ax.set_xlabel("x")

fig.suptitle(
    "Loss weight sensitivity: trajectories under low / baseline / high settings",
    fontsize=18,
    fontweight="bold",
    y=0.975
)
fig.subplots_adjust(top=0.93, hspace=0.4)

# ------------------------------------------------------------------ #
#  Save
# ------------------------------------------------------------------ #
output_folder = os.path.join("results", "loss_weight", "circle")
os.makedirs(output_folder, exist_ok=True)

save_path = os.path.join(output_folder, "sensitivity_4x3.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {save_path}")