# This experiment fixes the intrusion, varies the 

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pinnlib.pinnsfunctions import *
from pinnlib.trainingNN import train_model

# Create 
def compute_curvature(model, t_list, T, BC):
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)
    x_tt = derivative(x_t, t_list)
    y_tt = derivative(y_t, t_list)

    safety = 1e-6
    kappa = (x_t * y_tt - y_t * x_tt) / ((x_t**2 + y_t**2 + safety)**(3/2))

    return torch.max(torch.abs(kappa)).item()

def get_trajectory(model, t_list, T, BC):
    """
    Output: returns the lists of x,y values in lists. 
    """
    with torch.no_grad():
        nn_input = model(t_list)
        x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)
    return x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy()

T = 1
N = 100

lambda_phys = 1
lambda_obs = 10
lambda_length = 0
lambda_omega = 0.0001

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

BC = [0, 0, 1, 0]

radii  = [0.10, 0.15, 0.20, 0.25]
Delta = 0.1   # one fixed intrusion value
x_positions = [0.30, 0.35, 0.40, 0.45]   # longitudinal distance

results = []
scenarios = []

for r in radii:
    # for y=0 reference line: d_perp = |y_c|, so y_c = r - Delta
    y_c = r - Delta
    if y_c < 0:
        raise ValueError(f"Delta={Delta} is too large for r={r}: would require y_c < 0")

    print(f"\nRadius r={r:.2f}, fixed intrusion Delta={Delta:.2f} => y_c={y_c:.2f}")

    for x_c in x_positions:
        obs = [x_c, y_c, r]

        model = train_model(
            T=T,
            BC=BC,
            obs=obs,
            epochs=2000,
            lambda_phys=lambda_phys,
            lambda_obs=lambda_obs,
            lambda_length=lambda_length,
            lambda_omega=lambda_omega,
            N=N
        )

        kappa_max = compute_curvature(model, t_list=t_list, T=T, BC=BC)
        x_np, y_np = get_trajectory(model, t_list, T=T, BC=BC)

        results.append([r, x_c, y_c, Delta, kappa_max])
        scenarios.append({
            "r": r, "x_c": x_c, "y_c": y_c, "Delta": Delta,
            "kappa_max": kappa_max, "x": x_np, "y": y_np
        })

output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

heatmap = np.zeros((len(radii), len(x_positions)))
for r, x_c, y_c, Delta_val, kappa in results:
    i = radii.index(r)
    j = x_positions.index(x_c)
    heatmap[i, j] = kappa

fig_hm, ax_hm = plt.subplots(figsize=(6.5, 5.5))
im = ax_hm.imshow(
    heatmap,
    origin="lower",
    aspect="auto",
    extent=[min(x_positions), max(x_positions), min(radii), max(radii)]
)
fig_hm.colorbar(im, ax=ax_hm, label=r"Max curvature $\kappa_{\max}$")
ax_hm.set_xlabel(r"Obstacle longitudinal position $x_c$")
ax_hm.set_ylabel(r"Obstacle radius $r$")
ax_hm.set_title(rf"Curvature heatmap (fixed intrusion $\Delta={Delta:.2f}$)")

heatmap_path = os.path.join(output_folder, "curvature_heatmap.png")
fig_hm.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close(fig_hm)
print(f"Saved: {heatmap_path}")



# 4×4 Trajectory grid (rows=radii, cols=x_positions)
# include obstacle extents in axis limits so circles never get clipped
all_x = np.concatenate([s["x"] for s in scenarios] + [
    np.array([s["x_c"] - s["r"], s["x_c"] + s["r"]]) for s in scenarios
])
all_y = np.concatenate([s["y"] for s in scenarios] + [
    np.array([s["y_c"] - s["r"], s["y_c"] + s["r"]]) for s in scenarios
])

pad = 0.08
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)

lookup = {(s["r"], s["x_c"]): s for s in scenarios}

fig_grid, axes = plt.subplots(len(radii), len(x_positions), figsize=(14, 14), sharex=True, sharey=True)

for i, r in enumerate(radii):
    for j, x_c in enumerate(x_positions):
        ax = axes[i, j]
        s = lookup[(r, x_c)]

        # trajectory
        ax.plot(s["x"], s["y"], linewidth=2)

        # baseline straight line from start to goal (reference)
        ax.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", linewidth=1, alpha=0.5)

        # obstacle: filled + outline
        fill = patches.Circle((s["x_c"], s["y_c"]), s["r"], fill=True, alpha=0.15, linewidth=0)
        edge = patches.Circle((s["x_c"], s["y_c"]), s["r"], fill=False, linewidth=2.5)
        ax.add_patch(fill)
        ax.add_patch(edge)

        # obstacle center marker
        ax.plot(s["x_c"], s["y_c"], marker="x", markersize=6, mew=2)

        # start/goal markers
        ax.plot([BC[0]], [BC[1]], marker="o")
        ax.plot([BC[2]], [BC[3]], marker="o")

        # titles + small annotation
        ax.set_title(f"r={r}, x_c={x_c}\nκmax={s['kappa_max']:.3g}", fontsize=10)
        ax.text(
            0.02, 0.06,
            f"c=({s['x_c']:.2f},{s['y_c']:.2f})\nr={s['r']:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left"
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

fig_grid.suptitle(
    rf"PINN Trajectories for 16 Scenarios (4×4 grid) — fixed intrusion $\Delta={Delta:.2f}$",
    fontsize=16
)
fig_grid.tight_layout(rect=[0, 0, 1, 0.97])

grid_path = os.path.join(output_folder, "trajectories_grid_4x4.png")
fig_grid.savefig(grid_path, dpi=300, bbox_inches="tight")
plt.close(fig_grid)
print(f"Saved: {grid_path}")