# Vary obstacle x-position (longitudinal), keep rectangle size fixed.
# Saves:
#   figures_1D_longitudinal/curvature_vs_xc.png
#   figures_1D_longitudinal/trajectories_grid_Nx1.png
#   figures_1D_longitudinal/scenario_results.csv

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pinnlib.pinn_functions import *
from pinnlib.training_pinn import train_model


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
N = 100

lambda_phys = 1
lambda_obs = 10
lambda_length = 0
lambda_omega = 0.0001

BC = [0, 0, 1, 0]  # Start (0,0) -> Goal (1,0)

# Rectangle dimensions
width = 0.2
height = 0.2

Delta = 0.10
y_c = height/2 - Delta

if y_c < 0:
    raise ValueError(rf"Delta={Delta} too large for $\mathrm{{fixed}}\ w={width:.2f},\ h={height:.2f}$: y_c would be negative.")

# Vary longitudinal obstacle position
x_positions = [0.25, 0.30, 0.35, 0.40, 0.45]


# Grids
t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)


# Run experiments
scenarios = []   # store trajectories for plotting

print(f"Fixed height={height:.2f} and width={width:.2f}")
print(f"Using y_c={y_c:.2f} (Delta={Delta:.2f})")

for x_c in x_positions:
    print(f"Training for x_c = {x_c:.2f}")

    xmin = x_c - width/2
    xmax = x_c + width/2
    ymin = y_c - height/2
    ymax = y_c + height/2

    obs = [[xmin, xmax, ymin, ymax]]

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

    t_np, k_np, kappa_max = compute_curvature(model, t_list=t_list, T=T, BC=BC)
    x_np, y_np = get_trajectory(model, t_list=t_list, T=T, BC=BC)

    scenarios.append({
    "model": model,
    "xmin": xmin,
    "xmax": xmax,
    "ymin": ymin,
    "ymax": ymax,
    "x_c": x_c,
    "y_c": y_c, 
    "Delta": Delta,
    "kappa_max": kappa_max,
    "x": x_np, "y": y_np,
    "t": t_np, "kappa": k_np
    })


# Plot 1: curvature vs x_c
output_folder = os.path.join("figures", "longitudinal", "rectangle")
os.makedirs(output_folder, exist_ok=True)

xs = np.array([s["x_c"] for s in scenarios], dtype=float)
ks = np.array([s["kappa_max"] for s in scenarios], dtype=float)

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(xs, ks, marker="o")
ax1.set_xlabel(r"Obstacle center position $x_c$")
ax1.set_ylabel(r"Max curvature $\kappa_{\max}$")
ax1.set_title(
    rf"$\kappa_{{\max}}$ vs $x_c$  "
    rf"($\mathrm{{fixed}}\ w={width:.2f},\ h={height:.2f}$)"
)
ax1.grid(True, alpha=0.3)

path1 = os.path.join(output_folder, "curvature_vs_xc.png")
fig1.savefig(path1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {path1}")


# Plot 2: trajectories in 1×N grid
# axis limits include obstacle extents so rectangles never get clipped
all_x = np.concatenate([s["x"] for s in scenarios] + [
    np.array([s["xmin"], s["xmax"]]) for s in scenarios
])
all_y = np.concatenate([s["y"] for s in scenarios] + [
    np.array([s["ymin"], s["ymax"]]) for s in scenarios
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
    ax.plot(s["x"], s["y"], linewidth=2.5)

    # reference straight line
    ax.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", linewidth=1, alpha=0.5)

    # obstacle: filled + outline
    fill = patches.Rectangle(
        (s["xmin"], s["ymin"]),
        width,
        height,
        fill=True,
        alpha=0.15,
        linewidth=0
    )

    edge = patches.Rectangle(
        (s["xmin"], s["ymin"]),
        width,
        height,
        fill=False,
        linewidth=2.5
    )
    ax.add_patch(fill)
    ax.add_patch(edge)

    # center marker
    ax.plot(s["x_c"], s["y_c"], marker="x", markersize=7, mew=2)

    # start/goal markers
    ax.plot([BC[0]], [BC[1]], marker="o")
    ax.plot([BC[2]], [BC[3]], marker="o")

    ax.set_title(f"x_c={x_c:.2f}\nκ_max={s['kappa_max']:.3g}", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    ax.text(
        0.02, 0.06,
        f"w={width:.2f}\nh={height:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        ha="left"
    )

fig2.suptitle(rf"Trajectories for different $x_c$ ($\mathrm{{fixed}}\ w={width:.2f},\ h={height:.2f}$)", fontsize=18)
fig2.tight_layout(rect=[0, 0, 1, 0.92])

path2 = os.path.join(output_folder, "trajectories_grid_1xN.png")
fig2.savefig(path2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {path2}")


""""""
# Build lookup by x_c
lookup = {s["x_c"]: s for s in scenarios}

# Axis limits for trajectory row (include obstacle extents)
all_x = np.concatenate([s["x"] for s in scenarios] + [
    np.array([s["xmin"], s["xmax"]]) for s in scenarios
])
all_y = np.concatenate([s["y"] for s in scenarios] + [
    np.array([s["ymin"], s["ymax"]]) for s in scenarios
])

pad = 0.08
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)

# Common y-limit for curvature plots
k_all = np.concatenate([np.abs(s["kappa"]) for s in scenarios])
k_ylim = (0.0, 1.10 * k_all.max())

fig, axes = plt.subplots(
    len(x_positions), 2,
    figsize=(8, 3.5 * len(x_positions)),
    sharex="col"
)

if len(x_positions) == 1:
    axes = np.array(axes).reshape(2, 1)

for j, x_c in enumerate(x_positions):
    s = lookup[x_c]

    # Row 1: trajectory
    ax_traj = axes[j, 0]
    ax_traj.plot(s["x"], s["y"], linewidth=2.5)

    # reference straight line
    ax_traj.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", linewidth=1, alpha=0.5)

    # obstacle (filled + outline)
    width  = s["xmax"] - s["xmin"]
    height = s["ymax"] - s["ymin"]

    fill = patches.Rectangle((s["xmin"], s["ymin"]), width, height, fill=True, alpha=0.15, linewidth=0)
    edge = patches.Rectangle((s["xmin"], s["ymin"]), width, height, fill=False, linewidth=2.5)
    ax_traj.add_patch(fill)
    ax_traj.add_patch(edge)

    # obstacle center marker
    ax_traj.plot(s["x_c"], s["y_c"], marker="x", markersize=7, mew=2)

    # start/goal markers
    ax_traj.plot([BC[0]], [BC[1]], marker="o")
    ax_traj.plot([BC[2]], [BC[3]], marker="o")

    ax_traj.set_title(f"x_c={x_c:.2f}\nκmax={s['kappa_max']:.3g}", fontsize=14)
    ax_traj.set_xlim(*xlim)
    ax_traj.set_ylim(*ylim)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True, alpha=0.25)

    ax_traj.text(
        0.02, 0.06,
        f"w={width:.2f}\nh={height:.2f}",
        transform=ax_traj.transAxes,
        fontsize=11,
        va="bottom",
        ha="left"
    )

    # Row 2: curvature over time
    ax_k = axes[j, 1]
    t_np = s["t"]
    k_np = s["kappa"]

    ax_k.plot(t_np, np.abs(k_np), linewidth=2.5)
    ax_k.set_ylim(*k_ylim)
    ax_k.grid(True, alpha=0.25)

    # mark the peak
    idx = np.argmax(np.abs(k_np))
    ax_k.plot([t_np[idx]], [np.abs(k_np[idx])], marker="o")

    ax_traj.set_ylabel("y")
    ax_k.set_ylabel(r"$|\kappa(t)|$")
    if j == len(x_positions) - 1:
        ax_k.set_xlabel("t")

fig.suptitle(rf"Trajectories (top) and curvature over time (bottom) ($\mathrm{{fixed}}\ w={width:.2f},\ h={height:.2f}$)", fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(output_folder, "trajectories_and_kappa_5x2.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_path}")