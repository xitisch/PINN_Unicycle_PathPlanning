import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.pinn.pinn_functions import *
from src.pinn.training_pinn import train_model


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

output_folder = os.path.join("results", "radius", "circle")
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


# Plot 3: trajectories (left) and curvature (right), one row per radius
fig2, axes = plt.subplots(
    len(scenarios), 2,
    figsize=(7, 3.2 * len(scenarios)),
    gridspec_kw={"width_ratios": [1.35, 0.9]}
)

if len(scenarios) == 1:
    axes = np.array([axes])

# Gemensamma axelgränser för trajektorierna
all_x = np.concatenate([s["x"] for s in scenarios] + [
    np.array([s["x_c"] - s["r"], s["x_c"] + s["r"]]) for s in scenarios
])
all_y = np.concatenate([s["y"] for s in scenarios] + [
    np.array([s["y_c"] - s["r"], s["y_c"] + s["r"]]) for s in scenarios
])

pad = 0.08
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)

# Gemensamma gränser för krökning
all_kappa = np.concatenate([s["kappa"] for s in scenarios])
kappa_min = np.min(all_kappa)
kappa_max = np.max(all_kappa)

for i, s in enumerate(scenarios):
    axL = axes[i, 0]
    axR = axes[i, 1]

    # vänster: trajektori
    axL.plot(s["x"], s["y"], linewidth=1.5)

    circle_fill = patches.Circle((x_c, y_c), s["r"], fill=True, alpha=0.15)
    circle_edge = patches.Circle((x_c, y_c), s["r"], fill=False, linewidth=1.2)
    axL.add_patch(circle_fill)
    axL.add_patch(circle_edge)

    axL.plot(x_c, y_c, "x", markersize=5)
    axL.plot(BC[0], BC[1], "o", markersize=3)
    axL.plot(BC[2], BC[3], "o", markersize=3)

    axL.set_xlim(*xlim)
    axL.set_ylim(*ylim)
    axL.set_aspect("equal", adjustable="box")
    axL.grid(True)
    axL.set_ylabel("y", fontsize=8)
    axL.tick_params(labelsize=7)
    axL.set_title(f"r={s['r']:.2f}\nκmax={s['kappa_max']:.2f}", fontsize=8)

    axL.text(
        0.03, 0.05,
        f"c=({x_c:.2f},{y_c:.2f})\nr={s['r']:.2f}",
        transform=axL.transAxes,
        fontsize=6,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5)
    )

    # höger: krökning
    axR.plot(s["t"], s["kappa"], linewidth=1.5)
    idx = np.argmax(np.abs(s["kappa"]))
    axR.plot(s["t"][idx], s["kappa"][idx], "o", markersize=3)

    axR.set_xlim(0.0, 1.0)
    axR.set_ylim(kappa_min - 0.05, kappa_max + 0.05)
    axR.grid(True)
    axR.set_ylabel(r"$\kappa(t)$", fontsize=8)
    axR.tick_params(labelsize=7)

    if i == len(scenarios) - 1:
        axL.set_xlabel("x", fontsize=8)
        axR.set_xlabel("t", fontsize=8)

fig2.suptitle(
    rf"Trajectories (left) and curvature over time (right) "
    rf"(fixed center at ({x_c:.2f}, {y_c:.2f}))",
    fontsize=10,
    fontweight="bold"
)

fig2.tight_layout(rect=[0, 0, 1, 0.98])
fig2.savefig(
    os.path.join(output_folder, "trajectories_and_curvature_radius.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig2)
