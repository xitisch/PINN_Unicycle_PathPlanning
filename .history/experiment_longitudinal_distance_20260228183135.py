import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from PINNs_functions import *
from training_NN import train_model

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

radii  = [0.10, 0.15, 0.20, 0.25]
Delta = 0.05   # one fixed intrusion value
x_positions = [0.20, 0.35, 0.50, 0.65]   # longitudinal distance

T = 1
N = 100

lambda_phys = 1
lambda_obs = 10
lambda_length = 0
lambda_omega = 0.0001

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

BC = [0, 0, 1, 0]

results = []
scenarios = []

for r in radii:
    print(f"Now: r = {r} out of {radii[-1]}")
    for y_c in y_positions:

        print(f"Now: y_c = {y_c} out of {y_positions[-1]}")
        obs = [0.5, y_c, r]

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
        
        results.append([r, y_c, kappa_max])
        scenarios.append({
            "r": r,
            "y_c": y_c,
            "x_c": obs[0],
            "kappa_max": kappa_max,
            "x": x_np,
            "y": y_np,
        })


output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

fig, axes = plt.subplots(len(radii), len(y_positions), figsize=(14, 14), sharex=True, sharey=True)

# optional: consistent limits so all panels are comparable
all_x = np.concatenate([s["x"] for s in scenarios] + [
    np.array([s["x_c"] - s["r"], s["x_c"] + s["r"]]) for s in scenarios
])
all_y = np.concatenate([s["y"] for s in scenarios] + [
    np.array([s["y_c"] - s["r"], s["y_c"] + s["r"]]) for s in scenarios
])

pad = 0.08  # slightly larger pad so circles are comfortably visible
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)

# Put scenarios into a lookup by (r, y_c) so the grid is easy to fill
lookup = {(s["r"], s["y_c"]): s for s in scenarios}

for i, r in enumerate(radii):
    for j, y_c in enumerate(y_positions):
        ax = axes[i, j]
        s = lookup[(r, y_c)]

        # trajectory
        ax.plot(s["x"], s["y"], linewidth=2)

        # obstacle circle (filled + boundary) + center marker
        fill = patches.Circle(
            (s["x_c"], s["y_c"]), s["r"],
            fill=True, alpha=0.15, linewidth=0
        )
        edge = patches.Circle(
            (s["x_c"], s["y_c"]), s["r"],
            fill=False, linewidth=2.5
        )
        ax.add_patch(fill)
        ax.add_patch(edge)

        # obstacle center marker
        ax.plot(s["x_c"], s["y_c"], marker="x", markersize=6, mew=2)

        # annotate obstacle parameters (small, unobtrusive)
        ax.text(
            0.02, 0.06,
            f"c=({s['x_c']:.2f},{s['y_c']:.2f})\nr={s['r']:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left"
        )

        # start & goal markers (optional)
        ax.plot([BC[0], BC[2]], [BC[1], BC[3]], marker='o', linestyle='None')

        # title with parameters
        ax.set_title(f"r={r}, y_c={y_c}\nκmax={s['kappa_max']:.3g}", fontsize=10)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

fig.suptitle("PINN Trajectories for 16 Scenarios (4×4 grid)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])

grid_path = os.path.join(output_folder, "trajectories_grid_4x4.png")
fig.savefig(grid_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {grid_path}")


heatmap = np.zeros((len(radii), len(y_positions)))

for r, y_c, kappa in results:
    i = radii.index(r)
    j = y_positions.index(y_c)
    heatmap[i, j] = kappa

plt.figure(figsize=(6, 5))

im = plt.imshow(
    heatmap,
    origin='lower',
    aspect='auto',
    extent=[
        min(y_positions), max(y_positions),
        min(radii), max(radii)
    ]
)

plt.colorbar(im, label="Max curvature κ")
plt.xlabel("Obstacle y_position")
plt.ylabel("Obstacle radius")
plt.title("Curvature heatmap")

plt.show()

output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

for i, r in enumerate(radii):

    plt.figure(figsize=(6, 4))

    plt.plot(
        y_positions,
        heatmap[i, :],
        marker='o',
        linewidth=2
    )

    plt.xlabel("Obstacle y-position")
    plt.ylabel("Max curvature κ")
    plt.title(f"Curvature vs displacement (r = {r})")
    plt.grid(True)

    heatmap_path = os.path.join(output_folder, "curvature_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path}")

plt.figure(figsize=(6, 4))

for i, r in enumerate(radii):
    plt.plot(
        y_positions,
        heatmap[i, :],
        marker='o',
        label=f"r = {r}"
    )

plt.xlabel("Obstacle y-position")
plt.ylabel("Max curvature κ")
plt.title("Curvature vs displacement for different radii")
plt.legend()
plt.grid(True)

plt.savefig("figures/curvature_all_radii.png", dpi=300, bbox_inches='tight')
plt.close()