# Parameter sensitivity: lambda_phys^nom and lambda_obs^nom.
# 4x3 grid:
#   rows 1-2 : circle    (lambda_phys^nom, lambda_obs^nom)
#   rows 3-4 : rectangle (lambda_phys^nom, lambda_obs^nom)
#   cols     : too low / baseline / too high
#
# Saves to:
#   results/loss_weight/sensitivity_obs.png

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
epochs = 2000

x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
v0     = 2
theta0 = 0
BC     = [x0, y0, xT, yT, v0, theta0]

# Circle obstacle
x_c_circ, y_c_circ, r_obs = 0.3, 0.1, 0.2
obs_circ = [[x_c_circ, y_c_circ, r_obs]]

# Rectangle obstacle (centered at y=0, intruding into the path)
x_c_rect, y_c_rect = 0.3, 0.0
w_rect = float(np.sqrt(2) * 0.2)
h_rect = float(np.sqrt(2) * 0.2)
xmin = x_c_rect - w_rect / 2
xmax = x_c_rect + w_rect / 2
ymin = y_c_rect - h_rect / 2
ymax = y_c_rect + h_rect / 2
obs_rect = [[xmin, xmax, ymin, ymax]]

# Baseline nominal weights
BASE = dict(
    lambda_phy   = 20,
    lambda_obs   = 50,
    lambda_v     = 0.2,
    lambda_omega = 2,
)

# Sweep configuration: each row varies one parameter
SWEEPS = [
    ("lambda_phy", r"\lambda_{\mathrm{phys}}^{\mathrm{nom}}", [2,  20,  200]),
    ("lambda_obs", r"\lambda_{\mathrm{obs}}^{\mathrm{nom}}",  [5,  50,  500]),
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
#  Train: 6 circle models + 6 rectangle models
# ------------------------------------------------------------------ #
def train_sweep(obs, label):
    results = []
    total   = len(SWEEPS) * 3
    counter = 0
    for param_name, _, values in SWEEPS:
        row_results = []
        for val in values:
            counter += 1
            weights = {**BASE, param_name: val}
            print(f"[{label}] [{counter}/{total}] {param_name} = {val}")
            model = train_model(
                T=T, BC=BC, obs=obs, epochs=epochs,
                lambda_phy   = weights["lambda_phy"],
                lambda_obs   = weights["lambda_obs"],
                lambda_v     = weights["lambda_v"],
                lambda_omega = weights["lambda_omega"],
                N=N,
            )
            x_np, y_np = get_trajectory(model, T, BC, N)
            row_results.append((x_np, y_np, val))
        results.append(row_results)
    return results

results_circ = train_sweep(obs_circ, "circle")
results_rect = train_sweep(obs_rect, "rectangle")


# ------------------------------------------------------------------ #
#  Axis limits — computed separately per obstacle type
# ------------------------------------------------------------------ #
def compute_limits(results, x_extents, y_extents, pad=0.08):
    all_x = np.concatenate(
        [x for row in results for x, _, _ in row] + [np.array(x_extents)]
    )
    all_y = np.concatenate(
        [y for row in results for _, y, _ in row] + [np.array(y_extents)]
    )
    return (all_x.min() - pad, all_x.max() + pad), (all_y.min() - pad, all_y.max() + pad)

xlim_circ, ylim_circ = compute_limits(
    results_circ,
    [x_c_circ - r_obs, x_c_circ + r_obs],
    [y_c_circ - r_obs, y_c_circ + r_obs],
)
xlim_rect, ylim_rect = compute_limits(
    results_rect,
    [xmin, xmax],
    [ymin, ymax],
)


# ------------------------------------------------------------------ #
#  Plot 4 × 3 grid
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(
    4, 3,
    figsize=(4.2 * 3, 4.2 * 4),
)

# Row separator line between circle and rectangle halves
fig.add_artist(plt.Line2D(
    [0.02, 0.98], [0.505, 0.505],
    transform=fig.transFigure,
    color="gray", linewidth=1.2, linestyle="--"
))

def draw_panel(ax, x_np, y_np, obstacle_patch_fn, cx, cy, annotation,
               title, xlim, ylim, show_ylabel, show_xlabel):

    ax.plot(x_np, y_np, linewidth=2.5, color="tab:blue")
    ax.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", linewidth=1, alpha=0.5, color="orange")

    fill_patch, edge_patch = obstacle_patch_fn()
    ax.add_patch(fill_patch)
    ax.add_patch(edge_patch)

    ax.plot(cx, cy, marker="x", markersize=7, mew=2, color="tab:blue")
    ax.plot([BC[0]], [BC[1]], marker="o", color="orange")
    ax.plot([BC[2]], [BC[3]], marker="o", color="green")

    ax.set_title(title, fontsize=14)
    ax.text(0.02, 0.06, annotation,
            transform=ax.transAxes, fontsize=11, va="bottom", ha="left")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    if show_ylabel:
        ax.set_ylabel("y")
    if show_xlabel:
        ax.set_xlabel("x")


# Rows 0-1: circle
for row_idx, ((param_name, latex_label, values), row_results) in enumerate(zip(SWEEPS, results_circ)):
    for col_idx, (x_np, y_np, val) in enumerate(row_results):
        ax = axes[row_idx, col_idx]

        def circ_patches():
            fill = patches.Circle((x_c_circ, y_c_circ), r_obs,
                                  facecolor="#c6d6e3", alpha=0.6, linewidth=0)
            edge = patches.Circle((x_c_circ, y_c_circ), r_obs,
                                  fill=False, edgecolor="black", linewidth=2)
            return fill, edge

        # Top row gets the column header (Too low / Baseline / Too high)
        if row_idx == 0:
            title = f"{COL_LABELS[col_idx]}\n${latex_label} = {val}$"
        else:
            title = f"${latex_label} = {val}$"

        draw_panel(
            ax, x_np, y_np,
            circ_patches,
            x_c_circ, y_c_circ,
            f"c=({x_c_circ:.2f},{y_c_circ:.2f})\nr={r_obs:.2f}",
            title,
            xlim_circ, ylim_circ,
            show_ylabel=(col_idx == 0),
            show_xlabel=False,
        )

# Rows 2-3: rectangle
for row_idx, ((param_name, latex_label, values), row_results) in enumerate(zip(SWEEPS, results_rect)):
    ax_row = row_idx + 2
    for col_idx, (x_np, y_np, val) in enumerate(row_results):
        ax = axes[ax_row, col_idx]

        def rect_patches():
            fill = patches.Rectangle((xmin, ymin), w_rect, h_rect,
                                     facecolor="#c6d6e3", alpha=0.6, linewidth=0)
            edge = patches.Rectangle((xmin, ymin), w_rect, h_rect,
                                     fill=False, edgecolor="black", linewidth=2)
            return fill, edge

        draw_panel(
            ax, x_np, y_np,
            rect_patches,
            x_c_rect, y_c_rect,
            f"c=({x_c_rect:.2f},{y_c_rect:.2f})\nw={w_rect:.2f}, h={h_rect:.2f}",
            f"${latex_label} = {val}$",
            xlim_rect, ylim_rect,
            show_ylabel=(col_idx == 0),
            show_xlabel=(ax_row == 3),
        )

# Row labels on the right-hand side
row_labels = [
    r"Circle: $\lambda_{\mathrm{phys}}^{\mathrm{nom}}$",
    r"Circle: $\lambda_{\mathrm{obs}}^{\mathrm{nom}}$",
    r"Rect: $\lambda_{\mathrm{phys}}^{\mathrm{nom}}$",
    r"Rect: $\lambda_{\mathrm{obs}}^{\mathrm{nom}}$",
]
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 2].annotate(
        label,
        xy=(1.02, 0.5), xycoords="axes fraction",
        fontsize=11, va="center", ha="left", rotation=270
    )

fig.suptitle(
    r"Trajectory sensitivity to $\lambda_{\mathrm{phys}}^{\mathrm{nom}}$ and "
    r"$\lambda_{\mathrm{obs}}^{\mathrm{nom}}$ (circle top, rectangle bottom)",
    fontsize=16, fontweight="bold", y=0.975
)
fig.subplots_adjust(top=0.93, hspace=0.45, wspace=0.25)

# ------------------------------------------------------------------ #
#  Save
# ------------------------------------------------------------------ #
output_folder = os.path.join("results", "loss_weight")
os.makedirs(output_folder, exist_ok=True)

save_path = os.path.join(output_folder, "sensitivity_obs.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {save_path}")