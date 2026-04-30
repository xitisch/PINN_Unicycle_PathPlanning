# Parameter sensitivity: lambda_v and lambda_omega.
# 4x3 grid:
#   rows 1-2 : circle    (lambda_v, lambda_omega)
#   rows 3-4 : rectangle (lambda_v, lambda_omega)
#   cols     : too low / baseline / too high
#
# Each panel shows v(t) and omega(t) on a dual y-axis.
#
# Saves to:
#   results/loss_weight/sensitivity_controls.png

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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

# Rectangle obstacle
x_c_rect, y_c_rect = 0.3, 0.0
w_rect = float(np.sqrt(2) * 0.2)
h_rect = float(np.sqrt(2) * 0.2)
xmin = x_c_rect - w_rect / 2
xmax = x_c_rect + w_rect / 2
ymin = y_c_rect - h_rect / 2
ymax = y_c_rect + h_rect / 2
obs_rect = [[xmin, xmax, ymin, ymax]]

# Baseline weights
BASE = dict(
    lambda_phy   = 20,
    lambda_obs   = 50,
    lambda_v     = 0.02,
    lambda_omega = 2,
)

# Only sweep these two parameters
SWEEPS = [
    ("lambda_v",     [0.002, 0.02, 0.2]),
    ("lambda_omega", [0.2,  2,   20 ]),
]

COL_LABELS = ["Too low", "Baseline", "Too high"]

# ------------------------------------------------------------------ #
#  Helper: extract v(t) and omega(t)
# ------------------------------------------------------------------ #
def get_controls(model, T, BC, N):
    t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
    with torch.no_grad():
        nn_input = model(t_list)
        _, _, _, v, omega = hard_bc_transform(t_list, nn_input, T, BC)
    t_np     = t_list.squeeze().cpu().numpy()
    v_np     = v.squeeze().cpu().numpy()
    omega_np = omega.squeeze().cpu().numpy()
    return t_np, v_np, omega_np


# ------------------------------------------------------------------ #
#  Train: 6 circle models + 6 rectangle models
# ------------------------------------------------------------------ #
def train_sweep(obs, label):
    results = []
    total   = len(SWEEPS) * 3
    counter = 0
    for param_name, values in SWEEPS:
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
            t_np, v_np, omega_np = get_controls(model, T, BC, N)
            row_results.append((t_np, v_np, omega_np, val))
        results.append(row_results)
    return results

results_circ = train_sweep(obs_circ, "circle")
results_rect = train_sweep(obs_rect, "rectangle")


# ------------------------------------------------------------------ #
#  Shared y-limits per signal, computed across all runs
# ------------------------------------------------------------------ #
def signal_limits(results, pad=0.1):
    all_v     = np.concatenate([v     for row in results for _, v, _, _     in row])
    all_omega = np.concatenate([omega for row in results for _, _, omega, _ in row])
    def lim(arr):
        lo, hi = arr.min(), arr.max()
        span = max(hi - lo, 1e-3)
        return lo - pad * span, hi + pad * span
    return lim(all_v), lim(all_omega)

vlim_circ, olim_circ = signal_limits(results_circ)
vlim_rect, olim_rect = signal_limits(results_rect)


# ------------------------------------------------------------------ #
#  Plot 4 × 3 grid  (dual y-axis per panel)
# ------------------------------------------------------------------ #
COLOR_V     = "tab:blue"
COLOR_OMEGA = "tab:red"

fig, axes = plt.subplots(
    4, 3,
    figsize=(4.2 * 3, 4.0 * 4),
)

# Dashed separator between circle and rectangle halves
fig.add_artist(plt.Line2D(
    [0.02, 0.98], [0.505, 0.505],
    transform=fig.transFigure,
    color="gray", linewidth=1.2, linestyle="--"
))

def draw_panel(ax, t_np, v_np, omega_np, title,
               vlim, olim, show_xlabel):

    # Left axis — v(t)
    ax.plot(t_np, v_np, color=COLOR_V, linewidth=2.0, label=r"$v(t)$")
    ax.set_ylabel(r"$v(t)$", color=COLOR_V, fontsize=10)
    ax.tick_params(axis="y", labelcolor=COLOR_V)
    ax.set_ylim(*vlim)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.0, T)

    # Right axis — omega(t)
    ax2 = ax.twinx()
    ax2.plot(t_np, omega_np, color=COLOR_OMEGA, linewidth=2.0,
             linestyle="--", label=r"$\omega(t)$")
    ax2.set_ylabel(r"$\omega(t)$", color=COLOR_OMEGA, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=COLOR_OMEGA)
    ax2.set_ylim(*olim)

    ax.set_title(title, fontsize=14)

    if show_xlabel:
        ax.set_xlabel("t")


# Rows 0-1: circle
for row_idx, ((param_name, values), row_results) in enumerate(zip(SWEEPS, results_circ)):
    vlim, olim = vlim_circ, olim_circ
    for col_idx, (t_np, v_np, omega_np, val) in enumerate(row_results):
        ax = axes[row_idx, col_idx]

        title = (f"{COL_LABELS[col_idx]}\n{param_name}={val}"
                 if row_idx == 0 else f"{param_name}={val}")

        draw_panel(ax, t_np, v_np, omega_np, title,
                   vlim, olim, show_xlabel=False)

# Rows 2-3: rectangle
for row_idx, ((param_name, values), row_results) in enumerate(zip(SWEEPS, results_rect)):
    ax_row = row_idx + 2
    vlim, olim = vlim_rect, olim_rect
    for col_idx, (t_np, v_np, omega_np, val) in enumerate(row_results):
        ax = axes[ax_row, col_idx]

        draw_panel(ax, t_np, v_np, omega_np,
                   f"{param_name}={val}",
                   vlim, olim,
                   show_xlabel=(ax_row == 3))

# Row labels
row_labels = [
    r"Circle: $\lambda_{v}$",
    r"Circle: $\lambda_{\omega}$",
    r"Rect: $\lambda_{v}$",
    r"Rect: $\lambda_{\omega}$",
]
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 2].annotate(
        label,
        xy=(1.15, 0.5), xycoords="axes fraction",
        fontsize=11, va="center", ha="left", rotation=270
    )

# Shared legend (top-right, outside grid)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=COLOR_V,     linewidth=2.0,              label=r"$v(t)$"),
    Line2D([0], [0], color=COLOR_OMEGA, linewidth=2.0, linestyle="--", label=r"$\omega(t)$"),
]
fig.legend(handles=legend_handles, loc="upper right",
           bbox_to_anchor=(0.99, 0.99), fontsize=11, framealpha=0.9)

fig.suptitle(
    r"Control sensitivity: $\lambda_{v}$ and $\lambda_{\omega}$ "
    "(circle top, rectangle bottom)",
    fontsize=16, fontweight="bold", y=0.975
)
fig.subplots_adjust(top=0.93, hspace=0.5, wspace=0.45)

# ------------------------------------------------------------------ #
#  Save
# ------------------------------------------------------------------ #
output_folder = os.path.join("results", "loss_weight")
os.makedirs(output_folder, exist_ok=True)

save_path = os.path.join(output_folder, "sensitivity_controls.png")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {save_path}")