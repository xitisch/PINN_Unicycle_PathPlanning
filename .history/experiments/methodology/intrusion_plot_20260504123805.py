# Illustrate the effective intrusion distance Delta for circular and 
# rectangular obstacles. The path is the straight line from start to goal 
# along y=0. Delta is the vertical distance the obstacle penetrates 
# below the path (i.e. into y<0).
#
# Saves:
#   results/intrusion_illustration/effective_intrusion.png

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


output_folder = os.path.join("results", "thesis_figures")
os.makedirs(output_folder, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
BC = [x0, y0, xT, yT]

x_c = 0.5
Delta = 0.10

# Circle: position so bottom edge is at y = -Delta below the path
r_circ   = 0.20
y_c_circ = r_circ - Delta

# Rectangle: square with corner-to-center distance equal to circle radius
w_rect   = h_rect = float(2 * r_circ * np.sin(np.pi / 4))   # = r_circ * sqrt(2)
y_c_rect = h_rect / 2 - Delta


# ============================================================
# Axis limits (shared across both subplots)
# ============================================================
pad = 0.08

obs_x_extents = [x_c - r_circ, x_c + r_circ,
                 x_c - w_rect/2, x_c + w_rect/2]
obs_y_extents = [y_c_circ - r_circ, y_c_circ + r_circ,
                 y_c_rect - h_rect/2, y_c_rect + h_rect/2]

xlim = (min(BC[0], BC[2], min(obs_x_extents)) - pad,
        max(BC[0], BC[2], max(obs_x_extents)) + pad)
ylim = (min(BC[1], BC[3], min(obs_y_extents)) - pad,
        max(BC[1], BC[3], max(obs_y_extents)) + pad)


# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=False)


# ---- Subplot 1: Circle ----
ax = axes[0]

# reference straight line
ax.plot([BC[0], BC[2]], [BC[1], BC[3]],
        linestyle="--", linewidth=1, alpha=0.5)

# obstacle: filled + outline (matching circle.py style)
fill = patches.Circle((x_c, y_c_circ), r_circ,
                      fill=True, alpha=0.15, linewidth=0)
edge = patches.Circle((x_c, y_c_circ), r_circ,
                      fill=False, linewidth=2.5)
ax.add_patch(fill)
ax.add_patch(edge)

# center marker
ax.plot(x_c, y_c_circ, marker="x", markersize=7, mew=2)

# start/goal markers
ax.plot([BC[0]], [BC[1]], marker="o")
ax.plot([BC[2]], [BC[3]], marker="o")

# intrusion arrow: from path (y=0) down to bottom of circle (y=-Delta)
arrow_x = x_c + r_circ + 0.05
ax.annotate(
    '', xy=(arrow_x, -Delta), xytext=(arrow_x, 0),
    arrowprops=dict(arrowstyle='<->', color='red', lw=2)
)
ax.text(
    arrow_x + 0.03, -Delta / 2,
    rf'$\Delta = {Delta:.2f}$',
    fontsize=13, color='red', va='center'
)

ax.set_title("Circular Obstacle", fontsize=14)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.25)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.text(
    0.97, 0.97,
    f"c=({x_c:.2f},{y_c_circ:.2f})\nr={r_circ:.2f}",
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    ha="right",
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              alpha=0.8, edgecolor='gray')
)


# ---- Subplot 2: Rectangle ----
ax = axes[1]

# reference straight line
ax.plot([BC[0], BC[2]], [BC[1], BC[3]],
        linestyle="--", linewidth=1, alpha=0.5)

# obstacle: filled + outline
fill = patches.Rectangle(
    (x_c - w_rect/2, y_c_rect - h_rect/2), w_rect, h_rect,
    fill=True, alpha=0.15, linewidth=0
)
edge = patches.Rectangle(
    (x_c - w_rect/2, y_c_rect - h_rect/2), w_rect, h_rect,
    fill=False, linewidth=2.5
)
ax.add_patch(fill)
ax.add_patch(edge)

# center marker
ax.plot(x_c, y_c_rect, marker="x", markersize=7, mew=2)

# start/goal markers
ax.plot([BC[0]], [BC[1]], marker="o")
ax.plot([BC[2]], [BC[3]], marker="o")

# intrusion arrow: from path (y=0) down to bottom of rectangle (y=-Delta)
arrow_x = x_c + w_rect/2 + 0.05
ax.annotate(
    '', xy=(arrow_x, -Delta), xytext=(arrow_x, 0),
    arrowprops=dict(arrowstyle='<->', color='red', lw=2)
)
ax.text(
    arrow_x + 0.03, -Delta / 2,
    rf'$\Delta = {Delta:.2f}$',
    fontsize=13, color='red', va='center'
)

ax.set_title("Rectangular Obstacle", fontsize=14)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.25)
ax.set_xlabel("x")

ax.text(
    0.97, 0.97,
    f"c=({x_c:.2f},{y_c_rect:.2f})\nw={w_rect:.2f}, h={h_rect:.2f}",
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    ha="right",
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              alpha=0.8, edgecolor='gray')
)


# ============================================================
# Title and save
# ============================================================
fig.suptitle(
    rf"Example of Simulation Setup",
    fontsize=18,
    fontweight='bold',
    y=0.95
)
fig.tight_layout()
fig.subplots_adjust(top=0.90, left=0.1, right=0.9, wspace=0.30)

out_path = os.path.join(output_folder, "simlulation_setup.png")
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved: {out_path}")