import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ============================================================
# Setup
# ============================================================
output_folder = os.path.join("results", "thesis_figures")
os.makedirs(output_folder, exist_ok=True)

# Common parameters
x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
x_c    = 0.5

# Intrusion value
Delta = 0.10

# Circle: radius and y_c such that obstacle intrudes by Delta
r_circ = 0.20
y_c_circ = r_circ - Delta   # so top of circle is at y = -Delta + 2*Delta = ...
                            # actually: top of circle = y_c + r = (r - Delta) + r? No:
                            # We want bottom of circle to be at y = -Delta? No:
                            # The path is at y=0. We want the circle to intrude by Delta into y>0.
                            # If circle center is at y_c and radius is r, and we want the bottom edge at y = -Delta below path:
                            # Actually for "intrusion below path", lower edge at y = y_c - r = -Delta
                            # → y_c = r - Delta with the convention that "intrusion" means how far the obstacle reaches across the path
                            # Here: y_c = r - Delta means top edge at y = 2r - Delta, bottom edge at y = -Delta

# Rectangle: square with same effective intrusion
w_rect = h_rect = 2 * r_circ * np.sin(np.pi/4)   # = r_circ*sqrt(2), same corner-to-center as circle
y_c_rect = h_rect/2 - Delta

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ----- Circle -----
ax = axes[0]

# Reference line (path)
ax.axhline(0, color='orange', linestyle='--', alpha=0.6, linewidth=1.5, zorder=1)

# Obstacle
circle = patches.Circle(
    (x_c, y_c_circ), r_circ,
    edgecolor='black', facecolor='#c6d6e3', linewidth=2, zorder=2
)
ax.add_patch(circle)

# Center marker
ax.plot(x_c, y_c_circ, 'x', color='black', markersize=8, mew=2, zorder=3)

# Intrusion arrow (vertical from path to top of circle)
top_of_obs_circ = y_c_circ + r_circ
ax.annotate(
    '', xy=(x_c + 0.05, top_of_obs_circ), xytext=(x_c + 0.05, 0),
    arrowprops=dict(arrowstyle='<->', color='red', lw=2)
)
ax.text(
    x_c + 0.10, top_of_obs_circ/2,
    rf'$\Delta = {Delta:.2f}$',
    fontsize=14, color='red', va='center'
)

# Start and goal
ax.plot(x0, y0, 'o', color='green', markersize=10, zorder=4)
ax.plot(xT, yT, 'o', color='red', markersize=10, zorder=4)
ax.text(x0, -0.05, 'Start', ha='center', va='top', fontsize=11)
ax.text(xT, -0.05, 'Goal',  ha='center', va='top', fontsize=11)

# Labels
ax.set_title('Circular Obstacle', fontsize=14, pad=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.2, 0.4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# ----- Rectangle -----
ax = axes[1]

# Reference line (path)
ax.axhline(0, color='orange', linestyle='--', alpha=0.6, linewidth=1.5, zorder=1)

# Obstacle
rect = patches.Rectangle(
    (x_c - w_rect/2, y_c_rect - h_rect/2), w_rect, h_rect,
    edgecolor='black', facecolor='#c6d6e3', linewidth=2, zorder=2
)
ax.add_patch(rect)

# Center marker
ax.plot(x_c, y_c_rect, 'x', color='black', markersize=8, mew=2, zorder=3)

# Intrusion arrow (vertical from path to top of rectangle)
top_of_obs_rect = y_c_rect + h_rect/2
ax.annotate(
    '', xy=(x_c + 0.12, top_of_obs_rect), xytext=(x_c + 0.12, 0),
    arrowprops=dict(arrowstyle='<->', color='red', lw=2)
)
ax.text(
    x_c + 0.17, top_of_obs_rect/2,
    rf'$\Delta = {Delta:.2f}$',
    fontsize=14, color='red', va='center'
)

# Start and goal
ax.plot(x0, y0, 'o', color='green', markersize=10, zorder=4)
ax.plot(xT, yT, 'o', color='red', markersize=10, zorder=4)
ax.text(x0, -0.05, 'Start', ha='center', va='top', fontsize=11)
ax.text(xT, -0.05, 'Goal',  ha='center', va='top', fontsize=11)

# Labels
ax.set_title('Rectangular Obstacle', fontsize=14, pad=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.2, 0.4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# ============================================================
# Save
# ============================================================
fig.suptitle(
    'Effective Intrusion Distance $\\Delta$',
    fontsize=16, fontweight='bold', y=1.02
)
plt.tight_layout()

out_path = os.path.join(output_folder, 'effective_intrusion.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"Saved: {out_path}")