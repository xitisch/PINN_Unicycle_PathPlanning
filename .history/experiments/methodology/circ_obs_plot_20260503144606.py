import matplotlib.pyplot as plt
import numpy as np
import os

# Circle parameters
xc, yc = 0.4, 0.1
r = 0.2

# Define 2 scenarios
scenarios = [
    ("(i) Outside", (0.85, 0.45)),
    ("(ii) Inside", (0.45, 0.15)),
]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, (title, (px, py)) in zip(axes, scenarios):

    # Draw circle (same style as rectangle)
    circle = plt.Circle(
        (xc, yc),
        r,
        facecolor="#c6d6e3",
        edgecolor="black",
        linewidth=2
    )
    ax.add_patch(circle)

    # Plot point
    ax.scatter(px, py, color="red", s=60, zorder=3)

    # Label point (x,y)
    ax.text(px, py + 0.05, r'$(x,y)$', ha='center', fontsize=10)

    # Closest point on circle (projection onto boundary)
    dx = px - xc
    dy = py - yc
    dist = np.sqrt(dx**2 + dy**2)

    # Project onto circle boundary (works for both inside and outside)
    cx = xc + r * dx / dist
    cy = yc + r * dy / dist

    # Dashed line to closest boundary
    ax.plot([px, cx], [py, cy], linestyle='--', linewidth=1.5)

    # Optional: mark projection point
    ax.scatter(cx, cy, s=40, zorder=3)

    # Formatting
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Axis labels (important for thesis clarity)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
# Ensure directory exists
output_folder = os.path.join("results", "thesis_figures")
os.makedirs(output_folder, exist_ok=True)

# Save figure
save_path = os.path.join(output_folder, "sdf_cases_circle.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()