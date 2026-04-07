import matplotlib.pyplot as plt
import numpy as np

# Rectangle parameters (same style as your plot)
xc, yc = 0.4, 0.1
w, h = 0.4, 0.4

xmin, xmax = xc - w/2, xc + w/2
ymin, ymax = yc - h/2, yc + h/2

# Define 4 scenarios
scenarios = [
    ("Both outside (corner)", (0.85, 0.45)),
    ("x inside, y outside", (0.4, 0.5)),
    ("x outside, y inside", (0.85, 0.1)),
    ("Both inside", (0.4, 0.1)),
]

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for ax, (title, (px, py)) in zip(axes.flatten(), scenarios):

    # Draw rectangle (same style as your plot)
    rect = plt.Rectangle(
        (xmin, ymin),
        w, h,
        facecolor="#c6d6e3",  # light blue fill
        edgecolor="black",
        linewidth=2
    )
    ax.add_patch(rect)

    # Plot point
    ax.scatter(px, py, color="red", s=60, zorder=3)

    # Optional: dashed reference lines (like your plot style)
    ax.axhline(y=yc, linestyle="--", color="orange", alpha=0.6)
    ax.axvline(x=xc, linestyle="--", color="orange", alpha=0.6)

    # Formatting
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

# Layout
plt.tight_layout()
plt.savefig("sdf_cases.png", dpi=300)
plt.show()