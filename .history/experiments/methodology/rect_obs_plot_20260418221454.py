import matplotlib.pyplot as plt
import os

# Rectangle parameters
xc, yc = 0.4, 0.1
w, h = 0.4, 0.4

xmin, xmax = xc - w/2, xc + w/2
ymin, ymax = yc - h/2, yc + h/2

# Define 3 scenarios
scenarios = [
    ("(i) Outside–Outside", (0.85, 0.45)),
    ("(ii) Inside–Outside", (0.4, 0.5)),
    ("(iii) Inside–Inside", (0.45, 0.1)),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (title, (px, py)) in zip(axes, scenarios):

    # Draw rectangle (same style)
    rect = plt.Rectangle(
        (xmin, ymin),
        w, h,
        facecolor="#c6d6e3",
        edgecolor="black",
        linewidth=2
    )
    ax.add_patch(rect)

    # Plot point
    ax.scatter(px, py, color="red", s=60, zorder=3)

    # Label point (x,y)
    ax.text(px, py + 0.05, r'$(x,y)$', ha='center', fontsize=10)

    # Closest point on rectangle (projection)
    if xmin < px < xmax and ymin < py < ymax:
        # Inside case → force projection to nearest edge (here right edge)
        cx = xmax
        cy = py
    else:
        cx = min(max(px, xmin), xmax)
        cy = min(max(py, ymin), ymax)

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
plt.savefig("sdf_cases_3.png", dpi=300)
plt.show()

