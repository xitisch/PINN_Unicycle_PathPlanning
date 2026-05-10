import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

# ============================================================
# Setup
# ============================================================
output_folder = os.path.join("results", "thesis_figures")
os.makedirs(output_folder, exist_ok=True)


# ============================================================
# Parameters (matching code implementation)
# ============================================================
x_c, y_c = 0.3, 0.1
r        = 0.20
d_safe   = 0.05      # safety distance set to 0.05 for better visualization
beta     = 40        # softplus beta


# ============================================================
# Compute obstacle loss on a grid (matching circ_obs_loss exactly)
# ============================================================
N = 300
x_range = np.linspace(-0.1, 0.7, N)
y_range = np.linspace(-0.3, 0.5, N)
X, Y = np.meshgrid(x_range, y_range)

# Distance to obstacle center (matches code: d = sqrt(... + 1e-8))
d = np.sqrt((X - x_c)**2 + (Y - y_c)**2 + 1e-8)

# softplus(z, beta) = (1/beta) * log(1 + exp(beta * z))
# Matches: F.softplus(r - d + buffer, beta=40)
z = r - d + d_safe
violation = (1.0 / beta) * np.log1p(np.exp(beta * z))
L_obs = violation**2


# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(13, 5.5))

# ---- Left: 2D heatmap ----
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.pcolormesh(
    X, Y, L_obs, cmap='plasma', shading='auto',
    vmin=0, vmax=0.02
)

# obstacle outline (solid white)
obs_outline = patches.Circle(
    (x_c, y_c), r,
    fill=False, edgecolor='white', linewidth=2.5, linestyle='-'
)
ax1.add_patch(obs_outline)

# safety distance contour (dashed cyan)
safety_outline = patches.Circle(
    (x_c, y_c), r + d_safe,
    fill=False, edgecolor='cyan', linewidth=2, linestyle='--'
)
ax1.add_patch(safety_outline)

# center marker
ax1.plot(x_c, y_c, marker='x', color='white', markersize=10, mew=2.5)

# colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label(r'$\ell_{\mathrm{obs}}(x, y)$', fontsize=12)

ax1.set_title("Obstacle Loss (Top-Down View)", fontsize=13)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlim(x_range.min(), x_range.max())
ax1.set_ylim(y_range.min(), y_range.max())

# ---- Right: 3D surface ----
z_cap = 0.02
L_obs_capped = np.minimum(L_obs, z_cap)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(
    X, Y, L_obs_capped,
    cmap='plasma', edgecolor='none',
    rstride=4, cstride=4, antialiased=True, alpha=0.9,
    vmin=0, vmax=z_cap
)

# Boundary ring at the obstacle edge
theta_ring = np.linspace(0, 2*np.pi, 200)
x_ring = x_c + r * np.cos(theta_ring)
y_ring = y_c + r * np.sin(theta_ring)
d_ring = np.sqrt((x_ring - x_c)**2 + (y_ring - y_c)**2 + 1e-8)
z_ring_input = r - d_ring + d_safe
z_ring_violation = (1.0 / beta) * np.log1p(np.exp(beta * z_ring_input))
z_ring = z_ring_violation**2

ax2.plot(
    x_ring, y_ring, z_ring,
    color='red', linewidth=2.5, zorder=10
)

# Safety distance ring (at r + d_safe)
x_safe_ring = x_c + (r + d_safe) * np.cos(theta_ring)
y_safe_ring = y_c + (r + d_safe) * np.sin(theta_ring)
d_safe_ring = np.sqrt(
    (x_safe_ring - x_c)**2 + (y_safe_ring - y_c)**2 + 1e-8
)
z_safe_input = r - d_safe_ring + d_safe
z_safe_violation = (1.0 / beta) * np.log1p(np.exp(beta * z_safe_input))
z_safe_ring = z_safe_violation**2

ax2.plot(
    x_safe_ring, y_safe_ring, z_safe_ring,
    color='red', linewidth=2, linestyle='--', zorder=10
)

# Zoom in on z-axis to see the smooth approximation
z_cap = 0.02     # adjust this value to see more or less of the gradient
ax2.set_zlim(0, z_cap)

ax2.set_title("Obstacle Loss Surface", fontsize=13)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel(r'$\ell_{\mathrm{obs}}$')
ax2.view_init(elev=30, azim=-60)

cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.6, pad=0.1)
cbar2.set_label(r'$\ell_{\mathrm{obs}}$', fontsize=12)


# ============================================================
# Title and save
# ============================================================
fig.suptitle(
    rf"Obstacle Loss Landscape "
    rf"($c=({x_c},{y_c}),\ r={r},\ d_{{\mathrm{{safe}}}}={d_safe}$)",
    fontsize=15, fontweight='bold', y=1.0
)
plt.tight_layout()
fig.subplots_adjust(top=0.88)

out_path = os.path.join(output_folder, "obstacle_loss_landscape.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved: {out_path}")