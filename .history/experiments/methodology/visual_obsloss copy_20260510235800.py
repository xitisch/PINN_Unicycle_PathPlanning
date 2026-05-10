# Rectangular obstacle: loss landscape visualization
#
# Saves:
#   results/thesis_figures/obstacle_loss_landscape_rect.png

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


output_folder = os.path.join("results", "thesis_figures")
os.makedirs(output_folder, exist_ok=True)


# ============================================================
# Parameters (matching code implementation)
# ============================================================
x_c, y_c    = 0.3, 0.1
w = h       = float(0.2 * np.sqrt(2))   # square matched to r=0.20
d_safe      = 0.05                       # safety distance (buffer)
beta_softp  = 40                         # softplus beta for violation
beta_inout  = 50                         # softplus beta for outside-distance (matches rect_sdf)
beta_lse    = 50                         # smoothness for lse_max/lse_min


# ============================================================
# Helper functions matching pinn_functions.py
# ============================================================
def lse_max(a, b, beta=beta_lse):
    """Smooth approximation of max(a, b)."""
    return (1.0 / beta) * np.log(np.exp(beta * a) + np.exp(beta * b))


def lse_min(a, b, beta=beta_lse):
    """Smooth approximation of min(a, b)."""
    return -(1.0 / beta) * np.log(np.exp(-beta * a) + np.exp(-beta * b))


# ============================================================
# Compute obstacle loss on a grid (matching rect_obs_loss)
# ============================================================
N = 300
x_range = np.linspace(-0.1, 0.7, N)
y_range = np.linspace(-0.3, 0.5, N)
X, Y = np.meshgrid(x_range, y_range)

hx = 0.5 * w
hy = 0.5 * h

qx = np.abs(X - x_c) - hx
qy = np.abs(Y - y_c) - hy

# Outside distance via softplus (matches code)
ox      = (1.0 / beta_inout) * np.log1p(np.exp(beta_inout * qx))
oy      = (1.0 / beta_inout) * np.log1p(np.exp(beta_inout * qy))
outside = np.sqrt(ox**2 + oy**2 + 1e-12)

# Inside distance via LSE smoothing
inside  = lse_min(lse_max(qx, qy), np.zeros_like(qx))

# Signed distance
d = outside + inside

# Violation via softplus (matches code)
z_input   = d_safe - d
violation = (1.0 / beta_softp) * np.log1p(np.exp(beta_softp * z_input))
L_obs     = violation**2


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

# obstacle outline (solid white rectangle)
obs_outline = patches.Rectangle(
    (x_c - hx, y_c - hy), w, h,
    fill=False, edgecolor='white', linewidth=2.5, linestyle='-'
)
ax1.add_patch(obs_outline)

# safety distance contour (dashed cyan rectangle, slightly larger)
safety_outline = patches.Rectangle(
    (x_c - hx - d_safe, y_c - hy - d_safe),
    w + 2 * d_safe, h + 2 * d_safe,
    fill=False, edgecolor='cyan', linewidth=2, linestyle='--'
)
ax1.add_patch(safety_outline)

# center marker
ax1.plot(x_c, y_c, marker='x', color='white', markersize=10, mew=2.5)

# legend
ax1.plot([], [], color='white', linewidth=2.5, linestyle='-',
         label='Obstacle boundary')
ax1.plot([], [], color='cyan',  linewidth=2,   linestyle='--',
         label='Safety distance')
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)

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
    rstride=4, cstride=4, antialiased=True, alpha=0.7,
    vmin=0, vmax=z_cap
)


def loss_value_at(x_eval, y_eval):
    """Compute the obstacle loss at a single (x, y) point."""
    qx_e = np.abs(x_eval - x_c) - hx
    qy_e = np.abs(y_eval - y_c) - hy
    ox_e = (1.0 / beta_inout) * np.log1p(np.exp(beta_inout * qx_e))
    oy_e = (1.0 / beta_inout) * np.log1p(np.exp(beta_inout * qy_e))
    outside_e = np.sqrt(ox_e**2 + oy_e**2 + 1e-12)
    inside_e  = lse_min(lse_max(qx_e, qy_e), np.zeros_like(qx_e))
    d_e = outside_e + inside_e
    z_e = d_safe - d_e
    v_e = (1.0 / beta_softp) * np.log1p(np.exp(beta_softp * z_e))
    return v_e**2


# Obstacle boundary outline (solid red, on the surface)
N_edge = 200
# 4 edges of the rectangle
t = np.linspace(0, 1, N_edge // 4)
edge_x = np.concatenate([
    x_c - hx + w * t,                    # bottom edge
    np.full_like(t, x_c + hx),           # right edge
    x_c + hx - w * t,                    # top edge
    np.full_like(t, x_c - hx),           # left edge
])
edge_y = np.concatenate([
    np.full_like(t, y_c - hy),
    y_c - hy + h * t,
    np.full_like(t, y_c + hy),
    y_c + hy - h * t,
])
edge_z = loss_value_at(edge_x, edge_y)

ax2.plot(
    edge_x, edge_y, edge_z,
    color='red', linewidth=2.5, zorder=10
)

# Safety distance outline (dashed red, slightly larger rectangle)
safe_x = np.concatenate([
    x_c - hx - d_safe + (w + 2 * d_safe) * t,
    np.full_like(t, x_c + hx + d_safe),
    x_c + hx + d_safe - (w + 2 * d_safe) * t,
    np.full_like(t, x_c - hx - d_safe),
])
safe_y = np.concatenate([
    np.full_like(t, y_c - hy - d_safe),
    y_c - hy - d_safe + (h + 2 * d_safe) * t,
    np.full_like(t, y_c + hy + d_safe),
    y_c + hy + d_safe - (h + 2 * d_safe) * t,
])
safe_z = loss_value_at(safe_x, safe_y)

ax2.plot(
    safe_x, safe_y, safe_z,
    color='red', linewidth=2, linestyle='--', zorder=10
)

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
    rf"($c=({x_c},{y_c}),\ w=h={w:.2f},\ d_{{\mathrm{{safe}}}}={d_safe}$)",
    fontsize=15, fontweight='bold', y=1.0
)
plt.tight_layout()
fig.subplots_adjust(top=0.88)

out_path = os.path.join(output_folder, "obstacle_loss_landscape_rect.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved: {out_path}")