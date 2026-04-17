# Use for testing

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pinnlib.pinn_functions import *
from pinnlib.training_pinn import train_model

lambda_phy = 10
lambda_obs = 1e5
lambda_smooth = 1e-8

T = 1
N = 100

x0, y0 = 0.0, 0.0
xT, yT = 1.0, 0.0
v0 = 2
theta0 = 0
BC = [x0,y0,xT,yT,v0,theta0]

# Obstacle 1
x_c1, y_c1, r1 = 0.4, 0.2, 0.2

# Obstacle 2
x_c2, y_c2, r2 = 0.7, -0.2, 0.2


obs_circ = [
    [x_c1, y_c1, r1],
    [x_c2, y_c2, r2]
]

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

results = []


model = train_model(
    T=T,
    BC=[0, 0, 1, 0],
    obs=obs_circ,
    lambda_phy=lambda_phy,
    lambda_obs=lambda_obs,
    lambda_smooth=lambda_smooth,
    epochs=2000,
    N=200
)

with torch.no_grad():
    nn_input = model(t_list)
    x, y, theta, v, omega = hard_bc_transform(t_list, nn_input, T, BC)

    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    tT = torch.tensor([[T]], dtype=torch.float32, device=device)
    x0p, y0p, *_ = hard_bc_transform(t0, model(t0), T, BC)
    xTp, yTp, *_ = hard_bc_transform(tT, model(tT), T, BC)
    print("\nBC check:")
    print(f"x(0)={x0p.item():.6f}, y(0)={y0p.item():.6f}")
    print(f"x(T)={xTp.item():.6f}, y(T)={yTp.item():.6f}")




# -----------------------------------
# Prepare numpy arrays
# -----------------------------------
x_np = x.squeeze().cpu().numpy()
y_np = y.squeeze().cpu().numpy()
t_np = t_list.detach().cpu().numpy().squeeze()
v_np = v.squeeze().cpu().numpy()
omega_np = omega.squeeze().cpu().numpy()

# Obstacles
obstacles = [
    {"x_c": x_c1, "y_c": y_c1, "r": r1, "label": "Obstacle 1"},
    {"x_c": x_c2, "y_c": y_c2, "r": r2, "label": "Obstacle 2"},
]

# -----------------------------------
# Axis limits for trajectory plot
# Include full obstacle extents so circles are never clipped
# -----------------------------------
all_x = [x_np]
all_y = [y_np]

for obs in obstacles:
    all_x.append(np.array([obs["x_c"] - obs["r"], obs["x_c"] + obs["r"]]))
    all_y.append(np.array([obs["y_c"] - obs["r"], obs["y_c"] + obs["r"]]))

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

pad = 0.08
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim = (all_y.min() - pad, all_y.max() + pad)

# -----------------------------------
# Figure layout
# -----------------------------------
fig = plt.figure(figsize=(11, 7))

# Top: trajectory spans both columns
ax_traj = plt.subplot(2, 2, (1, 2))
# Bottom left: velocity
ax_v = plt.subplot(2, 2, 3)
# Bottom right: angular velocity
ax_w = plt.subplot(2, 2, 4)

# -----------------------------------
# Trajectory plot
# -----------------------------------
ax_traj.plot(x_np, y_np, linewidth=2)

# Reference straight line, same style idea as circle.py
ax_traj.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", linewidth=1, alpha=0.5)

# Obstacles: filled + outline + center marker "x"
for obs in obstacles:
    fill = patches.Circle(
        (obs["x_c"], obs["y_c"]),
        obs["r"],
        fill=True,
        alpha=0.15,
        linewidth=0,
        label=obs["label"]
    )
    edge = patches.Circle(
        (obs["x_c"], obs["y_c"]),
        obs["r"],
        fill=False,
        linewidth=2.0
    )
    ax_traj.add_patch(fill)
    ax_traj.add_patch(edge)

    ax_traj.plot(
        obs["x_c"], obs["y_c"],
        marker="x", markersize=7, mew=2
    )

# Start/goal markers, same style idea as circle.py
ax_traj.plot([BC[0]], [BC[1]], marker="o")
ax_traj.plot([BC[2]], [BC[3]], marker="o")

ax_traj.set_title("PINN unicycle path w/ hard x,y BCs")
ax_traj.set_xlabel("x")
ax_traj.set_ylabel("y")
ax_traj.set_xlim(*xlim)
ax_traj.set_ylim(*ylim)
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.grid(True, alpha=0.25)
ax_traj.legend()

# Optional text box like circle.py style
ax_traj.text(
    0.02, 0.06,
    f"c1=({x_c1:.2f},{y_c1:.2f}), r1={r1:.2f}\n"
    f"c2=({x_c2:.2f},{y_c2:.2f}), r2={r2:.2f}",
    transform=ax_traj.transAxes,
    fontsize=8,
    va="bottom",
    ha="left"
)

# -----------------------------------
# Velocity plot
# -----------------------------------
ax_v.plot(t_np, v_np, linewidth=2)
ax_v.set_title("Velocity")
ax_v.set_xlabel("t (s)")
ax_v.set_ylabel("v (m/s)")
ax_v.grid(True, alpha=0.25)

# -----------------------------------
# Angular velocity plot
# -----------------------------------
ax_w.plot(t_np, omega_np, linewidth=2)
ax_w.set_title("Angular velocity")
ax_w.set_xlabel("t (s)")
ax_w.set_ylabel("ω (rad/s)")
ax_w.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()