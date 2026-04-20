import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from src.pinn.pinn_functions import *
from src.pinn.training_pinn import train_model

output_folder = os.path.join("results", "el_verification_circle_new")
os.makedirs(output_folder, exist_ok=True)

def compute_obstacle_terms(x, y, obs, buffer=0.01, beta=50):
    """
    Computes:
      d           : distance to obstacle center
      violation   : soft obstacle violation
      l_obs       : pointwise obstacle penalty = violation^2
      dl_dx, dl_dy: partial derivatives of l_obs wrt x and y
    """
    x_c, y_c, r = obs
    d = torch.sqrt((x - x_c)**2 + (y - y_c)**2 + 1e-8)

    violation = torch.nn.functional.softplus((r - d + buffer), beta=beta)
    l_obs = violation**2

    dl_dx = derivative(l_obs, x)
    dl_dy = derivative(l_obs, y)

    return d, violation, l_obs, dl_dx, dl_dy


def compute_residuals_and_derivatives(model, t_list, T, BC):
    """
    Returns trajectory states, first derivatives, residuals,
    and time derivatives of residuals.
    """
    nn_input = model(t_list)
    x, y, theta, v, omega = hard_bc_transform(t_list, nn_input, T, BC)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)
    theta_t = derivative(theta, t_list)

    r_x = x_t - v * torch.cos(theta)
    r_y = y_t - v * torch.sin(theta)
    r_theta = theta_t - omega

    r_x_t = derivative(r_x, t_list)
    r_y_t = derivative(r_y, t_list)

    return x, y, theta, v, omega, x_t, y_t, theta_t, r_x, r_y, r_theta, r_x_t, r_y_t


def to_np(tensor):
    return tensor.detach().cpu().numpy().flatten()

def set_dynamic_ylim(*arrays, margin=0.1):
    data = np.concatenate([a for a in arrays if len(a) > 0])
    if len(data) == 0:
        return
    ymin, ymax = np.min(data), np.max(data)
    span = ymax - ymin
    if span < 1e-6:
        span = 1e-3
    plt.ylim(ymin - margin * span, ymax + margin * span)


# ============================================================
# Main script
# ============================================================

def main():
    # -------------------------
    # Parameters
    # -------------------------
    T = 1.0
    N = 400
    epochs = 4000

    x0, y0 = 0.0, 0.0
    xT, yT = 1.0, 0.0
    v0 = 2
    theta0 = 0
    BC = [x0,y0,xT,yT,v0,theta0]

    lambda_phy = 5
    lambda_obs = 5
    lambda_smooth = 0.5

    # Circle obstacle
    r = 0.2
    Delta = 0.1
    x_c = 0.4
    y_c = r - Delta
    obs = [[x_c, y_c, r]]

    # -------------------------
    # Train model
    # -------------------------
    model = train_model(
        T=T,
        BC=BC,
        obs=obs,
        lambda_phy=lambda_phy,
        lambda_obs=lambda_obs,
        lambda_smooth=lambda_smooth,
        epochs=epochs,
        N=N
    )

    # -------------------------
    # Evaluation grid
    # -------------------------
    t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
    t_list.requires_grad_(True)

    # -------------------------
    # Compute states and residuals
    # -------------------------
    (
        x, y, theta, v, omega,
        x_t, y_t, theta_t,
        r_x, r_y, r_theta,
        r_x_t, r_y_t
    ) = compute_residuals_and_derivatives(model, t_list, T, BC)

    # -------------------------
    # Obstacle penalty terms
    # -------------------------
    _, violation, l_obs, dl_dx, dl_dy = compute_obstacle_terms(x, y, obs[0])

    # ============================================================
    # 1) Spatial EL equation:
    #    r_x_dot = (lambda_obs / (2 lambda_phy)) * d l_obs / dx
    #    r_y_dot = (lambda_obs / (2 lambda_phy)) * d l_obs / dy
    # ============================================================
    scale = lambda_obs / (2.0 * lambda_phy)

    lhs_x = r_x_t
    rhs_x = scale * dl_dx

    lhs_y = r_y_t
    rhs_y = scale * dl_dy

    # ============================================================
    # 2) Velocity EL equation:
    #    lambda_smooth * v = lambda_phy * (r_x cos(theta) + r_y sin(theta))
    # ============================================================
    lhs_v = lambda_smooth * v
    rhs_v = lambda_phy * (r_x * torch.cos(theta) + r_y * torch.sin(theta))

    # ============================================================
    # 3) Angular equation:
    #    lambda_smooth / lambda_phy = r_theta / omega
    #    Also plot LHS and RHS separately
    # ============================================================
    lhs_theta = lambda_smooth * omega
    rhs_theta = lambda_phy * r_theta

    # -------------------------
    # Convert to numpy
    # -------------------------
    t_np = to_np(t_list)

    x_np = to_np(x)
    y_np = to_np(y)

    lhs_x_np = to_np(lhs_x)
    rhs_x_np = to_np(rhs_x)
    lhs_y_np = to_np(lhs_y)
    rhs_y_np = to_np(rhs_y)

    lhs_v_np = to_np(lhs_v)
    rhs_v_np = to_np(rhs_v)

    lhs_theta_np = to_np(lhs_theta)
    rhs_theta_np = to_np(rhs_theta)

    dl_dx_np = to_np(dl_dx)
    dl_dy_np = to_np(dl_dy)
    omega_np = to_np(omega)

    # -------------------------
    # Masks for stable plotting
    # -------------------------
    mask_x = np.isfinite(rhs_x_np)
    mask_y = np.isfinite(rhs_y_np)
    mask_theta = np.isfinite(rhs_theta_np)

    # -------------------------
    # Print some summary info
    # -------------------------
    print(f"Obstacle: center=({x_c:.3f}, {y_c:.3f}), r={r:.3f}, Delta={Delta:.3f}")
    print(f"lambda_obs / (2 lambda_phy) = {lambda_obs / (2.0 * lambda_phy):.6f}")
    print(f"lambda_smooth / lambda_phy  = {lambda_smooth / lambda_phy:.6f}")

    # ============================================================
    # Plot 0: trajectory
    # ============================================================
    plt.figure(figsize=(6, 5))

    # Trajectory
    plt.plot(x_np, y_np, linewidth=2.5)

    # Reference line (dashed, lighter)
    plt.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", alpha=0.5)

    # Circle obstacle (filled + edge)
    circle = plt.Circle((x_c, y_c), r, edgecolor="black", facecolor="#c6d6e3", linewidth=2)
    plt.gca().add_patch(circle)

    # Obstacle center
    plt.scatter(x_c, y_c, marker="x", s=60)

    # Start / goal points
    plt.scatter(BC[0], BC[1], s=50)
    plt.scatter(BC[2], BC[3], s=50)

    # Text annotation
    plt.text(
        0.05, -0.28,
        f"c=({x_c:.2f},{y_c:.2f})\nr={r:.2f}",
        fontsize=11
    )

    # Axes formatting
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "trajectory.png"), dpi=300)

    plt.show()
    plt.close()

    # ============================================================
    # Plot 1: Spatial EL, x-direction (LHS vs RHS)
    # ============================================================
    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np[mask_x], lhs_x_np[mask_x], label=r"LHS: $\dot{r}_x$")
    plt.plot(t_np[mask_x], rhs_x_np[mask_x], "--", label=r"RHS: $\frac{\lambda_{\mathrm{obs}}}{2\lambda_{\mathrm{phys}}}\frac{\partial \ell_{\mathrm{obs}}}{\partial x}$")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title("Spatial EL verification in x-direction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    set_dynamic_ylim(lhs_x_np, rhs_x_np)
    plt.savefig(os.path.join(output_folder, "spatial_EL_x.png"), dpi=300)

    plt.show()
    plt.close()

    # ============================================================
    # Plot 2: Spatial EL, y-direction (LHS vs RHS)
    # ============================================================
    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np[mask_y], lhs_y_np[mask_y], label=r"LHS: $\dot{r}_y$")
    plt.plot(t_np[mask_y], rhs_y_np[mask_y], "--", label=r"RHS: $\frac{\lambda_{\mathrm{obs}}}{2\lambda_{\mathrm{phys}}}\frac{\partial \ell_{\mathrm{obs}}}{\partial y}$")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title("Spatial EL verification in y-direction")
    plt.grid(True)
    plt.legend()
    set_dynamic_ylim(lhs_y_np, rhs_y_np)

    plt.savefig(os.path.join(output_folder, "spatial_EL_y.png"), dpi=300)

    plt.show()
    plt.close()

    # ============================================================
    # Plot 3: Velocity EL (LHS vs RHS)
    # ============================================================
    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np, lhs_v_np, label=r"LHS: $\lambda_{\mathrm{smooth}} v$")
    plt.plot(t_np, rhs_v_np, "--", label=r"RHS: $\lambda_{\mathrm{phys}}(r_x\cos\theta + r_y\sin\theta)$")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title("Velocity EL verification")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    set_dynamic_ylim(lhs_v_np, rhs_v_np)

    plt.savefig(os.path.join(output_folder, "velocity_EL.png"), dpi=300)
    plt.show()
    plt.close()

    # ============================================================
    # Plot 4: Angular relation (LHS vs RHS)
    # ============================================================
    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np, lhs_theta_np, label=r"LHS: $\lambda_{\mathrm{smooth}} \omega$")
    plt.plot(t_np, rhs_theta_np, "--", label=r"RHS: $\lambda_{\mathrm{phys}} r_\theta$")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title("Angular EL verification")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    set_dynamic_ylim(lhs_theta_np, rhs_theta_np)

    plt.savefig(os.path.join(output_folder, "angular_EL.png"), dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()