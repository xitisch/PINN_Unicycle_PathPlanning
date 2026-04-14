import torch
import numpy as np
import matplotlib.pyplot as plt

from pinnlib.pinn_functions import *
from pinnlib.training_pinn import train_model


# ============================================================
# Helper functions
# ============================================================

def compute_obstacle_terms(x, y, obs, buffer=0.01, beta=80):
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


# ============================================================
# Main script
# ============================================================

def main():
    # -------------------------
    # Parameters
    # -------------------------
    T = 1.0
    N = 200
    epochs = 5000

    BC = [0.0, 0.0, 1.0, 0.0]

    lambda_phy = 1.0
    lambda_obs = 10.0
    lambda_smooth = 0.0001

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
    lhs_theta_const = torch.full_like(omega, lambda_smooth / lambda_phy)
    rhs_theta = r_theta / (omega + 1e-6)

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

    lhs_theta_np = to_np(lhs_theta_const)
    rhs_theta_np = to_np(rhs_theta)

    dl_dx_np = to_np(dl_dx)
    dl_dy_np = to_np(dl_dy)
    omega_np = to_np(omega)

    # -------------------------
    # Masks for stable plotting
    # -------------------------
    mask_x = np.abs(rhs_x_np) > 1e-4
    mask_y = np.abs(rhs_y_np) > 1e-4
    mask_theta = np.abs(omega_np) > 1e-4

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
    plt.plot(x_np, y_np, label="Trajectory")
    plt.plot([BC[0], BC[2]], [BC[1], BC[3]], "--", alpha=0.6, label="Reference line")

    circle = plt.Circle((x_c, y_c), r, fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.scatter([BC[0], BC[2]], [BC[1], BC[3]], s=40, label="Start/Goal")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory with circular obstacle")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    plt.show()

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
    plt.tight_layout()
    plt.show()

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
    plt.show()

    # ============================================================
    # Plot 4: Angular relation (LHS vs RHS)
    # ============================================================
    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np[mask_theta], lhs_theta_np[mask_theta], label=r"LHS: $\lambda_{\mathrm{smooth}}/\lambda_{\mathrm{phys}}$")
    plt.plot(t_np[mask_theta], rhs_theta_np[mask_theta], "--", label=r"RHS: $r_\theta/\omega$")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.title("Angular EL verification")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Optional error plots
    # ============================================================
    err_x = lhs_x_np - rhs_x_np
    err_y = lhs_y_np - rhs_y_np
    err_v = lhs_v_np - rhs_v_np
    err_theta = lhs_theta_np - rhs_theta_np

    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np[mask_x], err_x[mask_x], label="x-error")
    plt.plot(t_np[mask_y], err_y[mask_y], label="y-error")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.title("Spatial EL error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np, err_v, label="velocity error")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.title("Velocity EL error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4.5))
    plt.plot(t_np[mask_theta], err_theta[mask_theta], label="angular error")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.title("Angular EL error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()