import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from src.pinn.pinn_functions import *
from src.pinn.train_pinn import train_model

output_folder = os.path.join("results", "el_evaluation")
os.makedirs(output_folder, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def compute_obstacle_terms_circ(x, y, obs, safety=0.03, beta=40):
    x_c, y_c, r = obs
    d = torch.sqrt((x - x_c)**2 + (y - y_c)**2 + 1e-8)
    violation = F.softplus((r - d + safety), beta=beta)
    l_obs = violation**2
    dl_dx = derivative(l_obs, x)
    dl_dy = derivative(l_obs, y)
    return d, violation, l_obs, dl_dx, dl_dy

def compute_obstacle_terms_rect(x, y, theta, v, obs, safety=0.03, beta=40):
    xmin, xmax, ymin, ymax = obs

    T_L = 0.02
    x_L = x + v * T_L * torch.cos(theta)
    y_L = y + v * T_L * torch.sin(theta)

    d = rect_sdf(x_L, y_L, xmin, xmax, ymin, ymax)

    violation = F.softplus((safety - d), beta=beta)
    l_obs = violation**2

    dl_dx = derivative(l_obs, x)
    dl_dy = derivative(l_obs, y)
    dl_dtheta = derivative(l_obs, theta)
    dl_dv = derivative(l_obs, v)

    return d, violation, l_obs, dl_dx, dl_dy, dl_dtheta, dl_dv

def compute_residuals_and_derivatives(model, t_list, T, BC):
    nn_input = model(t_list)
    x, y, theta, v, omega = hard_bc_transform(t_list, nn_input, T, BC)

    x_t     = derivative(x,     t_list)
    y_t     = derivative(y,     t_list)
    theta_t = derivative(theta, t_list)

    r_x     = x_t     - v * torch.cos(theta)
    r_y     = y_t     - v * torch.sin(theta)
    r_theta = theta_t - omega

    r_x_t = derivative(r_x, t_list)
    r_y_t = derivative(r_y, t_list)

    return x, y, theta, v, omega, r_x, r_y, r_theta, r_x_t, r_y_t


def to_np(tensor):
    return tensor.detach().cpu().numpy().flatten()


def effective_weights(scales, lambda_phy, lambda_obs, lambda_v, lambda_omega):
    eps = 1e-8
    return (
        lambda_phy   / (scales["phy"]   + eps),
        lambda_obs   / (scales["obs"]   + eps),
        lambda_v     / (scales["v"]     + eps),
        lambda_omega / (scales["omega"] + eps),
    )


def compute_all(model, t_list, T, BC, obs, obstacle_type,
                lam_obs_eff, lam_phy_eff, lam_v_eff, lam_omega_eff):
    (x, y, theta, v, omega,
     r_x, r_y, r_theta,
     r_x_t, r_y_t) = compute_residuals_and_derivatives(model, t_list, T, BC)

    if obstacle_type == "circle":
        _, _, _, dl_dx, dl_dy = compute_obstacle_terms_circ(x, y, obs)
        dl_dtheta = torch.zeros_like(theta)
        dl_dv = torch.zeros_like(v)

    else:
        _, _, _, dl_dx, dl_dy, dl_dtheta, dl_dv = compute_obstacle_terms_rect(
            x, y, theta, v, obs
        )

    scale = lam_obs_eff / (2.0 * lam_phy_eff)

    return {
        "x":         to_np(x),
        "y":         to_np(y),
        "lhs_x":     to_np(r_x_t),
        "rhs_x":     to_np(scale * dl_dx),
        "lhs_y":     to_np(r_y_t),
        "rhs_y":     to_np(scale * dl_dy),
        "lhs_v":     to_np(lam_v_eff * v),
        "rhs_v":     to_np(lam_phy_eff * (r_x * torch.cos(theta)
                                          + r_y * torch.sin(theta))),
        "lhs_omega": to_np(lam_omega_eff * omega),
        "rhs_omega": to_np(lam_phy_eff * r_theta),
    }


def plot_dual_axis(ax, t_np, lhs, rhs, lhs_label, rhs_label, title):
    ax2  = ax.twinx()
    mask = np.isfinite(rhs)

    ax.plot(t_np[mask],  lhs[mask],        color="tab:blue")
    ax2.plot(t_np[mask], rhs[mask], "--",  color="tab:red")

    ax.set_xlabel("$t$")
    ax.set_ylabel(f"LHS: {lhs_label}",  color="tab:blue", fontsize=8)
    ax2.set_ylabel(f"RHS: {rhs_label}", color="tab:red",  fontsize=8)
    ax.tick_params(axis='y',  labelcolor="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12, pad=12)


def plot_trajectory(ax, data, obs_type, obs_def, BC):
    ax.plot(data["x"], data["y"], linewidth=2.5, color="tab:blue")
    ax.plot(
        [BC[0], BC[2]], [BC[1], BC[3]],
        linestyle="--", alpha=0.5, color="orange"
    )

    if obs_type == "circle":
        patch = plt.Circle(
            (obs_def[0], obs_def[1]), obs_def[2],
            edgecolor="black", facecolor="#c6d6e3", linewidth=2
        )
        ax.add_patch(patch)
        ax.scatter(obs_def[0], obs_def[1],
                   marker="x", s=60, color="tab:blue")
        ax.text(
            0.02, -0.28,
            f"c=({obs_def[0]:.2f},{obs_def[1]:.2f})\nr={obs_def[2]:.2f}",
            fontsize=9, transform=ax.transData
        )
        traj_title = "Trajectory (Circle)"
    else:
        x_c_r = 0.5 * (obs_def[0] + obs_def[1])
        y_c_r = 0.5 * (obs_def[2] + obs_def[3])
        w_r   = obs_def[1] - obs_def[0]
        h_r   = obs_def[3] - obs_def[2]
        patch = plt.Rectangle(
            (obs_def[0], obs_def[2]), w_r, h_r,
            edgecolor="black", facecolor="#c6d6e3", linewidth=2
        )
        ax.add_patch(patch)
        ax.scatter(x_c_r, y_c_r,
                   marker="x", s=60, color="tab:blue")
        ax.text(
            0.02, -0.28,
            f"c=({x_c_r:.2f},{y_c_r:.2f})\nw={w_r:.2f}, h={h_r:.2f}",
            fontsize=9, transform=ax.transData
        )
        traj_title = "Trajectory (Rectangle)"

    ax.scatter(BC[0], BC[1], s=60, color="orange", zorder=5)
    ax.scatter(BC[2], BC[3], s=60, color="green",  zorder=5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.4,  0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_title(traj_title, fontsize=12, pad=12)


# ============================================================
# Main
# ============================================================

def main():
    # -------------------------
    # Parameters
    # -------------------------
    T        = 1.0
    N        = 400
    epochs   = 3000

    lambda_phy   = 20
    lambda_obs   = 50
    lambda_v     = 0.2
    lambda_omega = 2

    x0, y0 = 0.0, 0.0
    xT, yT = 1.0, 0.0
    BC = [x0, y0, xT, yT]

    # -------------------------
    # Obstacle definitions
    # -------------------------
    r        = 0.2
    Delta    = 0.1
    x_c_circ = 0.4
    y_c_circ = r - Delta
    obs_circ = [x_c_circ, y_c_circ, r]

    w = h        = float(0.2 * np.sqrt(2))
    Delta_rect   = 0.1 * np.sqrt(2) - 0.1
    x_c_rect     = 0.4
    y_c_rect     = h/2 - Delta_rect
    xmin = x_c_rect - w/2
    xmax = x_c_rect + w/2
    ymin = y_c_rect - h/2
    ymax = y_c_rect + h/2
    obs_rect = [xmin, xmax, ymin, ymax]

    # -------------------------
    # Train models
    # -------------------------
    print("Training circular model...")
    model_circ, scales_circ = train_model(
        T=T, BC=BC, obs=[obs_circ],
        lambda_phy=lambda_phy, lambda_obs=lambda_obs,
        lambda_v=lambda_v, lambda_omega=lambda_omega,
        epochs=epochs, N=N,
        return_scales=True
    )

    print("Training rectangular model...")
    model_rect, scales_rect = train_model(
        T=T, BC=BC, obs=[obs_rect],
        lambda_phy=lambda_phy, lambda_obs=lambda_obs,
        lambda_v=lambda_v, lambda_omega=lambda_omega,
        epochs=epochs, N=N,
        return_scales=True
    )

    # -------------------------
    # Effective weights
    # -------------------------
    lam_phy_c, lam_obs_c, lam_v_c, lam_omega_c = effective_weights(
        scales_circ, lambda_phy, lambda_obs, lambda_v, lambda_omega
    )
    lam_phy_r, lam_obs_r, lam_v_r, lam_omega_r = effective_weights(
        scales_rect, lambda_phy, lambda_obs, lambda_v, lambda_omega
    )

    # -------------------------
    # Evaluation grid
    # -------------------------
    t_list = torch.linspace(0.0, T, N).view(-1, 1)
    t_list.requires_grad_(True)
    t_np = to_np(t_list)

    # -------------------------
    # Compute all EL quantities
    # -------------------------
    circ = compute_all(
        model_circ, t_list, T, BC, obs_circ, "circle",
        lam_obs_c, lam_phy_c, lam_v_c, lam_omega_c
    )
    rect = compute_all(
        model_rect, t_list, T, BC, obs_rect, "rectangle",
        lam_obs_r, lam_phy_r, lam_v_r, lam_omega_r
    )

    el_plots = [
        (
            "lhs_x", "rhs_x",
            r"$\dot{r}_x$",
            r"$\frac{\lambda_{\rm obs}}{2\lambda_{\rm phys}}"
            r"\frac{\partial \ell_{\rm obs}}{\partial x}$",
            "Spatial EL Evaluation In X-Direction",
        ),
        (
            "lhs_y", "rhs_y",
            r"$\dot{r}_y$",
            r"$\frac{\lambda_{\rm obs}}{2\lambda_{\rm phys}}"
            r"\frac{\partial \ell_{\rm obs}}{\partial y}$",
            "Spatial EL Evaluation In Y-Direction",
        ),
        (
            "lhs_omega", "rhs_omega",
            r"$\lambda_{\omega}\,\omega$",
            r"$\lambda_{\rm phys}\,r_\theta$",
            "Angular EL Evaluation",
        ),
        (
            "lhs_v", "rhs_v",
            r"$\lambda_v\,v$",
            r"$\lambda_{\rm phys}(r_x\cos\theta + r_y\sin\theta)$",
            "Velocity EL Evaluation",
        ),
    ]

    # ============================================================
    # Plot A: 5x2 figure (vertical layout — rows are quantities)
    # ============================================================
    fig, axes = plt.subplots(5, 2, figsize=(12, 22))

    # Row 0: Trajectories
    plot_trajectory(axes[0, 0], circ, "circle",    obs_circ, BC)
    plot_trajectory(axes[0, 1], rect, "rectangle", obs_rect, BC)

    # Rows 1-4: EL plots
    for row_idx, (lhs_key, rhs_key,
                  lhs_label, rhs_label, title) in enumerate(el_plots):
        for col, (data, obs_label) in enumerate([
            (circ, "Circle"),
            (rect, "Rectangle"),
        ]):
            plot_dual_axis(
                axes[row_idx + 1, col], t_np,
                data[lhs_key], data[rhs_key],
                lhs_label, rhs_label,
                f"{title} ({obs_label})"
            )

    fig.suptitle(
        "Euler--Lagrange Numerical Evaluation: "
        "Circle (Left) vs Rectangle (Right)",
        fontsize=16, y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(
        os.path.join(output_folder, "EL_combined_5x2.png"),
        dpi=300
    )
    plt.close()
    print("Saved 5x2 figure.")

    # ============================================================
    # Plot B: 2x5 figure (horizontal layout — rows are obstacles)
    # ============================================================
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))

    for row, (data, obs_type, obs_def) in enumerate([
        (circ, "circle",    obs_circ),
        (rect, "rectangle", obs_rect),
    ]):
        # Column 0: Trajectory
        plot_trajectory(axes[row, 0], data, obs_type, obs_def, BC)

        # Columns 1-4: EL plots
        for col, (lhs_key, rhs_key,
                  lhs_label, rhs_label, title) in enumerate(el_plots):
            plot_dual_axis(
                axes[row, col + 1], t_np,
                data[lhs_key], data[rhs_key],
                lhs_label, rhs_label, title
            )

    fig.suptitle(
        "Euler--Lagrange Numerical Evaluation: "
        "Circle (Top) vs Rectangle (Bottom)",
        fontsize=14, y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(
        os.path.join(output_folder, "EL_combined_2x5.png"),
        dpi=300
    )
    plt.close()
    print("Saved 2x5 figure.")


if __name__ == "__main__":
    main()