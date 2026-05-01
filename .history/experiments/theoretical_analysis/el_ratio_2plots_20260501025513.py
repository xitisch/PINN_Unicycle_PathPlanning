# ============================================================
    # Figure 1: Trajectories (1x2)
    # ============================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))

    for col, (data, obs_type, obs_def) in enumerate([
        (circ, "circle",    obs_circ),
        (rect, "rectangle", obs_rect),
    ]):
        ax = axes1[col]

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
        ax.set_title(traj_title)

    fig1.suptitle(
        "Euler--Lagrange Numerical Evaluation: Trajectories",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "EL_trajectories.png"),
        dpi=300
    )
    plt.show()
    plt.close()
    print("Saved trajectory figure.")

    # ============================================================
    # Figure 2: EL plots (2x4)
    # ============================================================
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 8))

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

    for row, data in enumerate([circ, rect]):
        ax = axes2[row]
        row_label = "Circle" if row == 0 else "Rectangle"

        for col, (lhs_key, rhs_key,
                  lhs_label, rhs_label, title) in enumerate(el_plots):
            plot_dual_axis(
                ax[col], t_np,
                data[lhs_key], data[rhs_key],
                lhs_label, rhs_label,
                f"{title} ({row_label})"
            )

    fig2.suptitle(
        "Euler--Lagrange Numerical Evaluation: "
        "Circle (Top) vs Rectangle (Bottom)",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "EL_evaluation.png"),
        dpi=300
    )
    plt.show()
    plt.close()
    print("Saved EL evaluation figure.")