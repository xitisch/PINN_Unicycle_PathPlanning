from src.pinn.pinn_functions import *
import matplotlib.pyplot as plt

def plot_trajectory_live(model, t_list, T, BC, obs, epoch):
    plt.cla()

    # Forward pass
    nn_input = model(t_list)
    x, y, theta, v, omega = hard_bc_transform(t_list, nn_input, T, BC)

    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()

    # Trajectory
    plt.plot(x_np, y_np, linewidth=2.5)

    # Reference line
    plt.plot([BC[0], BC[2]], [BC[1], BC[3]], linestyle="--", alpha=0.5)

    # Obstacle visualization
    if obs is not None:
        for obstacle in obs:

            # --- Circle ---
            if len(obstacle) == 3:
                x_c, y_c, r = obstacle

                circle = plt.Circle(
                    (x_c, y_c), r,
                    edgecolor="black",
                    facecolor="#c6d6e3",
                    linewidth=2
                )
                plt.gca().add_patch(circle)

                plt.scatter(x_c, y_c, marker="x", s=60)

            # --- Rectangle ---
            elif len(obstacle) == 4:
                xmin, xmax, ymin, ymax = obstacle

                width  = xmax - xmin
                height = ymax - ymin

                rect = plt.Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    edgecolor="black",
                    facecolor="#f2c6c6",
                    linewidth=2
                )
                plt.gca().add_patch(rect)

                # center marker
                x_c = (xmin + xmax) / 2
                y_c = (ymin + ymax) / 2
                plt.scatter(x_c, y_c, marker="x", s=60)

            # --- Annotation (generic) ---
            plt.text(
                0.05, -0.28,
                f"obs={np.round(obstacle,2)}",
                fontsize=10
            )

    # Start / goal
    plt.scatter(BC[0], BC[1], s=50)
    plt.scatter(BC[2], BC[3], s=50)

    # Format
    plt.xlim(0-0.08, 1+0.08)
    plt.ylim(-0.4, 0.4)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal")

    plt.title(f"Epoch {epoch}")
    plt.pause(0.001)

def train_model(
    T,
    BC,
    obs=None,
    lambda_phy=1,
    lambda_obs=1,
    lambda_smooth=1,
    epochs=2000,
    lr=0.001,
    N=100
    ):
    
    torch.manual_seed(0)  
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    plt.ion()
    fig = plt.figure(figsize=(6, 5))

    t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
    t_list.requires_grad_(True)

    for epoch in range(epochs):
        optimizer.zero_grad()

        L_phy = phyics_loss(model, t_list, T, BC)

        L_obs = torch.tensor(0.0, device=device)

        if obs is not None:
            for obstacle in obs:

                if len(obstacle) == 3:
                    L_obs += circ_obs_loss(model, t_list, obstacle, T, BC)

                elif len(obstacle) == 4:
                    L_obs += rect_obs_loss(model, t_list, obstacle, T, BC)

        L_smooth = smooth_loss(model, t_list, T, BC)
        L_v0, L_theta0 = initial_condition_loss(model, T, BC)

        
        
        if epoch == 0:
            scale_phy = L_phy.detach()
            scale_obs = L_obs.detach()
            scale_smooth = L_smooth.detach()
            scale_v0 = L_v0.detach()
            scale_theta0 = L_theta0.detach()
        eps = 1e-8

        # Normalize
        L_phy_norm = L_phy / (scale_phy + eps)
        L_obs_norm = L_obs / (scale_obs + eps)
        L_smooth_norm = L_smooth / (scale_smooth + eps)
        L_v0_norm = L_v0 / (scale_v0 + eps)
L_theta0_norm = L_theta0 / (scale_theta0 + eps)

        loss = (
            lambda_phy * L_phy_norm
            + lambda_obs * L_obs_norm
            + lambda_smooth * L_smooth_norm
        )

        if epoch % 100 == 0:
            plot_trajectory_live(model, t_list, T, BC, obs, epoch)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(loss.item())
            print("L_phy:", lambda_phy * L_phy_norm.item())
            print("L_obs:", lambda_obs * L_obs_norm.item())
            print("L_smooth:", lambda_smooth * L_smooth_norm.item())
        loss.backward()
        optimizer.step()
    plt.ioff()
    plt.close(fig)

    return model