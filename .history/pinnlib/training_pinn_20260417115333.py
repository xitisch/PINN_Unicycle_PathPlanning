from pinnlib.pinn_functions import *

def train_model(
    T,
    BC,
    obs=None,
    lambda_phy=1,
    lambda_obs=10,
    lambda_smooth=0.0001,
    epochs=2000,
    lr=0.001,
    N=100
    ):
    
    torch.manual_seed(0)  
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        
        if epoch == 0:
            scale_phy = L_phy.detach()
            scale_obs = L_obs.detach()
            scale_smooth = L_smooth.detach()

        eps = 1e-8

        # normalize
        L_phy_norm = L_phy / (scale_phy + eps)
        L_obs_norm = L_obs / (scale_obs + eps)
        L_smooth_norm = L_smooth / (scale_smooth + eps)

        loss = (
            lambda_phy * L_phy_norm
            + lambda_obs * L_obs_norm
            + lambda_smooth * L_smooth_norm
        )

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(loss.item())
            print("L_phy:", lambda_phy * L_phy_norm.item())
            print("L_obs:", lambda_obs * L_obs_norm.item())
            print("L_smooth:", lambda_smooth * L_smooth_norm.item())
        loss.backward()
        optimizer.step()

    return model