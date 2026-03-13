from pinnlib.pinn_functions import *

def train_model(
    T,
    BC,
    obs=None,
    lambda_phys=1,
    lambda_obs=10,
    lambda_length=0,
    lambda_omega=0.0001,
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

        L_phys = physics_loss(model, t_list, T, BC)

        L_obs = 0
        if len(obs) == 3:
            L_obs = circ_obs_loss(model, t_list, obs, T, BC)
        if len(obs) == 4:
            L_obs = rect_obs_loss(model, t_list, obs, T, BC)

        L_length = length_loss(model, t_list, T, BC)

        L_omega = smooth_loss(model, t_list, T, BC)

        loss = (
            lambda_phys * L_phys
            + lambda_obs * L_obs
            + lambda_length * L_length
            + lambda_omega * L_omega
        )

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(loss.item())
            print("L_obs:", L_obs.item())
            print("L_phys:", L_phys.item())
            print("L_omega:", L_omega.item())
        loss.backward()
        optimizer.step()

    return model