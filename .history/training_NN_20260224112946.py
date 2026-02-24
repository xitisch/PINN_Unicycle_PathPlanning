from PINNs_functions import *

def train_model(
    T,
    BC,
    obs_circ=None,
    lambda_phys=1,
    lambda_obs=1,
    lambda_length=0,
    lambda_omega=0,
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
        if obs_circ is not None:
            L_obs = circ_obs_loss(model, t_list, obs_circ, T, BC)

        L_length = length_loss(model, t_list, T, BC)

        L_omega = omega_loss(model, t_list, T, BC)

        loss = (
            lambda_phys * L_phys
            + lambda_obs * L_obs
            + lambda_length * L_length
            + lambda_omega * L_omega
        )

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(loss.item())
        loss.backward()
        optimizer.step()

    return model