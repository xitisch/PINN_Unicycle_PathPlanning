import PINN_wBC


# Training setup:
# Repeateldly adjusting the NN's parameters so that 
# its output trajectory satisfies physics and constraints by minimizing loss functions.

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# lr = eta, the factors that is multiplied with the gradient of the loss. 

lambda_phys = 1
lambda_ic = 1
lambda_optim = 0

num_epochs = 2000       # Num. of iterations of training
print_every = 200       # Print every 200 iterations

T = 1
N = 100

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)


# Define BC
x0, y0 = 0.0, 0.0
xT, yT = 1.0, 1.0

BC = [x0,y0,xT,yT]

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute losses
    L_phys = physics_loss(model, t_list, BC)

    loss = L_phys
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, loss.item())

t_eval = torch.linspace(0.0, T, 200, device=device).view(-1, 1)
with torch.no_grad():
    nn_input = model(t_eval)
    x, y, theta, v = hard_bc_transform(t_eval, nn_input, BC)

    # Check boundary conditions (should be exact up to float precision)
    t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    tT = torch.tensor([[T]], dtype=torch.float32, device=device)
    x0p, y0p, *_ = hard_bc_transform(t0, model(t0), BC)
    xTp, yTp, *_ = hard_bc_transform(tT, model(tT), BC)
    print("\nBC check:")
    print(f"x(0)={x0p.item():.6f}, y(0)={y0p.item():.6f}")
    print(f"x(T)={xTp.item():.6f}, y(T)={yTp.item():.6f}")

plt.figure()
plt.plot(x.cpu().numpy(), y.cpu().numpy())
plt.scatter([x0, xT], [y0, yT])
plt.title("PINN unicycle path (hard x,y boundary conditions)")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
plt.show()


def main():

