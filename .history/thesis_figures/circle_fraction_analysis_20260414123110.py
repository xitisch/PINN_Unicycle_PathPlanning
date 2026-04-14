import torch
import numpy as np
import matplotlib.pyplot as plt

from pinnlib.pinn_functions import *
from pinnlib.training_pinn import train_model


# =========================
# PARAMETERS
# =========================
T = 1.0
N = 100
BC = [0, 0, 1, 0]

lambda_phy = 1
lambda_obs = 10
lambda_smooth = 0.0001

# Circle parameters
r = 0.2
Delta = 0.1
x_c = 0.4
y_c = r - Delta   # intrusion definition

obs = [[x_c, y_c, r]]


# =========================
# TRAIN MODEL
# =========================
model = train_model(
    T=T,
    BC=BC,
    obs=obs,
    lambda_phy=lambda_phy,
    lambda_obs=lambda_obs,
    lambda_smooth=lambda_smooth,
    epochs=2000,
    N=N
)


# =========================
# EVALUATION GRID
# =========================
t_list = torch.linspace(0.0, T, N).view(-1, 1)
t_list.requires_grad_(True)

nn_input = model(t_list)
x, y, theta, v, omega = hard_bc_transform(t_list, nn_input, T, BC)


# =========================
# LOSSES (recompute)
# =========================
L_phy = phyics_loss(model, t_list, T, BC)
L_obs = circ_obs_loss(model, t_list, obs[0], T, BC)
L_smooth = smooth_loss(model, t_list, T, BC)

L_total = (
    lambda_phy * L_phy +
    lambda_obs * L_obs +
    lambda_smooth * L_smooth
)

frac_phy = (lambda_phy * L_phy / L_total).item()
frac_obs = (lambda_obs * L_obs / L_total).item()
frac_smooth = (lambda_smooth * L_smooth / L_total).item()

print("Loss fractions:")
print("Physics:", frac_phy)
print("Obstacle:", frac_obs)
print("Smooth:", frac_smooth)


# =========================
# CURVATURE
# =========================
x_t = derivative(x, t_list)
y_t = derivative(y, t_list)
x_tt = derivative(x_t, t_list)
y_tt = derivative(y_t, t_list)

eps = 1e-6
kappa = (x_t * y_tt - y_t * x_tt) / ((x_t**2 + y_t**2 + eps)**(3/2))


# =========================
# PLOTS
# =========================
x_np = x.detach().numpy()
y_np = y.detach().numpy()
k_np = kappa.detach().numpy()
t_np = t_list.detach().numpy()


# --- Trajectory ---
plt.figure()
plt.plot(x_np, y_np)

circle = plt.Circle((x_c, y_c), r, fill=False)
plt.gca().add_patch(circle)

plt.scatter([BC[0], BC[2]], [BC[1], BC[3]])
plt.title(f"Trajectory (Δ={Delta}, r={r})")
plt.axis("equal")
plt.grid()
plt.show()


# --- Curvature ---
plt.figure()
plt.plot(t_np, np.abs(k_np))
plt.title("Curvature over time")
plt.xlabel("t")
plt.ylabel("|kappa|")
plt.grid()
plt.show()


# --- Loss fractions ---
labels = ["Physics", "Obstacle", "Smooth"]
values = [frac_phy, frac_obs, frac_smooth]

plt.figure()
plt.bar(labels, values)
plt.title("Loss contribution fractions")
plt.ylabel("Fraction")
plt.show()