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

# =========================
# PARAMETERS
# =========================
T = 1.0
N = 500

BC = [0, 0, 1, 0]

lambda_phy = 1
lambda_obs = 10
lambda_smooth = 0.0001

# Circle parameters (intrusion-controlled)
r = 0.2
Delta = 0.1
x_c = 0.4
y_c = r - Delta

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
    epochs=5000,
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
# DERIVATIVES
# =========================
x_t = derivative(x, t_list)
y_t = derivative(y, t_list)
theta_t = derivative(theta, t_list)

# residuals
r_x = x_t - v * torch.cos(theta)
r_y = y_t - v * torch.sin(theta)
r_theta = theta_t - omega

# time derivatives of residuals
r_x_t = derivative(r_x, t_list)
r_y_t = derivative(r_y, t_list)


# =========================
# OBSTACLE GRADIENT
# =========================
d = torch.sqrt((x - x_c)**2 + (y - y_c)**2 + 1e-8)

buffer = 0.01
beta = 80

violation = torch.nn.functional.softplus((r - d + buffer), beta=beta)

# ℓ_obs = violation^2
l_obs = violation**2

dl_dx = derivative(l_obs, x)
dl_dy = derivative(l_obs, y)


# =========================
# EL RIGHT-HAND SIDE
# =========================
scale = lambda_obs / (2 * lambda_phy)

rhs_x = scale * dl_dx
rhs_y = scale * dl_dy


# =========================
# RATIOS
# =========================
eps = 1e-6

ratio_x = r_x_t / (rhs_x + eps)
ratio_y = r_y_t / (rhs_y + eps)

ratio_theta = r_theta / (omega + eps)


# =========================
# ERROR FORM (more stable)
# =========================
error_x = r_x_t - rhs_x
error_y = r_y_t - rhs_y


# =========================
# NUMPY CONVERSION
# =========================
t_np = t_list.detach().numpy().flatten()

ratio_x_np = ratio_x.detach().numpy().flatten()
ratio_y_np = ratio_y.detach().numpy().flatten()
ratio_theta_np = ratio_theta.detach().numpy().flatten()

error_x_np = error_x.detach().numpy().flatten()
error_y_np = error_y.detach().numpy().flatten()


# =========================
# VELOCITY EL: LHS vs RHS
# =========================

# LHS
lhs_v = lambda_smooth * v

# RHS
rhs_v = lambda_phy * (r_x * torch.cos(theta) + r_y * torch.sin(theta))

# Convert to numpy
lhs_v_np = lhs_v.detach().numpy().flatten()
rhs_v_np = rhs_v.detach().numpy().flatten()
t_np = t_list.detach().numpy().flatten()