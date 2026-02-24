# Use for testing

import torch
import matplotlib.pyplot as plt
import numpy as np
from PINNs_functions import *
from training_NN import train_model

lambda_phys = 1
lambda_obs = 1
lambda_length = 0
lambda_omega = 1

t_list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_list.requires_grad_(True)

results = []

for r in radii:
    print(f"Now: r = {r} out of {radii[-1]}")
    for y_c in y_positions:

        print(f"Now: y_c = {y_c} out of {y_positions[-1]}")
        obs = [0.5, y_c, r]

        model = train_model(
            T=T,
            BC=[0, 0, 1, 0],
            obs_circ=obs,
            epochs=2000
        )

        kappa_max = compute_curvature(model, t_list=t_list, T=T, BC=[0, 0, 1, 0])
        results.append([r, y_c, kappa_max])