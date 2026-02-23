import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PINNs_functions import *
from training import train_model

def compute_curvature(model, t_list, T, BC):
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)
    x_tt = derivative(x_t, t_list)
    y_tt = derivative(y_t, t_list)

    safety = 0
    kappa = (x_t * y_tt - y_t * x_tt) / ((x_t**2 + y_t**2 + safety)**(3/2))

    return torch.max(torch.abs(kappa)).item()

radii = [0.05, 0.1, 0.15, 0.2]
x_positions = [0.3, 0.4, 0.5, 0.6]

results = []

for r in radii:
    for x_c in x_positions:

        obs = [x_c, 0.1, r]

        model = train_model(
            T=1,
            BC=[0, 0, 1, 0],
            obs_circ=obs,
            epochs=1000
        )

        kappa_max = compute_curvature(model)

        results.append([r, x_c, kappa_max])

data = np.array(results)