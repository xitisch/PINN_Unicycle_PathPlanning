import torch
import matplotlib.pyplot as plt
import numpy as np
from PINNs_functions import *
from training_NN import train_model
import os

def compute_curvature(model, t_list, T, BC):
    nn_input = model(t_list)
    x, y, _, _, _ = hard_bc_transform(t_list, nn_input, T, BC)

    x_t = derivative(x, t_list)
    y_t = derivative(y, t_list)
    x_tt = derivative(x_t, t_list)
    y_tt = derivative(y_t, t_list)

    safety = 1e-6
    kappa = (x_t * y_tt - y_t * x_tt) / ((x_t**2 + y_t**2 + safety)**(3/2))

    return torch.max(torch.abs(kappa)).item()

radii = [0.05, 0.1, 0.15, 0.2]
y_positions = [0.3, 0.4, 0.5, 0.6]

T = 1
N = 100

    lambda_phys=1,
    lambda_obs=1,
    lambda_length

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

data = np.array(results)

heatmap = np.zeros((len(radii), len(y_positions)))

for r, y_c, kappa in results:
    i = radii.index(r)
    j = y_positions.index(y_c)
    heatmap[i, j] = kappa

plt.figure(figsize=(6, 5))

im = plt.imshow(
    heatmap,
    origin='lower',
    aspect='auto',
    extent=[
        min(y_positions), max(y_positions),
        min(radii), max(radii)
    ]
)

plt.colorbar(im, label="Max curvature κ")
plt.xlabel("Obstacle y_position")
plt.ylabel("Obstacle radius")
plt.title("Curvature heatmap")

plt.show()

output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

for i, r in enumerate(radii):

    plt.figure(figsize=(6, 4))

    plt.plot(
        y_positions,
        heatmap[i, :],
        marker='o',
        linewidth=2
    )

    plt.xlabel("Obstacle y-position")
    plt.ylabel("Max curvature κ")
    plt.title(f"Curvature vs displacement (r = {r})")
    plt.grid(True)

    filename = os.path.join(output_folder, f"curvature_radius_{r}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filename}")

plt.figure(figsize=(6, 4))

for i, r in enumerate(radii):
    plt.plot(
        y_positions,
        heatmap[i, :],
        marker='o',
        label=f"r = {r}"
    )

plt.xlabel("Obstacle y-position")
plt.ylabel("Max curvature κ")
plt.title("Curvature vs displacement for different radii")
plt.legend()
plt.grid(True)

plt.savefig("figures/curvature_all_radii.png", dpi=300, bbox_inches='tight')
plt.close()