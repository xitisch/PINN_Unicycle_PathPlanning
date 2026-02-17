import PINNs_wConstraints
import torch
import torch.nn as nn

softplus = nn.Softplus()
torch.linspace()
list = torch.linspace(0.0, T, N, device=device).view(-1, 1)
ox1 = softplus(list)
ox = torch.relu(list)

print(ox1)
print(ox)