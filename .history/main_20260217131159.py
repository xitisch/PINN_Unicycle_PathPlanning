import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

softplus = nn.Softplus()
torch.linspace()
T = 1
N = 100
list = torch.linspace(-5.0, 5.0, 100)
ox1 = softplus(list)
ox = torch.relu(list)

print(ox1)
print(ox)