import PINNs_wConstraints
import torch
import torch.nn as nn

softplus = nn.Softplus()
ox1 = softplus([0,1,2,3,4])
ox = torch.relu([0,1,2,3,4])

print(ox1)
ox