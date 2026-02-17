import PINNs_wConstraints
import torch
import torch.nn as nn

softplus = nn.Softplus()
ox1 = softplus(qx)
ox = torch.relu(qx)