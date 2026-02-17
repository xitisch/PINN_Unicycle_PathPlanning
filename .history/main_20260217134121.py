import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

a = torch.randn(3, 3)

print(a.numpy())

compare = torch.logsumexp(a, 1)