import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define range
x = torch.linspace(-1, 1, 1000)

# Standard softplus
y_softplus = F.softplus(x)

# Your soft_relu (multiply by k, divide by k)
def soft_relu(x, k=20):
    return F.softplus(k * x) / k

# Softplus without division (for comparison)
def softplus_scaled(x, k=20):
    return F.softplus(x, beta=k)

k = 20
y_soft_relu = soft_relu(x, k)
y_softplus_scaled = softplus_scaled(x, k)

# Plot
plt.figure()
plt.plot(x.numpy(), y_softplus.numpy(), label="softplus(x)")
# plt.plot(x.numpy(), y_soft_relu.numpy(), label="soft_relu(x, k)")
plt.plot(x.numpy(), y_softplus_scaled.numpy(), label="softplus(kx)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Softplus Variants")
plt.legend()
plt.grid(True)
plt.show()