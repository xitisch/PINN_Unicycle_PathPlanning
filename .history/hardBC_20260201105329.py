import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = "cpu"

# -----------------------------
# Problem setup (choose any BCs)
# -----------------------------
T = 1.0
x0, y0 = 0.0, 0.0
xT, yT = 1.0, 1.0

# -----------------------------
# Simple MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=4, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        # Optional: small init helps stability in PINNs
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

# Network outputs: raw_xhat, raw_yhat, theta, v_raw
# We'll enforce x,y BCs via transformation; theta and v are free.
model = MLP(out_dim=4).to(device)

def hard_bc_transform(t, raw):
    """
    raw: [N,4] -> returns x(t), y(t), theta(t), v(t)
    x,y satisfy x(0)=x0, y(0)=y0, x(T)=xT, y(T)=yT exactly.
    """
    raw_xhat = raw[:, 0:1]
    raw_yhat = raw[:, 1:2]
    theta    = raw[:, 2:3]
    v_raw    = raw[:, 3:4]

    # Linear "bridge" between endpoints
    x_lin = x0 + (t / T) * (xT - x0)
    y_lin = y0 + (t / T) * (yT - y0)

    # Add a term that is zero at t=0 and t=T
    bump = t * (T - t)

    x = x_lin + bump * raw_xhat
    y = y_lin + bump * raw_yhat

    # Keep v bounded-ish (optional)
    v = torch.tanh(v_raw)  # v in (-1,1)

    return x, y, theta, v

def grad(outputs, inputs):
    """dy/dx with autograd; returns same shape as outputs."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

# -----------------------------
# Training data: collocation points in time
# -----------------------------
N = 256
t_col = torch.linspace(0.0, T, N, device=device).view(-1, 1)
t_col.requires_grad_(True)

# -----------------------------
# Loss: unicycle physics residuals (no data)
# xdot = v cos(theta), ydot = v sin(theta)
# -----------------------------
def pinn_loss():
    raw = model(t_col)
    x, y, theta, v = hard_bc_transform(t_col, raw)

    x_t = grad(x, t_col)
    y_t = grad(y, t_col)

    rx = x_t - v * torch.cos(theta)
    ry = y_t - v * torch.sin(theta)

    # Physics loss (mean squared residuals)
    L_phys = (rx**2).mean() + (ry**2).mean()

    # Optional regularization for smoother theta/v (helps training)
    theta_t = grad(theta, t_col)
    v_t = grad(v, t_col)
    L_reg = 1e-3 * (theta_t**2).mean() + 1e-3 * (v_t**2).mean()

    return L_phys + L_reg

# -----------------------------
# Optimizer
# -----------------------------
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Train
# -----------------------------
epochs = 5000
for ep in range(1, epochs + 1):
    opt.zero_grad()
    loss = pinn_loss()
    loss.backward()
    opt.step()

    if ep % 500 == 0:
        print(f"epoch {ep:5d} | loss = {loss.item():.4e}")

# -----------------------------
# Evaluate + plot
# -----------------------------
t_eval = torch.linspace(0.0, T, 200, device=device).view(-1, 1)
with torch.no_grad():
    raw = model(t_eval)
    x, y, theta, v = hard_bc_transform(t_eval, raw)

    # Check boundary conditions (should be exact up to float precision)
    t0 = torch.tensor([[0.0]], device=device)
    tT = torch.tensor([[T]], device=device)
    x0p, y0p, *_ = hard_bc_transform(t0, model(t0))
    xTp, yTp, *_ = hard_bc_transform(tT, model(tT))
    print("\nBC check:")
    print(f"x(0)={x0p.item():.6f}, y(0)={y0p.item():.6f}")
    print(f"x(T)={xTp.item():.6f}, y(T)={yTp.item():.6f}")

plt.figure()
plt.plot(x.cpu().numpy(), y.cpu().numpy())
plt.scatter([x0, xT], [y0, yT])
plt.title("PINN unicycle path (hard x,y boundary conditions)")
plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
plt.show()
