import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

a = 19.05
b = 9.525
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def analytical_k2(m, n, a, b):
    return (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2

def uniform_domain_points(nx, ny):
    xs = np.linspace(0, a, nx)
    ys = np.linspace(0, b, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    X = torch.tensor(XX.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    Y = torch.tensor(YY.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    X.requires_grad_(True)
    Y.requires_grad_(True)
    return X, Y

def uniform_boundary_points(n_points):
    ys = np.linspace(0, b, n_points)
    xs = np.linspace(0, a, n_points)
    x0 = torch.zeros(n_points, 1, device=DEVICE)
    xa = torch.full((n_points, 1), a, device=DEVICE)
    yb = torch.tensor(ys, dtype=torch.float32, device=DEVICE).view(-1, 1)
    y0 = torch.zeros(n_points, 1, device=DEVICE)
    yb2 = torch.full((n_points, 1), b, device=DEVICE)
    xb2 = torch.tensor(xs, dtype=torch.float32, device=DEVICE).view(-1, 1)
    Xb = torch.cat([x0, xa, xb2, xb2], dim=0)
    Yb = torch.cat([yb, yb, y0, yb2], dim=0)
    Xb.requires_grad_(True)
    Yb.requires_grad_(True)
    return Xb, Yb

class PINN(nn.Module):
    def __init__(self, hidden_dim=50, hidden_layers=3, k2_value=None):
        super().__init__()
        layers = []
        in_dim = 2
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.k2 = k2_value
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)

def laplacian(u, x, y):
    grads = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    du_dx, du_dy = grads
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]
    return d2u_dx2 + d2u_dy2

def pde_loss(model, x, y):
    u = model(x, y)
    res = laplacian(u, x, y) + model.k2 * u
    return torch.mean(res**2)

def dirichlet_bc_loss(model, xb, yb):
    u_b = model(xb, yb)
    return torch.mean(u_b**2)

def multi_anchor_loss(model, anchor_coords, anchor_values):
    total = 0.0
    for (xi, yi), vi in zip(anchor_coords, anchor_values):
        xa = torch.tensor([[xi]], device=DEVICE, dtype=torch.float32, requires_grad=True)
        ya = torch.tensor([[yi]], device=DEVICE, dtype=torch.float32, requires_grad=True)
        u_pred = model(xa, ya)
        total += (u_pred - vi) ** 2
    return total / len(anchor_coords)

def train_PINN_fixed_k2_multi_anchors(
        m=1, n=1, epochs=4000, lr=1e-3, nx=60, ny=30, n_bnd=40,
        anchor_coords=None, anchor_values=None):
    k2_val = analytical_k2(m, n, a, b)
    model = PINN(hidden_dim=50, hidden_layers=3, k2_value=k2_val).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Xc, Yc = uniform_domain_points(nx, ny)
    Xb, Yb = uniform_boundary_points(n_bnd)

    for ep in range(epochs+1):
        optimizer.zero_grad()
        loss_pde = pde_loss(model, Xc, Yc)
        loss_bc = dirichlet_bc_loss(model, Xb, Yb)
        loss_anchor = multi_anchor_loss(model, anchor_coords, anchor_values)
        loss = loss_pde + loss_bc + 0.1 * loss_anchor
        loss.backward()
        optimizer.step()
        if ep % 1000 == 0:
            print(f"TE({m},{n})  Epoch {ep:>5d}  Total Loss={loss.item():.3e}")
    return model

def plot_field(model, m, n, analytical_k2_val, ax, idx):
    x = np.linspace(0, a, 100)
    y = np.linspace(0, b, 100)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Xtorch = torch.tensor(X.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    Ytorch = torch.tensor(Y.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    with torch.no_grad():
        Hz = model(Xtorch, Ytorch).cpu().numpy().reshape(100, 100)
    Hz_norm = Hz / np.abs(Hz).max()
    cp = ax.contourf(x, y, Hz_norm.T, levels=100, cmap='jet', vmin=-1, vmax=1)
    # Clean one-line title
    ax.set_title(f"TE({m},{n}),  $k^2={analytical_k2_val:.3f}$", fontsize=10)
    # Only show y-labels on leftmost column
    if idx % 4 == 0:
        ax.set_ylabel('y')
    else:
        ax.set_ylabel("")
    # Only show x-labels on bottom row
    if idx // 4 == 1:
        ax.set_xlabel('x')
    else:
        ax.set_xlabel("")
    return cp

def get_mode_anchors(m, n):
    coords = []
    values = []
    for i in range(m):
        for j in range(n):
            xi = (i + 0.5) * a / m
            yj = (j + 0.5) * b / n
            val = (-1) ** (i + j)
            coords.append((xi, yj))
            values.append(val)
    return coords, values

if __name__ == "__main__":
    # Your requested TE modes
    mode_list = [
        (1, 1), (2, 1), (3, 1), (1, 2),
        (4, 1), (2, 2), (3, 2), (5, 1)
    ]
    trained_models = []
    analytical_k2_vals = []

    for m, n in mode_list:
        anchor_coords, anchor_values = get_mode_anchors(m, n)
        print(f"\nTraining TE({m},{n}) mode with {len(anchor_coords)} anchors...")
        model = train_PINN_fixed_k2_multi_anchors(
            m, n, epochs=5000, lr=1e-3, nx=80, ny=40, n_bnd=40,
            anchor_coords=anchor_coords, anchor_values=anchor_values
        )
        trained_models.append(model)
        analytical_k2_vals.append(analytical_k2(m, n, a, b))

    # Plot all modes in a 2x4 grid with no colorbar overlap, and clean labels/titles
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for idx, (m, n) in enumerate(mode_list):
        ax = axes.flat[idx]
        plot_field(trained_models[idx], m, n, analytical_k2_vals[idx], ax, idx)
    fig.suptitle("First 8 TE$_{mn}$ Modes from PINNs (Fixed $k^2$)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.89, 0.95])
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    plt.colorbar(axes[0,0].collections[0], cax=cbar_ax, label='$H_z$ (normalized)')
    plt.show()