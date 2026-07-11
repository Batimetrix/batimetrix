import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=== Batimetrix Egitim Baslıyor ===")

# --- Sentetik veri uret ---
print("50.000 deniz noktası uretiliyor...")
np.random.seed(42)
n = 50000

lat   = np.random.uniform(25, 65, n)
lon   = np.random.uniform(-10, 45, n)
depth = np.random.exponential(500, n).clip(10, 4000)
ssh   = np.random.normal(0, 0.15, n).clip(-1, 1)
swh   = np.random.exponential(1.5, n).clip(0, 15)
speed = np.random.uniform(5, 22, n)
draft = np.random.uniform(4, 20, n)

# Fizik tabanli drag hesabi (ITTC 1957)
speed_ms = speed * 0.5144
Re       = speed_ms * draft / 1.35e-6
Cf       = 0.075 / (np.log10(Re + 1) - 2) ** 2
wave_f   = 1.0 + 0.15 * (swh / 3.0) ** 1.5
depth_f  = 1.0 + 0.3 * np.exp(-depth / 50.0)
drag_raw = Cf * speed_ms**2 * draft * wave_f * depth_f
drag     = (drag_raw / drag_raw.max()).clip(0, 1).astype(np.float32)

# Normalize et
X = np.column_stack([
    (lat   - 0)   / 90,
    (lon   + 180) / 360,
    depth         / 4000,
    (ssh   + 1)   / 2,
    swh           / 15,
    speed         / 25,
    draft         / 25,
]).astype(np.float32)

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(drag)
print(f"Veri ready: {X_t.shape}")

# --- Model ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 64),  nn.GELU(),
            nn.Linear(64, 1),    nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model     = PINN()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()
print(f"Model ready: {sum(params.numel() for params in model.parameters()):,} parametre")

# --- Egitim dongusu ---
print("\nEgitim baslıyor (50 epoch)...")
batch = 512
n_batch = len(X_t) // batch

for epoch in range(1, 51):
    model.train()
    total_loss = 0
    idx = torch.randperm(len(X_t))
    X_s = X_t[idx]
    y_s = y_t[idx]

    for i in range(n_batch):
        xb = X_s[i*batch:(i+1)*batch]
        yb = y_s[i*batch:(i+1)*batch]
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        avg = total_loss / n_batch
        print(f"Epoch {epoch:3d}/50 | Kayip: {avg:.6f}")

# --- Test ---
print("\n=== Gemi Testi ===")
model.eval()
scenarios = [
    ("Karadeniz - sakin deniz",  42.1, 31.5, 850,  0.05, 0.8, 12, 8.5),
    ("Karadeniz - firtina",      42.1, 31.5, 850,  0.15, 4.2, 10, 8.5),
    ("Akdeniz - sığ su",         36.5, 28.2, 45,   0.02, 1.1, 14, 9.0),
    ("Atlantik - derin okyanus", 45.0, -20.0, 3800, 0.20, 6.5, 18, 12.0),
]

for name, lat, lon, depth, ssh, swh, speed, draft in scenarios:
    inp = torch.tensor([[
        lat/90, (lon+180)/360, depth/4000,
        (ssh+1)/2, swh/15, speed/25, draft/25
    ]])
    with torch.no_grad():
        skor = model(inp).item()
    status = "Verimli" if skor < 0.4 else ("Dikkat" if skor < 0.7 else "Kritik")
    print(f"{name:<35} Drag: {skor:.4f} [{status}]")

# Modeli kaydet
torch.save(model.state_dict(), "batimetrix_model.pt")
print("\nModel saved: batimetrix_model.pt")
print("Egitim completed!")
