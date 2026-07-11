import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("=== Batimetrix — GERCEK NASA SWOT Egitimi ===")

# --- Model ---
class GucluPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(7, 512), nn.LayerNorm(512), nn.GELU(),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512), nn.LayerNorm(512),
                nn.GELU(), nn.Dropout(0.05),
            ) for _ in range(6)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.input_layer(x)
        for k in self.layers:
            h = h + k(h)
        return self.output_layer(h)

model = GucluPINN()
model.load_state_dict(torch.load("batimetrix_model_v2.pt", weights_only=True))
print("Model loaded!")

# --- SWOT Verisi Oku ---
print("SWOT verisi okunuyor...")
ds = nc.Dataset("swot_blacksea.nc", "r")
lat  = np.array(ds.variables["latitude"][:],    dtype=np.float64)
lon  = np.array(ds.variables["longitude"][:],   dtype=np.float64)
ssh  = np.array(ds.variables["ssh_karin"][:],   dtype=np.float64)
ssha = np.array(ds.variables["ssha_karin"][:],  dtype=np.float64)
ds.close()

# Duzlestir
lat  = lat.flatten()
lon  = lon.flatten()
ssh  = ssh.flatten()
ssha = ssha.flatten()

# Temizle
mask = (np.isfinite(lat) & np.isfinite(lon) &
        np.isfinite(ssh) & np.isfinite(ssha) &
        (np.abs(ssh) < 200) & (np.abs(ssha) < 10))

lat  = lat[mask].astype(np.float32)
lon  = lon[mask].astype(np.float32)
ssh  = ssh[mask].astype(np.float32)
ssha = ssha[mask].astype(np.float32)

print(f"Temiz nokta: {len(lat):,}")
print(f"SSH aralik : {ssh.min():.3f} - {ssh.max():.3f} m")

# Ornekle (max 10000 nokta)
n_max = min(10000, len(lat))
idx   = np.random.choice(len(lat), n_max, replace=False)
lat   = lat[idx]; lon = lon[idx]
ssh   = ssh[idx]; ssha = ssha[idx]
print(f"Orneklenen : {n_max:,} nokta")

# --- Drag Hesapla ---
speed_ms = 12.0 * 0.5144
draft    = 8.5
depth    = 900.0
Re       = speed_ms * draft / 1.35e-6
Cf       = 0.075 / (np.log10(Re) - 2) ** 2
swh_est  = 1.5 + 0.3 * np.abs(ssha)
wave_f   = 1.0 + 0.20 * (swh_est / 3.0) ** 1.8
depth_f  = 1.0 + 0.5 * np.exp(-depth / 30.0)
drag_raw = Cf * speed_ms**2 * draft * wave_f * depth_f
drag_arr = (drag_raw / (drag_raw.max() + 1e-9)).clip(0, 1).astype(np.float32)

# --- Tensore Cevir ---
X = np.column_stack([
    (lat + 70) / 150,
    (lon + 180) / 360,
    np.full(n_max, depth / 6000),
    np.clip((ssha + 2) / 4, 0, 1),
    np.clip(swh_est / 20, 0, 1),
    np.full(n_max, 12.0 / 25),
    np.full(n_max, draft / 22),
]).astype(np.float32)

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(drag_arr)
print(f"Tensör: {X_t.shape}")

# --- Fine-Tuning ---
print("\nNASA SWOT ile fine-tuning (30 epoch)...")
model.train()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)
loss_fn   = nn.MSELoss()
batch     = 256
n_batch   = max(1, len(X_t) // batch)

for epoch in range(1, 31):
    toplam = 0
    idx2   = torch.randperm(len(X_t))
    X_s    = X_t[idx2]; y_s = y_t[idx2]
    for i in range(n_batch):
        xb = X_s[i*batch:(i+1)*batch]
        yb = y_s[i*batch:(i+1)*batch]
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        toplam += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}/30 | Kayip: {toplam/n_batch:.6f}")

torch.save(model.state_dict(), "batimetrix_swot_real.pt")
print("\nModel: batimetrix_swot_real.pt")

# --- Final Test ---
model.eval()
print("\n=== GERCEK SWOT NOKTALARI TEST ===")
print(f"{'Koordinat':<25} {'SSH':>8} {'SSHA':>8} {'Drag':>8} {'Tasarruf':>10}")
print("-" * 62)

for i in range(min(5, len(lat))):
    la = float(lat[i]); lo = float(lon[i])
    s  = float(ssh[i]); sa = float(ssha[i])
    sw = float(swh_est[i])
    inp = torch.tensor([[
    (la+70)/150, (lo+180)/360, depth/6000,
    np.clip((sa+2)/4, 0, 1), np.clip(sw/20, 0, 1),
    12.0/25, draft/22
]]).float()    
    with torch.no_grad():
        drag = model(inp).item()
    savings = max(0, (0.5 - drag) * 30)
    print(f"({la:.2f}N {lo:.2f}E)          "
          f"{s:>8.3f} {sa:>8.4f} {drag:>8.4f} %{savings:>8.1f}")

print(f"\nKullanilan gercek NASA SWOT noktasi: {n_max:,}")
print("GERCEK NASA SWOT verisiyle egitim completed! 🚀")
