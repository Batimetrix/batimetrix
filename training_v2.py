import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

print("=== Batimetrix — Optimize Egitim ===")
print("100.000 veri | 100 epoch | 1.6M parametre")
print("Tahmini sure: 45-60 dakika\n")

np.random.seed(2026)
n = 100_000

print("100.000 okyanus noktasi uretiliyor...")
lat   = np.random.uniform(-70, 80, n)
lon   = np.random.uniform(-180, 180, n)
depth = np.random.exponential(800, n).clip(5, 6000)
ssh   = np.random.normal(0, 0.2, n).clip(-2, 2)
swh   = np.random.exponential(2.0, n).clip(0, 20)
speed = np.random.uniform(4, 25, n)
draft = np.random.uniform(3, 22, n)

speed_ms = speed * 0.5144
Re       = (speed_ms * draft / 1.35e-6).clip(1e6, 1e10)
Cf       = 0.075 / (np.log10(Re) - 2) ** 2
wave_f   = 1.0 + 0.20 * (swh / 3.0) ** 1.8
depth_f  = 1.0 + 0.5 * np.exp(-depth / 30.0)
lat_f    = 1.0 + 0.1 * np.abs(np.sin(np.radians(lat)))
drag_raw = Cf * speed_ms**2 * draft * wave_f * depth_f * lat_f
drag     = (drag_raw / np.percentile(drag_raw, 99)).clip(0, 1).astype(np.float32)
noise    = np.random.normal(0, 0.01, n).astype(np.float32)
drag     = np.clip(drag + noise, 0, 1)

X = np.column_stack([
    (lat + 70) / 150,
    (lon + 180) / 360,
    depth / 6000,
    (ssh + 2) / 4,
    swh / 20,
    speed / 25,
    draft / 22,
]).astype(np.float32)

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(drag)
print(f"Veri ready: {X_t.shape}\n")

class GucluPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(7, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.05),
            ) for _ in range(6)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input_layer(x)
        for katman in self.layers:
            h = h + katman(h)
        return self.output_layer(h)

model     = GucluPINN()
params    = sum(params.numel() for params in model.parameters())
print(f"Model: {params:,} parametre")

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
loss_fn   = nn.MSELoss()

print("Egitim baslıyor...\n")
batch     = 2048
n_batch   = len(X_t) // batch
en_iyi    = float('inf')
sabir     = 0
baslangic = time.time()

for epoch in range(1, 101):
    model.train()
    toplam = 0
    idx    = torch.randperm(len(X_t))
    X_s    = X_t[idx]
    y_s    = y_t[idx]

    for i in range(n_batch):
        xb   = X_s[i*batch:(i+1)*batch]
        yb   = y_s[i*batch:(i+1)*batch]
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        veri_kaybi = loss_fn(pred, yb)
        swh_b   = xb[:, 4]
        depth_b = xb[:, 2]
        fizik   = torch.mean(torch.relu(swh_b * 0.3 - pred))
        fizik  += torch.mean(torch.relu((1 - depth_b) * 0.2 - pred))
        kayip   = veri_kaybi + 0.05 * fizik
        kayip.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        toplam += kayip.item()

    scheduler.step()
    ort = toplam / n_batch

    if ort < en_iyi:
        en_iyi = ort
        sabir  = 0
        torch.save(model.state_dict(), "batimetrix_model_v2.pt")
    else:
        sabir += 1

    if epoch % 10 == 0:
        sure = time.time() - baslangic
        print(f"Epoch {epoch:3d}/100 | Kayip: {ort:.6f} | "
              f"En iyi: {en_iyi:.6f} | Sure: {sure:.0f}s")

    if sabir >= 15:
        print(f"\nErken durdurma: epoch {epoch}")
        break

model.load_state_dict(torch.load("batimetrix_model_v2.pt", weights_only=True))
model.eval()

print(f"\n=== FINAL TEST ===")
def tahmin(lat, lon, depth, ssh, swh, speed, draft):
    inp = torch.tensor([[
        (lat+70)/150, (lon+180)/360, depth/6000,
        (ssh+2)/4, swh/20, speed/25, draft/22
    ]])
    with torch.no_grad():
        return model(inp).item()

testler = [
    ("Karadeniz - Sakin",       42.1,  31.5,  920, 0.08, 1.2, 12.0, 8.5),
    ("Atlantik Firtina",        52.0, -20.0, 3200, 0.45, 8.5, 10.0, 10.0),
    ("Istanbul Bogazi",         41.1,  29.0,   35, 0.05, 0.3,  8.0,  7.0),
    ("Pasifik - Tam Guc",        0.0, 160.0, 4000, 0.18, 4.1, 20.0, 13.0),
    ("Kutup - Buz Denizi",      75.0,  15.0,  380, 0.35, 6.5,  6.0,  9.0),
]

print(f"\n{'Senaryo':<25} {'Drag':>8} {'Tasarruf':>10}")
print("-" * 45)
for name, la, lo, de, ss, sw, sp, dr in testler:
    drag = tahmin(la, lo, de, ss, sw, sp, dr)
    savings = max(0, (0.5 - drag) * 30)
    print(f"{name:<25} {drag:>8.4f} %{savings:>8.1f}")

print(f"\nEn iyi kayip : {en_iyi:.6f}")
print(f"Toplam sure  : {time.time()-baslangic:.0f} saniye")
print(f"Model        : batimetrix_model_v2.pt")
print("\nEgitim completed!")
