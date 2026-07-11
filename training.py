import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

print("=== Batimetrix — Guclu Egitim ===")
print("500.000 veri | 200 epoch | 512 noron | 8 katman")
print("Bu biraz zaman alacak...\n")

# --- Buyuk Sentetik Veri Uret ---
print("500.000 dunya okyanus noktasi uretiliyor...")
np.random.seed(2026)
n = 500_000

lat   = np.random.uniform(-70, 80, n)
lon   = np.random.uniform(-180, 180, n)
depth = np.random.exponential(800, n).clip(5, 6000)
ssh   = np.random.normal(0, 0.2, n).clip(-2, 2)
swh   = np.random.exponential(2.0, n).clip(0, 20)
speed = np.random.uniform(4, 25, n)
draft = np.random.uniform(3, 22, n)

# Fizik tabanli drag (ITTC 1957 + dalga + batimetri + sicaklik etkisi)
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

# Normalize
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
print(f"Veri ready: {X_t.shape} | Drag aralik: {drag.min():.3f} - {drag.max():.3f}\n")

# --- Guclu Model ---
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
            h = h + katman(h)  # Artik baglanti
        return self.output_layer(h)

model     = GucluPINN()
params    = sum(params.numel() for params in model.parameters())
print(f"Model: {params:,} parametre ({params/1000:.0f}K)")

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
loss_fn   = nn.MSELoss()

# --- Egitim ---
print("Egitim baslıyor...\n")
batch      = 1024
n_batch    = len(X_t) // batch
en_iyi     = float('inf')
sabir      = 0
max_sabir  = 20
baslangic  = time.time()

for epoch in range(1, 201):
    model.train()
    toplam = 0
    idx    = torch.randperm(len(X_t))
    X_s    = X_t[idx]
    y_s    = y_t[idx]

    for i in range(n_batch):
        xb = X_s[i*batch:(i+1)*batch]
        yb = y_s[i*batch:(i+1)*batch]
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        
        # Veri kaybi
        veri_kaybi = loss_fn(pred, yb)
        
        # Fizik kaybi: yuksek dalga -> yuksek drag olmali
        swh_b  = xb[:, 4]
        depth_b = xb[:, 2]
        fizik  = torch.mean(torch.relu(swh_b * 0.3 - pred))
        fizik += torch.mean(torch.relu((1 - depth_b) * 0.2 - pred))
        
        kayip = veri_kaybi + 0.05 * fizik
        kayip.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        toplam +=kayip.item()
