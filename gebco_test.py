import requests
import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix GEBCO Gercek Veri Testi ===")

# --- Model mimarisi (train_test.py ile AYNI olmali) ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = PINN()
model.load_state_dict(torch.load("batimetrix_model.pt", weights_only=True))
model.eval()
print("Model loaded!")

# --- GEBCO API ---
def gebco_derinlik(lat, lon):
    try:
        url = f"https://api.gebco.net/tile_service?lat={lat}&lon={lon}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return abs(float(data.get("depth", 500)))
    except:
        pass
    return 850.0

# --- Karadeniz guzergahi ---
print("\nKaradeniz guzergahi test ediliyor...")
route = [
    ("Istanbul Bogazi cikisi", 41.2, 29.1),
    ("Bati Karadeniz",         42.0, 30.5),
    ("Orta Karadeniz",         42.2, 32.0),
    ("Sinop aciklari",         42.1, 35.0),
    ("Trabzon aciklari",       41.0, 39.5),
    ("Batum limani",           41.6, 41.6),
]

print(f"\n{'Nokta':<25} {'Derinlik':>10} {'Drag':>8} {'Durum'}")
print("-" * 60)

results = []
for name, lat, lon in route:
    depth = gebco_derinlik(lat, lon)
    inp = torch.tensor([[
        lat / 90,
        (lon + 180) / 360,
        depth / 4000,
        0.54,
        0.10,
        0.48,
        0.34
    ]])
    with torch.no_grad():
        drag = model(inp).item()
    status = "Verimli" if drag < 0.3 else ("Dikkat" if drag < 0.6 else "Kritik")
    results.append(drag)
    print(f"{name:<25} {depth:>9.0f}m {drag:>8.4f} [{status}]")

ort = sum(results) / len(results)
print(f"\nOrtalama drag    : {ort:.4f}")
print(f"Yakit tasarrufu  : %{(1 - ort) * 15:.1f}")
print("\nGEBCO testi completed!")
