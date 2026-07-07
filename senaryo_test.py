import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix — Gercek Dunya Senaryo Testi ===\n")

# --- Model ---
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

def tahmin(lat, lon, depth, ssh, swh, speed, draft):
    inp = torch.tensor([[
        lat/90,
        (lon+180)/360,
        depth/4000,
        (ssh+1)/2,
        swh/15,
        speed/25,
        draft/25
    ]])
    with torch.no_grad():
        return model(inp).item()

# --- Senaryolar ---
scenarios = [
    # (name, lat, lon, depth, ssh, swh, speed, draft, description)
    
    # NORMAL KOSULLAR
    ("Karadeniz - Sakin",        42.1, 31.5,  920, 0.08, 1.2, 12.0, 8.5,
     "Tipik yaz gunu, orta speed"),
    
    # FIRTINA SENARYOLARI
    ("Kuzey Atlantik - Firtina", 52.0, -20.0, 3200, 0.45, 8.5, 10.0, 10.0,
     "Siddetli firtina, dalga 8.5m"),
    ("Karadeniz - Firtina",      42.3, 32.0,  1100, 0.28, 5.2, 8.0,  8.5,
     "Kuzey firtinasi, dalga 5.2m"),
    ("Biscay Korfezi - Firtina", 46.0, -5.0,  2800, 0.52, 9.8, 7.0,  11.0,
     "En tehlikeli Avrupa denizi"),
    
    # SIG SU SENARYOLARI
    ("Istanbul Bogazi",          41.1, 29.0,   35, 0.05, 0.3, 8.0,  7.0,
     "Cok sig, akinti guclu"),
    ("Suez Kanali",              30.5, 32.3,   20, 0.02, 0.1, 6.0,  9.0,
     "Kanal gecisi, minimum depth"),
    ("Hollanda Kiyisi",          52.5,  4.5,   18, 0.06, 1.8, 10.0, 8.0,
     "Kuzey Denizi sigi suyu"),
    ("Marmara Denizi",           40.7, 28.0,   60, 0.04, 0.8, 10.0, 8.5,
     "Nispeten sig, yogun trafik"),
    
    # DERIN OKYANUS SENARYOLARI
    ("Atlantik - Derin",         35.0, -40.0, 4200, 0.15, 3.2, 18.0, 12.0,
     "Derin okyanus, yuksek speed"),
    ("Hint Okyanusu",            10.0,  70.0, 3800, 0.22, 2.8, 16.0, 11.0,
     "Tropikal derin okyanus"),
    ("Pasifik - En Derin",        0.0, 160.0, 4000, 0.18, 4.1, 20.0, 13.0,
     "En buyuk okyanus, tam guc"),
    
    # EKSTREM SENARYOLAR  
    ("Kutup - Buz Denizi",       75.0,  15.0,  380, 0.35, 6.5, 6.0,  9.0,
     "Arktik gecisi, buz riski"),
    ("Aden Korfezi - Sicak",     12.0,  48.0,  280, 0.31, 1.5, 14.0, 8.5,
     "Sicak su, yuksek intensity"),
    ("Capraz Firtina+Sig",       41.0,  29.5,   25, 0.42, 4.8, 7.0,  8.5,
     "En kotu scenario: firtina+sig"),
]

# --- Sonuclar ---
print(f"{'Senaryo':<30} {'Derinlik':>9} {'SWH':>5} {'Drag':>7} {'Durum':<10} {'Tasarruf':>9} {'Aciklama'}")
print("-" * 105)

en_iyi = None
en_kotu = None

for s in scenarios:
    name, lat, lon, depth, ssh, swh, speed, draft, description = s
    drag = tahmin(lat, lon, depth, ssh, swh, speed, draft)
    
    if drag < 0.25:
        status = "✓ Verimli"
        savings = (0.5 - drag) * 30
    elif drag < 0.5:
        status = "~ Normal"
        savings = max(0, (0.5 - drag) * 30)
    elif drag < 0.75:
        status = "! Dikkat"
        savings = 0
    else:
        status = "✗ Kritik"
        savings = 0
    
    print(f"{name:<30} {depth:>8}m {swh:>5.1f} {drag:>7.4f} {status:<10} %{savings:>7.1f}  {description}")
    
    if en_iyi is None or drag < en_iyi[0]:
        en_iyi = (drag, name, savings)
    if en_kotu is None or drag > en_kotu[0]:
        en_kotu = (drag, name)

print("-" * 105)
print(f"\n✓ EN VERIMLI : {en_iyi[1]} (Drag: {en_iyi[0]:.4f}, Tasarruf: %{en_iyi[2]:.1f})")
print(f"✗ EN RISKLI  : {en_kotu[1]} (Drag: {en_kotu[0]:.4f})")
print(f"\nBatimetrix {len(scenarios)} farkli scenario basariyla analiz etti!")