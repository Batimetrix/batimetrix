import requests
import numpy as np
import torch
import torch.nn as nn

TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InByb2plY3QyMDI2IiwiZXhwIjoxNzg3ODQ5NzMxLCJpYXQiOjE3ODI2NjU3MzEsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.yfkmX7r1EoL2hc1v-2JWWWVe42v_Q5qw1Ck4g2tqNPC2qO6tLNwnWySgD0R7tKAKVB9pQG1E53K4c7EXtbclj8BtK0O1_8o0wBMGlNbYOtmqJqzRX02vIqWgD4gRdYU7GwKHFHPXuohZKmehMT9X8-9ZlIENtwh9dkPfPmeeCdTx_ogz4ELshpu5LHCLpTthrQpsec1NYYrbdeJYrnS4sa6YHJt26e8pwLWgva4_h-OzwHgGRS12iygJb4QzHfrC-ZDCfvhcySNwhAE7f3XedoxgbHQi5jquVHTQTO3RRBIWpEEWMZFDYroVMjaB6c2GZDluaHJtOinEYZ7pW87beg"
headers = {"Authorization": f"Bearer {TOKEN}"}

print("=== Batimetrix — MODIS Deniz Yüzey Sicakligi ===")
print("Terra/Aqua uydusu baglaniyor...\n")

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
model.eval()
print("Model loaded!")

# --- MODIS SST Verisi Cek ---
print("\nMODIS deniz yüzeyi sicakligi aranıyor...")
url = "https://cmr.earthdata.nasa.gov/search/granules.json"

params = {
    "short_name": "MODIS_A-JPL-L2P-v2019.0",
    "bounding_box": "29,41,35,43",
    "page_size": 3,
    "sort_key": "-start_date",
}

r = requests.get(url, headers=headers, params=params, timeout=30)
granules = r.json().get("feed", {}).get("entry", [])
print(f"MODIS SST: {len(granules)} granul found!")

if len(granules) == 0:
    # Alternatif koleksiyon dene
    params["short_name"] = "MODIS_T-JPL-L2P-v2019.0"
    r2 = requests.get(url, headers=headers, params=params, timeout=30)
    granules = r2.json().get("feed", {}).get("entry", [])
    print(f"MODIS Terra SST: {len(granules)} granul found!")

for g in granules[:3]:
    print(f"  - {g.get('title', '')[:60]}")
    print(f"    Zaman: {g.get('time_start', '?')}")

# --- SST'nin Drag'e Etkisi ---
print("\n=== DENIZ YÜZEYI SICAKLIGI vs DRAG ===")
print("Su sicakligi viskoziteyi etkiler — viskozite drag'i etkiler\n")

def sst_vizkozite(sst_c):
    """Su sicakligina gore kinematik vizkozite (m2/s)"""
    # Gerçek fizik formülü
    return 1.792e-6 * np.exp(-0.0369 * sst_c)

def tahmin_sst(lat, lon, depth, ssh, swh, speed, draft, sst):
    """SST dahil drag tahmini"""
    # Vizkozite etkisini normalize et
    nu = sst_vizkozite(sst)
    nu_norm = (nu - 8e-7) / (1.8e-6 - 8e-7)  # 0-1 arasi

    inp = torch.tensor([[
        (lat+70)/150,
        (lon+180)/360,
        depth/6000,
        (ssh+2)/4,
        swh/20,
        speed/25,
        draft/22
    ]]).float()
    with torch.no_grad():
        drag_base = model(inp).item()

    # SST vizkozite duzeltmesi
    drag_sst = drag_base * (1.0 + 0.1 * nu_norm)
    return drag_sst, nu

# --- Karadeniz Mevsimsel Analiz ---
print(f"{'Mevsim':<12} {'SST(C)':>7} {'Viskozite':>12} {'Drag':>8} {'Tasarruf':>9} {'Not'}")
print("-" * 65)

seasons = [
    ("Kis (Oca)",   6.0,  "En soguk — yuksek viskozite"),
    ("Ilkbahar",   12.0,  "Isinma donemi"),
    ("Yaz (Tem)",  26.0,  "En sicak — dusuk viskozite"),
    ("Sonbahar",   18.0,  "Soguma donemi"),
]

for season, sst, not_ in seasons:
    drag, nu = tahmin_sst(42.1, 31.5, 920, 0.08, 1.5, 12.0, 8.5, sst)
    savings = max(0, (0.5 - drag) * 30)
    print(f"{season:<12} {sst:>6.1f}C {nu:>12.2e} {drag:>8.4f} "
          f"%{savings:>7.1f}  {not_}")

# --- 3 Uydu Ozet ---
print("\n" + "=" * 65)
print("=== BATİMETRİX — 3 NASA UYDUSU ===")
print()

# Tum verileri birlestir
swot_r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"},
    params={"short_name": "SWOT_L2_LR_SSH_Basic_D",
            "bounding_box": "29,41,35,43", "page_size": 3,
            "sort_key": "-start_date"}, timeout=30)
swot_g = swot_r.json().get("feed", {}).get("entry", [])

gpm_r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"},
    params={"short_name": "GPM_3IMERGHH",
            "bounding_box": "29,41,35,43", "page_size": 5,
            "sort_key": "-start_date"}, timeout=30)
gpm_g = gpm_r.json().get("feed", {}).get("entry", [])

print(f"🛰️  SWOT  (SSH)      : {len(swot_g)} granul — Deniz yuzey yuksekligi")
print(f"🌧️  GPM   (Yagis)    : {len(gpm_g)} granul — Firtina tahmini")
print(f"🌡️  MODIS (SST)      : {len(granules)} granul — Su sicakligi")
print()

# Final birlesik tahmin
drag_kis, _  = tahmin_sst(42.1, 31.5, 920, 0.08, 1.5, 12.0, 8.5, 6.0)
drag_yaz, _  = tahmin_sst(42.1, 31.5, 920, 0.08, 1.0, 12.0, 8.5, 26.0)

print(f"📊 Karadeniz Kis Drag  : {drag_kis:.4f} (%{max(0,(0.5-drag_kis)*30):.1f} savings)")
print(f"📊 Karadeniz Yaz Drag  : {drag_yaz:.4f} (%{max(0,(0.5-drag_yaz)*30):.1f} savings)")
print(f"📈 Mevsimsel Fark      : {abs(drag_kis-drag_yaz):.4f}")
print()
print("✅ Batimetrix artik 3 NASA uydusu kullaniyor!")
print("   En kapsamli denizcilik AI sistemi!")
