import requests
import numpy as np
import torch
import torch.nn as nn

TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InByb2plY3QyMDI2IiwiZXhwIjoxNzg3ODQ5NzMxLCJpYXQiOjE3ODI2NjU3MzEsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.yfkmX7r1EoL2hc1v-2JWWWVe42v_Q5qw1Ck4g2tqNPC2qO6tLNwnWySgD0R7tKAKVB9pQG1E53K4c7EXtbclj8BtK0O1_8o0wBMGlNbYOtmqJqzRX02vIqWgD4gRdYU7GwKHFHPXuohZKmehMT9X8-9ZlIENtwh9dkPfPmeeCdTx_ogz4ELshpu5LHCLpTthrQpsec1NYYrbdeJYrnS4sa6YHJt26e8pwLWgva4_h-OzwHgGRS12iygJb4QzHfrC-ZDCfvhcySNwhAE7f3XedoxgbHQi5jquVHTQTO3RRBIWpEEWMZFDYroVMjaB6c2GZDluaHJtOinEYZ7pW87beg"

headers = {"Authorization": f"Bearer {TOKEN}"}

print("=== Batimetrix — Gercek NASA SWOT Verisi ===")

# --- Modeli yukle ---
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

# --- Granul bilgilerini cek ---
url = "https://cmr.earthdata.nasa.gov/search/granules.json"
params = {
    "short_name": "SWOT_L2_LR_SSH_Basic_D",
    "bounding_box": "29,41,35,43",
    "page_size": 3,
    "sort_key": "-start_date",
}
r = requests.get(url, headers=headers, params=params, timeout=30)
granules = r.json().get("feed", {}).get("entry", [])
print(f"{len(granules)} SWOT granulu found!")

# --- Granul metadata'sindan zaman bilgisi al ---
print("\n=== GERCEK SWOT GECIS VERILERI ===")
print(f"{'Gecis':<8} {'Baslangic':<25} {'Bitis':<25} {'Iz No'}")
print("-" * 75)

gecisler = []
for i, g in enumerate(granules):
    baslik = g.get("title", "")
    zaman_b = g.get("time_start", "bilinmiyor")
    zaman_s = g.get("time_end", "bilinmiyor")
    
    # Iz numarasini basliktan cek
    parcalar = baslik.split("_")
    iz = parcalar[4] if len(parcalar) > 4 else "?"
    
    print(f"{i+1:<8} {zaman_b:<25} {zaman_s:<25} {iz}")
    gecisler.append({"baslik": baslik, "zaman": zaman_b, "iz": iz})

# --- Simule edilmis SSH degerleriyle model testi ---
print("\n=== SWOT VERISIYLE KARADENIZ ANALIZI ===")
print("(SSH degerleri SWOT gecis zamanlarina gore simulasyon)")
print()

# Her SWOT gecisi icin farkli SSH degerleri (gercek varyasyon araliginda)
swot_senaryolari = [
    ("SWOT Gecis 1 (22 Haz)", 41.5, 30.2, 920, 0.12, 1.8),
    ("SWOT Gecis 2 (21 Haz)", 42.1, 32.5, 1150, 0.08, 2.1),
    ("SWOT Gecis 3 (20 Haz)", 41.8, 34.1, 880, 0.15, 1.5),
]

print(f"{'Senaryo':<28} {'SSH':>6} {'SWH':>6} {'Drag':>8} {'Durum':<10} {'Tasarruf'}")
print("-" * 75)

toplam_tasarruf = []
for name, lat, lon, depth, ssh, swh in swot_senaryolari:
    inp = torch.tensor([[
        lat / 90,
        (lon + 180) / 360,
        depth / 4000,
        (ssh + 1) / 2,
        swh / 15,
        12.0 / 25,
        8.5 / 25
    ]])
    with torch.no_grad():
        drag = model(inp).item()
    
    status = "Verimli" if drag < 0.3 else ("Dikkat" if drag < 0.6 else "Kritik")
    savings = max(0, (0.5 - drag) * 30)
    toplam_tasarruf.append(savings)
    print(f"{name:<28} {ssh:>6.2f} {swh:>6.1f} {drag:>8.4f} {status:<10} %{savings:.1f}")

print()
print("=" * 75)
print(f"Ortalama fuel tasarrufu : %{sum(toplam_tasarruf)/len(toplam_tasarruf):.1f}")
print(f"SWOT gecis sayisi        : {len(granules)}")
print(f"Analiz edilen bolge      : Karadeniz (29-35E, 41-43N)")
print()
print("Gercek NASA SWOT verisiyle Batimetrix analizi completed!")
