import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix — Gercekci Gemi Tipi Analizi ===")

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
print("Model loaded!\n")

# --- GERCEKCI Gemi Yakıt Profilleri ---
# Kaynak: Clarkson Research, MAN Energy Solutions, Lloyd's Register
vessel_types = {
    "VLCC Tanker": {
        "draft": 22.0, "speed": 15.0, "dwt": 300000,
        "real_daily_tons": 120,   # ton/gun (gercek olcum)
        "description": "Very Large Crude Carrier",
        "emoji": "🛢️"
    },
    "Panamax Konteyner": {
        "draft": 13.5, "speed": 20.0, "dwt": 65000,
        "real_daily_tons": 80,    # ton/gun
        "description": "Panama Kanali max konteyner",
        "emoji": "📦"
    },
    "Capesize Bulk": {
        "draft": 18.0, "speed": 14.5, "dwt": 180000,
        "real_daily_tons": 40,    # ton/gun
        "description": "Buyuk bulk carrier",
        "emoji": "⚓"
    },
    "LNG Carrier": {
        "draft": 12.0, "speed": 19.5, "dwt": 80000,
        "real_daily_tons": 65,    # ton/gun
        "description": "Sivilastirilmis dogal gaz",
        "emoji": "🔵"
    },
    "Kuru Yuk (Handy)": {
        "draft": 10.0, "speed": 14.0, "dwt": 35000,
        "real_daily_tons": 25,    # ton/gun
        "description": "Orta boy kuru yuk",
        "emoji": "📫"
    },
    "Feeder Konteyner": {
        "draft": 7.5, "speed": 16.0, "dwt": 15000,
        "real_daily_tons": 18,    # ton/gun
        "description": "Kucuk liman besleyici",
        "emoji": "🚢"
    },
    "RoRo Gemisi": {
        "draft": 8.0, "speed": 22.0, "dwt": 25000,
        "real_daily_tons": 35,    # ton/gun
        "description": "Arac tasiyan gemi",
        "emoji": "🚗"
    },
    "Karadeniz Kargo": {
        "draft": 8.5, "speed": 12.0, "dwt": 8000,
        "real_daily_tons": 12,    # ton/gun
        "description": "Tipik Karadeniz kargo — pilot hedef",
        "emoji": "🎯"
    },
}

def tahmin(lat, lon, depth, ssh, swh, speed, draft):
    inp = torch.tensor([[
        (lat+70)/150, (lon+180)/360, depth/6000,
        (ssh+2)/4, swh/20, speed/25, draft/22
    ]]).float()
    with torch.no_grad():
        return model(inp).item()

def gercekci_hesap(profile, drag):
    """
    Gercekci fuel ve savings hesabi
    Kaynak: Clarkson Research, DNV GL, Lloyd's Register
    """
    fuel_price = 650       # USD/ton (VLSFO 2026)
    voyage_days   = 280       # yillik sefer gunu
    savings_rate = max(0, (0.5 - drag) * 0.25)  # max %12.5

    # Gercek gunluk fuel (literaturden)
    gunluk_ton = profile["real_daily_tons"]

    annual_fuel_tons = gunluk_ton * voyage_days
    annual_fuel_usd = annual_fuel_tons * fuel_price
    savings_tons     = annual_fuel_tons * savings_rate
    savings_usd     = savings_tons * fuel_price
    batimetrix_gelir = savings_usd * 0.20  # tasarrufun %20si

    return {
        "gunluk_ton":       gunluk_ton,
        "yillik_usd":       annual_fuel_usd,
        "savings_rate":    savings_rate * 100,
        "savings_usd":     savings_usd,
        "batimetrix_gelir": batimetrix_gelir,
    }

# --- Karadeniz Sakin Senaryo ---
print("=" * 95)
print(f"{'Gemi Tipi':<22} {'Yakit/Gun':>10} {'Yillik Yakit':>13} "
      f"{'Tasarruf%':>10} {'Tasarruf$':>12} {'Batimetrix$':>12}")
print("=" * 95)

en_kazancli = None
en_kazancli_deger = 0
tum_sonuclar = {}

for name, profile in vessel_types.items():
    drag = tahmin(42.1, 31.5, 920, 0.08, 1.5,
                 profile["speed"], profile["draft"])
    h = gercekci_hesap(profile, drag)
    tum_sonuclar[name] = h

    if h["batimetrix_gelir"] > en_kazancli_deger:
        en_kazancli_deger = h["batimetrix_gelir"]
        en_kazancli = name

    emoji = profile["emoji"]
    print(f"{emoji} {name:<20} {h['gunluk_ton']:>8}t/gun "
          f"${h['yillik_usd']:>11,.0f} "
          f"%{h['savings_rate']:>8.1f} "
          f"${h['savings_usd']:>10,.0f} "
          f"${h['batimetrix_gelir']:>10,.0f}")

print("=" * 95)
print(f"\n🏆 En Kazancli: {en_kazancli} — "
      f"${en_kazancli_deger:,.0f}/yil Batimetrix geliri")

# --- Gercekcilik Notu ---
print("\n=== GERCEKCILIK NOTU ===")
print("Yakit degerleri: Clarkson Research + MAN Energy Solutions")
print("Fiyat: VLSFO $650/ton (2026 ortalama)")
print("Sefer gunu: 280/yil (liman + bakim duslulur)")
print("Tasarruf orani: %8-12 (literatur destekli)")
print("Batimetrix payi: Tasarrufun %20'si")

# --- Pitch Ozeti ---
print("\n=== PITCH OZETI (GERCEKCI) ===")
for name, h in tum_sonuclar.items():
    emoji = vessel_types[name]["emoji"]
    print(f"{emoji} {name:<22}: "
          f"${h['savings_usd']:>8,.0f} savings → "
          f"${h['batimetrix_gelir']:>7,.0f}/yil Batimetrix")

print("\nGercekci analiz completed!")
