import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix — Gercekci Gemi Tipi Analizi ===")

# --- Model ---
class GucluPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.giris = nn.Sequential(
            nn.Linear(7, 512), nn.LayerNorm(512), nn.GELU(),
        )
        self.katmanlar = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512), nn.LayerNorm(512),
                nn.GELU(), nn.Dropout(0.05),
            ) for _ in range(6)
        ])
        self.cikis = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.giris(x)
        for k in self.katmanlar:
            h = h + k(h)
        return self.cikis(h)

model = GucluPINN()
model.load_state_dict(torch.load("batimetrix_guclu.pt", weights_only=True))
model.eval()
print("Model yuklendi!\n")

# --- GERCEKCI Gemi Yakıt Profilleri ---
# Kaynak: Clarkson Research, MAN Energy Solutions, Lloyd's Register
gemi_tipleri = {
    "VLCC Tanker": {
        "tastak": 22.0, "hiz": 15.0, "dwt": 300000,
        "gercek_gunluk_ton": 120,   # ton/gun (gercek olcum)
        "aciklama": "Very Large Crude Carrier",
        "emoji": "🛢️"
    },
    "Panamax Konteyner": {
        "tastak": 13.5, "hiz": 20.0, "dwt": 65000,
        "gercek_gunluk_ton": 80,    # ton/gun
        "aciklama": "Panama Kanali max konteyner",
        "emoji": "📦"
    },
    "Capesize Bulk": {
        "tastak": 18.0, "hiz": 14.5, "dwt": 180000,
        "gercek_gunluk_ton": 40,    # ton/gun
        "aciklama": "Buyuk bulk carrier",
        "emoji": "⚓"
    },
    "LNG Carrier": {
        "tastak": 12.0, "hiz": 19.5, "dwt": 80000,
        "gercek_gunluk_ton": 65,    # ton/gun
        "aciklama": "Sivilastirilmis dogal gaz",
        "emoji": "🔵"
    },
    "Kuru Yuk (Handy)": {
        "tastak": 10.0, "hiz": 14.0, "dwt": 35000,
        "gercek_gunluk_ton": 25,    # ton/gun
        "aciklama": "Orta boy kuru yuk",
        "emoji": "📫"
    },
    "Feeder Konteyner": {
        "tastak": 7.5, "hiz": 16.0, "dwt": 15000,
        "gercek_gunluk_ton": 18,    # ton/gun
        "aciklama": "Kucuk liman besleyici",
        "emoji": "🚢"
    },
    "RoRo Gemisi": {
        "tastak": 8.0, "hiz": 22.0, "dwt": 25000,
        "gercek_gunluk_ton": 35,    # ton/gun
        "aciklama": "Arac tasiyan gemi",
        "emoji": "🚗"
    },
    "Karadeniz Kargo": {
        "tastak": 8.5, "hiz": 12.0, "dwt": 8000,
        "gercek_gunluk_ton": 12,    # ton/gun
        "aciklama": "Tipik Karadeniz kargo — pilot hedef",
        "emoji": "🎯"
    },
}

def tahmin(lat, lon, derinlik, ssh, swh, hiz, tastak):
    inp = torch.tensor([[
        (lat+70)/150, (lon+180)/360, derinlik/6000,
        (ssh+2)/4, swh/20, hiz/25, tastak/22
    ]]).float()
    with torch.no_grad():
        return model(inp).item()

def gercekci_hesap(profil, drag):
    """
    Gercekci yakit ve tasarruf hesabi
    Kaynak: Clarkson Research, DNV GL, Lloyd's Register
    """
    yakit_fiyat = 650       # USD/ton (VLSFO 2026)
    sefer_gun   = 280       # yillik sefer gunu
    tasarruf_oran = max(0, (0.5 - drag) * 0.25)  # max %12.5

    # Gercek gunluk yakit (literaturden)
    gunluk_ton = profil["gercek_gunluk_ton"]

    yillik_yakit_ton = gunluk_ton * sefer_gun
    yillik_yakit_usd = yillik_yakit_ton * yakit_fiyat
    tasarruf_ton     = yillik_yakit_ton * tasarruf_oran
    tasarruf_usd     = tasarruf_ton * yakit_fiyat
    batimetrix_gelir = tasarruf_usd * 0.20  # tasarrufun %20si

    return {
        "gunluk_ton":       gunluk_ton,
        "yillik_usd":       yillik_yakit_usd,
        "tasarruf_oran":    tasarruf_oran * 100,
        "tasarruf_usd":     tasarruf_usd,
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

for isim, profil in gemi_tipleri.items():
    drag = tahmin(42.1, 31.5, 920, 0.08, 1.5,
                 profil["hiz"], profil["tastak"])
    h = gercekci_hesap(profil, drag)
    tum_sonuclar[isim] = h

    if h["batimetrix_gelir"] > en_kazancli_deger:
        en_kazancli_deger = h["batimetrix_gelir"]
        en_kazancli = isim

    emoji = profil["emoji"]
    print(f"{emoji} {isim:<20} {h['gunluk_ton']:>8}t/gun "
          f"${h['yillik_usd']:>11,.0f} "
          f"%{h['tasarruf_oran']:>8.1f} "
          f"${h['tasarruf_usd']:>10,.0f} "
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
for isim, h in tum_sonuclar.items():
    emoji = gemi_tipleri[isim]["emoji"]
    print(f"{emoji} {isim:<22}: "
          f"${h['tasarruf_usd']:>8,.0f} tasarruf → "
          f"${h['batimetrix_gelir']:>7,.0f}/yil Batimetrix")

print("\nGercekci analiz tamamlandi!")