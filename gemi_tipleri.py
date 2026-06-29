import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix — Gemi Tipi Analizi ===")

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

# --- Gemi Profilleri ---
gemi_tipleri = {
    "Panamax Konteyner": {
        "tastak": 13.5, "hiz": 20.0,
        "uzunluk": 294, "dwt": 65000,
        "aciklama": "Panama Kanali max boyut konteyner",
        "emoji": "📦"
    },
    "VLCC Tanker": {
        "tastak": 22.0, "hiz": 15.0,
        "uzunluk": 333, "dwt": 300000,
        "aciklama": "Very Large Crude Carrier — ham petrol",
        "emoji": "🛢️"
    },
    "LNG Carrier": {
        "tastak": 12.0, "hiz": 19.5,
        "uzunluk": 290, "dwt": 80000,
        "aciklama": "Sivilastirilmis dogal gaz tanker",
        "emoji": "🔵"
    },
    "Capesize Bulk": {
        "tastak": 18.0, "hiz": 14.5,
        "uzunluk": 300, "dwt": 180000,
        "aciklama": "Panama gececek kadar buyuk bulk carrier",
        "emoji": "⚓"
    },
    "Feeder Konteyner": {
        "tastak": 7.5, "hiz": 16.0,
        "uzunluk": 150, "dwt": 15000,
        "aciklama": "Kucuk liman besleyici konteyner",
        "emoji": "🚢"
    },
    "RoRo Gemisi": {
        "tastak": 8.0, "hiz": 22.0,
        "uzunluk": 200, "dwt": 25000,
        "aciklama": "Roll-on Roll-off arac tasiyan gemi",
        "emoji": "🚗"
    },
    "Kuru Yuk (Handy)": {
        "tastak": 10.0, "hiz": 14.0,
        "uzunluk": 180, "dwt": 35000,
        "aciklama": "Orta boy kuru yuk gemisi",
        "emoji": "📫"
    },
    "Ozgun Gemi (Batimetrix Test)": {
        "tastak": 8.5, "hiz": 12.0,
        "uzunluk": 140, "dwt": 8000,
        "aciklama": "Karadeniz kargo gemisi — pilot hedef",
        "emoji": "🎯"
    },
}

# --- Test Senaryolari ---
senaryolar = [
    ("Karadeniz - Sakin",      42.1, 31.5,  920, 0.08, 1.2),
    ("Karadeniz - Firtina",    42.3, 32.0, 1100, 0.18, 4.5),
    ("Istanbul Bogazi",        41.1, 29.0,   35, 0.05, 0.4),
    ("Atlantik - Derin",       45.0,-20.0, 3200, 0.35, 7.2),
    ("Akdeniz - Yazin",        36.5, 18.0,  800, 0.06, 1.0),
    ("Hint Okyanusu",          10.0, 70.0, 2800, 0.22, 2.5),
]

def tahmin(lat, lon, derinlik, ssh, swh, hiz, tastak):
    inp = torch.tensor([[
        (lat+70)/150, (lon+180)/360, derinlik/6000,
        (ssh+2)/4, swh/20, hiz/25, tastak/22
    ]]).float()
    with torch.no_grad():
        return model(inp).item()

def yillik_yakit_tasarrufu(drag, dwt, hiz):
    # Basit yakit modeli: DWT ve hiz bazli
    # Gercek degerler gemi olcumleriyle kalibre edilmeli
    gunluk_yakit_ton = dwt * 0.000012 * (hiz ** 2.5)
    yillik_yakit_ton = gunluk_yakit_ton * 280  # 280 sefer gunu
    yillik_yakit_usd = yillik_yakit_ton * 650  # ton basi 650 USD
    tasarruf_oran = max(0, (0.5 - drag) * 0.30)  # maks %15
    tasarruf_usd = yillik_yakit_usd * tasarruf_oran
    return yillik_yakit_usd, tasarruf_usd

# --- Sonuclar ---
print("=" * 90)
print(f"{'Senaryo':<25} {'Gemi Tipi':<22} {'Drag':>7} {'Tasarruf%':>9} {'Yillik $':>12} {'Tasarruf$':>11}")
print("=" * 90)

ozet = {}
for isim, lat, lon, derinlik, ssh, swh in senaryolar:
    print(f"\n📍 {isim}")
    print("-" * 90)
    for gemi_adi, profil in gemi_tipleri.items():
        drag = tahmin(lat, lon, derinlik, ssh, swh,
                     profil["hiz"], profil["tastak"])
        tasarruf_pct = max(0, (0.5 - drag) * 30)
        yakit_usd, tasarruf_usd = yillik_yakit_tasarrufu(
            drag, profil["dwt"], profil["hiz"])

        emoji = profil["emoji"]
        print(f"  {emoji} {gemi_adi:<20} "
              f"Drag: {drag:.4f}  "
              f"Tasarruf: %{tasarruf_pct:>5.1f}  "
              f"Yakit: ${yakit_usd:>8,.0f}  "
              f"Kazan: ${tasarruf_usd:>8,.0f}")

        if gemi_adi not in ozet:
            ozet[gemi_adi] = []
        ozet[gemi_adi].append((drag, tasarruf_pct, tasarruf_usd))

# --- Genel Ozet ---
print("\n" + "=" * 90)
print("=== GEMI TIPI BAZLI OZET (TUM SENARYOLAR ORTALAMASI) ===")
print("=" * 90)
print(f"\n{'Gemi Tipi':<28} {'DWT':>8} {'Hiz':>6} "
      f"{'Ort.Drag':>9} {'Ort.Tasarruf%':>13} {'Ort.Kazan$/yil':>15}")
print("-" * 90)

en_kazancli = None
en_kazancli_deger = 0

for gemi_adi, profil in gemi_tipleri.items():
    veriler = ozet[gemi_adi]
    ort_drag = sum(v[0] for v in veriler) / len(veriler)
    ort_tasarruf = sum(v[1] for v in veriler) / len(veriler)
    ort_kazan = sum(v[2] for v in veriler) / len(veriler)

    if ort_kazan > en_kazancli_deger:
        en_kazancli_deger = ort_kazan
        en_kazancli = gemi_adi

    emoji = profil["emoji"]
    print(f"  {emoji} {gemi_adi:<26} "
          f"{profil['dwt']:>7,}  "
          f"{profil['hiz']:>5.1f}k  "
          f"{ort_drag:>8.4f}  "
          f"%{ort_tasarruf:>11.1f}  "
          f"${ort_kazan:>13,.0f}")

print(f"\n🏆 En Kazancli Gemi Tipi: {en_kazancli}")
print(f"   Ortalama Yillik Tasarruf: ${en_kazancli_deger:,.0f}")
print(f"\n💡 Batimetrix Fiyatlandirma Onerisı:")
print(f"   Tasarrufun %20'si = Gemi basina yillik abonelik")

for gemi_adi, profil in gemi_tipleri.items():
    veriler = ozet[gemi_adi]
    ort_kazan = sum(v[2] for v in veriler) / len(veriler)
    fiyat = ort_kazan * 0.20
    print(f"   {profil['emoji']} {gemi_adi:<26}: ${fiyat:>10,.0f}/yil")

print("\nGemi tipi analizi tamamlandi!")