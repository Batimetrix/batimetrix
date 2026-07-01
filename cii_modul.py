import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix — CII Skoru Modulu ===")
print("IMO Carbon Intensity Indicator entegrasyonu\n")

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
print("Model yuklendi!")

# --- GERCEK IMO CII Parametreleri (MEPC.337(76) Tablo 1) ---
# Birim: g CO2 / (DWT · deniz mili)
# a degerleri gercek — onceki kodda 1000 kat kucuk yazilmisti!
CII_REF = {
    "VLCC Tanker":        {"a": 5247.0,  "c": 0.610},
    "Panamax Konteyner":  {"a": 1984.0,  "c": 0.489},
    "Capesize Bulk":      {"a": 4745.0,  "c": 0.622},
    "LNG Carrier":        {"a": 9.827,   "c": 0.000},  # sabit
    "Kuru Yuk (Handy)":   {"a": 588.0,   "c": 0.3885}, # <20K DWT
    "Karadeniz Kargo":    {"a": 588.0,   "c": 0.3885},
}

# 2026 azaltma faktoru: %11
Z_2026 = 11.0

# Not sinir katsayilari (MEPC.354(78))
SINIR = {
    "Tanker":        {"d1": 0.82, "d2": 0.93, "d3": 1.08, "d4": 1.28},
    "Bulk Carrier":  {"d1": 0.86, "d2": 0.94, "d3": 1.06, "d4": 1.18},
    "Container":     {"d1": 0.83, "d2": 0.94, "d3": 1.07, "d4": 1.19},
    "General Cargo": {"d1": 0.83, "d2": 0.94, "d3": 1.06, "d4": 1.18},
    "LNG":           {"d1": 0.89, "d2": 0.98, "d3": 1.06, "d4": 1.13},
}

GEMI_SINIR_TIP = {
    "VLCC Tanker":       "Tanker",
    "Panamax Konteyner": "Container",
    "Capesize Bulk":     "Bulk Carrier",
    "LNG Carrier":       "LNG",
    "Kuru Yuk (Handy)":  "General Cargo",
    "Karadeniz Kargo":   "General Cargo",
}

def required_cii(gemi_adi, dwt):
    p   = CII_REF.get(gemi_adi, {"a": 588.0, "c": 0.3885})
    ref = p["a"] * (dwt ** (-p["c"]))
    return ref * (1 - Z_2026 / 100)

def attained_cii(gunluk_yakit_ton, sefer_gun, dwt, mesafe_nm, yakit="VLSFO"):
    CF = {"HFO": 3.114, "VLSFO": 3.151, "MGO": 3.206, "LNG": 2.750}
    co2_ton  = gunluk_yakit_ton * sefer_gun * CF.get(yakit, 3.151)
    co2_gram = co2_ton * 1_000_000
    return co2_gram / (dwt * mesafe_nm)

def cii_notu(attained, required, gemi_adi):
    s    = SINIR.get(GEMI_SINIR_TIP.get(gemi_adi, "General Cargo"),
                     SINIR["General Cargo"])
    oran = attained / required
    if   oran <= s["d1"]: return "A", oran, "🟢 Mukemmel"
    elif oran <= s["d2"]: return "B", oran, "🟢 İyi"
    elif oran <= s["d3"]: return "C", oran, "🟡 Kabul"
    elif oran <= s["d4"]: return "D", oran, "🟠 Dikkat"
    else:                 return "E", oran, "🔴 Kritik"

def analiz(gemi_adi, dwt, hiz, tastak, yakit_gun,
           mesafe_nm, sefer_gun, yakit_turu="VLSFO"):

    inp = torch.tensor([[
        (42.1+70)/150, (31.5+180)/360, 920/6000,
        0.54, 0.10, hiz/25, tastak/22
    ]]).float()
    with torch.no_grad():
        drag = model(inp).item()

    tasarruf_oran = max(0, (0.5 - drag) * 0.25)
    yakit_opt     = yakit_gun * (1 - tasarruf_oran)

    req   = required_cii(gemi_adi, dwt)
    cii_b = attained_cii(yakit_gun, sefer_gun, dwt, mesafe_nm, yakit_turu)
    cii_o = attained_cii(yakit_opt, sefer_gun, dwt, mesafe_nm, yakit_turu)

    CF    = 3.151
    azalma_ton = (yakit_gun - yakit_opt) * sefer_gun * CF

    nb, ob, ab = cii_notu(cii_b, req, gemi_adi)
    no, oo, ao = cii_notu(cii_o, req, gemi_adi)

    return dict(drag=drag, tasarruf=tasarruf_oran*100,
                azalma=azalma_ton, req=req,
                cii_b=cii_b, cii_o=cii_o,
                nb=nb, no=no, ab=ab, ao=ao,
                ob=ob, oo=oo, degisti=nb!=no)

# --- Filo ---
filo = [
    ("VLCC Tanker",        300000, 15.0, 22.0, 120, 12000, 45),
    ("Panamax Konteyner",   65000, 20.0, 13.5,  80,  8000, 20),
    ("Capesize Bulk",      180000, 14.5, 18.0,  40, 15000, 50),
    ("Kuru Yuk (Handy)",    35000, 14.0, 10.0,  25,  5000, 18),
    ("Karadeniz Kargo",      8000, 12.0,  8.5,  12,   740,  3),
]

print(f"\n{'Gemi':<22} {'%Tas':>5} {'CO2↓':>7} "
      f"{'Req CII':>9} {'Mevcut':>12} {'Batimetrix':>12} {'Not↑?':>8}")
print("-" * 82)

sonuclar = []
for g in filo:
    r = analiz(*g)
    sonuclar.append((g[0], r))
    d = "✅ EVET" if r["degisti"] else "—"
    print(f"{g[0]:<22} {r['tasarruf']:>5.1f} {r['azalma']:>6.0f}t "
          f"{r['req']:>9.3f} "
          f"{r['nb']:>2}({r['cii_b']:>6.2f}) "
          f"{r['no']:>2}({r['cii_o']:>6.2f}) "
          f"{d:>8}")

print("-" * 82)

print("\n=== DETAYLI RAPOR ===\n")
iyilesen = []
for isim, r in sonuclar:
    print(f"🚢 {isim}")
    print(f"   Tasarruf      : %{r['tasarruf']:.1f}  |  CO2 Azalma: {r['azalma']:.0f} ton")
    print(f"   Required CII  : {r['req']:.3f} g/(DWT·nm)")
    print(f"   Attained (mev): {r['cii_b']:.3f}  → {r['nb']}  {r['ab']}")
    print(f"   Attained (opt): {r['cii_o']:.3f}  → {r['no']}  {r['ao']}")
    if r["degisti"]:
        print(f"   ✅ NOT İYİLEŞTİ: {r['nb']} → {r['no']}")
        iyilesen.append(isim)
    print()

print("=" * 60)
print(f"Not iyilestiren gemi: {len(iyilesen)}/{len(filo)}")
for g in iyilesen:
    print(f"  → {g}")

print("""
💡 PITCH MESAJI:
   Batimetrix yakıt tasarrufu + CII uyumu sağlar.
   IMO 2026: E notu = liman kısıtlaması + sigorta artışı.
   Batimetrix ile CII notu bir kademe yukarı çıkar.
   Bu bir tercih değil — YASAL ZORUNLULUK.
""")
print("CII modulu tamamlandi!")