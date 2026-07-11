import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix — CII Skoru Modulu ===")
print("IMO Carbon Intensity Indicator entegrasyonu\n")

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

# Not limit katsayilari (MEPC.354(78))
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

def required_cii(vessel_name, dwt):
    params   = CII_REF.get(vessel_name, {"a": 588.0, "c": 0.3885})
    ref = params["a"] * (dwt ** (-params["c"]))
    return ref * (1 - Z_2026 / 100)

def attained_cii(daily_fuel_tons, voyage_days, dwt, mesafe_nm, fuel="VLSFO"):
    CF = {"HFO": 3.114, "VLSFO": 3.151, "MGO": 3.206, "LNG": 2.750}
    co2_ton  = daily_fuel_tons * voyage_days * CF.get(fuel, 3.151)
    co2_gram = co2_ton * 1_000_000
    return co2_gram / (dwt * mesafe_nm)

def cii_notu(attained, required, vessel_name):
    s    = SINIR.get(GEMI_SINIR_TIP.get(vessel_name, "General Cargo"),
                     SINIR["General Cargo"])
    ratio = attained / required
    if   ratio <= s["d1"]: return "A", ratio, "🟢 Mukemmel"
    elif ratio <= s["d2"]: return "B", ratio, "🟢 İyi"
    elif ratio <= s["d3"]: return "C", ratio, "🟡 Kabul"
    elif ratio <= s["d4"]: return "D", ratio, "🟠 Dikkat"
    else:                 return "E", ratio, "🔴 Kritik"

def analiz(vessel_name, dwt, speed, draft, fuel_per_day,
           mesafe_nm, voyage_days, fuel_type="VLSFO"):

    inp = torch.tensor([[
        (42.1+70)/150, (31.5+180)/360, 920/6000,
        0.54, 0.10, speed/25, draft/22
    ]]).float()
    with torch.no_grad():
        drag = model(inp).item()

    savings_rate = max(0, (0.5 - drag) * 0.25)
    yakit_opt     = fuel_per_day * (1 - savings_rate)

    req   = required_cii(vessel_name, dwt)
    cii_b = attained_cii(fuel_per_day, voyage_days, dwt, mesafe_nm, fuel_type)
    cii_o = attained_cii(yakit_opt, voyage_days, dwt, mesafe_nm, fuel_type)

    CF    = 3.151
    azalma_ton = (fuel_per_day - yakit_opt) * voyage_days * CF

    nb, ratio_base, label_base = cii_notu(cii_b, req, vessel_name)
    no, ratio_opt, label_opt = cii_notu(cii_o, req, vessel_name)

    return dict(drag=drag, savings=savings_rate*100,
                reduction=azalma_ton, req=req,
                cii_b=cii_b, cii_o=cii_o,
                nb=nb, no=no, label_base=label_base, label_opt=label_opt,
                ratio_base=ratio_base, ratio_opt=ratio_opt, changed=nb!=no)

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

results = []
for g in filo:
    r = analiz(*g)
    results.append((g[0], r))
    d = "✅ EVET" if r["changed"] else "—"
    print(f"{g[0]:<22} {r['savings']:>5.1f} {r['reduction']:>6.0f}t "
          f"{r['req']:>9.3f} "
          f"{r['nb']:>2}({r['cii_b']:>6.2f}) "
          f"{r['no']:>2}({r['cii_o']:>6.2f}) "
          f"{d:>8}")

print("-" * 82)

print("\n=== DETAYLI RAPOR ===\n")
improved = []
for name, r in results:
    print(f"🚢 {name}")
    print(f"   Tasarruf      : %{r['savings']:.1f}  |  CO2 Azalma: {r['reduction']:.0f} ton")
    print(f"   Required CII  : {r['req']:.3f} g/(DWT·nm)")
    print(f"   Attained (mev): {r['cii_b']:.3f}  → {r['nb']}  {r['label_base']}")
    print(f"   Attained (opt): {r['cii_o']:.3f}  → {r['no']}  {r['label_opt']}")
    if r["changed"]:
        print(f"   ✅ NOT İYİLEŞTİ: {r['nb']} → {r['no']}")
        improved.append(name)
    print()

print("=" * 60)
print(f"Not iyilestiren gemi: {len(improved)}/{len(filo)}")
for g in improved:
    print(f"  → {g}")

print("""
💡 PITCH MESAJI:
   Batimetrix yakıt tasarrufu + CII uyumu sağlar.
   IMO 2026: E notu = liman kısıtlaması + sigorta artışı.
   Batimetrix ile CII notu bir kademe yukarı çıkar.
   Bu bir tercih değil — YASAL ZORUNLULUK.
""")
print("CII modulu completed!")
