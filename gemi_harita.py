import folium
import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix — Gemi Tipi Harita Simulatoru ===")

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
print("Model yuklendi!")

def tahmin(lat, lon, derinlik, ssh, swh, hiz, tastak):
    inp = torch.tensor([[
        (lat+70)/150, (lon+180)/360, derinlik/6000,
        (ssh+2)/4, swh/20, hiz/25, tastak/22
    ]]).float()
    with torch.no_grad():
        return model(inp).item()

# --- Gemi Profilleri ---
gemiler = {
    "VLCC Tanker":        {"tastak": 22.0, "hiz": 15.0, "dwt": 300000, "renk": "#E74C3C", "icon": "tint"},
    "LNG Carrier":        {"tastak": 12.0, "hiz": 19.5, "dwt": 80000,  "renk": "#3498DB", "icon": "fire"},
    "Panamax Konteyner":  {"tastak": 13.5, "hiz": 20.0, "dwt": 65000,  "renk": "#F39C12", "icon": "cube"},
    "Capesize Bulk":      {"tastak": 18.0, "hiz": 14.5, "dwt": 180000, "renk": "#8E44AD", "icon": "anchor"},
    "Kuru Yuk (Handy)":   {"tastak": 10.0, "hiz": 14.0, "dwt": 35000,  "renk": "#27AE60", "icon": "ship"},
}

# --- Guzergahlar ---
guzergahlar = {
    "Istanbul - Novorossiysk (Tanker)": {
        "gemi": "VLCC Tanker",
        "renk": "#E74C3C",
        "noktalar": [
            ("Istanbul Bogazi",    41.10, 29.05,  35, 0.05, 0.4),
            ("Bati KB",           41.80, 30.50, 650, 0.08, 1.2),
            ("Orta KB",           42.10, 33.00,1100, 0.09, 1.8),
            ("Novorossiysk",      44.72, 37.77, 120, 0.06, 0.6),
        ]
    },
    "Odessa - Istanbul (Bulk)": {
        "gemi": "Capesize Bulk",
        "renk": "#8E44AD",
        "noktalar": [
            ("Odessa",            46.48, 30.73,  80, 0.07, 1.0),
            ("Bati KB",           44.00, 31.00, 800, 0.10, 2.1),
            ("Orta KB",           42.50, 32.00,1100, 0.09, 1.9),
            ("Istanbul Bogazi",   41.10, 29.05,  35, 0.05, 0.4),
        ]
    },
    "Istanbul - Trabzon (Kuru Yuk)": {
        "gemi": "Kuru Yuk (Handy)",
        "renk": "#27AE60",
        "noktalar": [
            ("Istanbul",          41.10, 29.05,  35, 0.05, 0.4),
            ("Zonguldak",         41.60, 31.80, 850, 0.08, 1.5),
            ("Sinop",             42.00, 35.10, 950, 0.10, 1.6),
            ("Trabzon",           41.00, 39.73, 200, 0.06, 0.8),
        ]
    },
    "Batumi - Constanta (LNG)": {
        "gemi": "LNG Carrier",
        "renk": "#3498DB",
        "noktalar": [
            ("Batumi",            41.65, 41.64, 150, 0.07, 0.9),
            ("Dogu KB",           42.00, 38.00, 900, 0.09, 1.7),
            ("Orta KB",           42.20, 33.00,1100, 0.08, 1.5),
            ("Constanta",         44.17, 28.65,  60, 0.06, 0.7),
        ]
    },
}

# --- Harita ---
harita = folium.Map(
    location=[42.5, 33.0],
    zoom_start=6,
    tiles="CartoDB dark_matter"
)

toplam_tasarruf_usd = 0
toplam_guzergah = 0

for guzergah_adi, veri in guzergahlar.items():
    gemi_adi = veri["gemi"]
    profil   = gemiler[gemi_adi]
    rota_renk = veri["renk"]
    koordinatlar = []
    guzergah_tasarruf = 0

    for isim, lat, lon, derinlik, ssh, swh in veri["noktalar"]:
        drag = tahmin(lat, lon, derinlik, ssh, swh,
                     profil["hiz"], profil["tastak"])
        tasarruf_pct = max(0, (0.5 - drag) * 30)
        gunluk_yakit = profil["dwt"] * 0.000012 * (profil["hiz"] ** 2.5)
        yillik_yakit_usd = gunluk_yakit * 280 * 650
        tasarruf_usd = yillik_yakit_usd * max(0, (0.5 - drag) * 0.30)
        guzergah_tasarruf += tasarruf_usd / len(veri["noktalar"])
        koordinatlar.append([lat, lon])

        # Nokta rengi
        if drag < 0.20:   nokta_renk = "#00FF88"
        elif drag < 0.35: nokta_renk = "#FFFF00"
        elif drag < 0.50: nokta_renk = "#FF8800"
        else:             nokta_renk = "#FF0000"

        popup_html = f"""
        <div style='font-family:Arial;width:220px;background:#1a1a2e;
                    color:white;padding:10px;border-radius:8px;
                    border:2px solid {rota_renk}'>
            <h4 style='color:{rota_renk};margin:0'>{isim}</h4>
            <hr style='border-color:{rota_renk};margin:5px 0'>
            <b>Gemi:</b> {gemi_adi}<br>
            <b>Drag:</b> {drag:.4f}<br>
            <b>Tasarruf:</b> %{tasarruf_pct:.1f}<br>
            <b>Yillik Kazan:</b> ${tasarruf_usd:,.0f}<br>
            <b>Derinlik:</b> {derinlik}m<br>
            <b>Durum:</b> {'✅ Verimli' if drag < 0.35 else '⚠️ Dikkat'}
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color=nokta_renk,
            fill=True,
            fill_color=nokta_renk,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{gemi_adi} | {isim} | Drag:{drag:.3f}"
        ).add_to(harita)

    # Rota cizgisi
    folium.PolyLine(
        koordinatlar,
        color=rota_renk,
        weight=4,
        opacity=0.8,
        tooltip=f"{guzergah_adi} — Yillik Tasarruf: ${guzergah_tasarruf:,.0f}"
    ).add_to(harita)

    toplam_tasarruf_usd += guzergah_tasarruf
    toplam_guzergah += 1

# --- Bilgi Paneli ---
panel_html = f"""
<div style='position:fixed;bottom:20px;left:20px;z-index:1000;
     background:rgba(10,22,40,0.92);color:white;padding:18px;
     border-radius:12px;border:2px solid #1ABC9C;font-family:Arial;
     min-width:260px'>
    <h3 style='margin:0 0 10px 0;color:#1ABC9C'>🚢 BATIMETRIX</h3>
    <b>Guzergah Sayisi:</b> {toplam_guzergah}<br>
    <b>Analiz Edilen Gemi:</b> {len(gemiler)} tip<br>
    <b>Toplam Yillik Tasarruf:</b><br>
    <span style='color:#00FF88;font-size:18px;font-weight:bold'>
        ${toplam_tasarruf_usd:,.0f}
    </span>
    <hr style='border-color:#1ABC9C;margin:8px 0'>
    <b>Guzergahlar:</b><br>
    <span style='color:#E74C3C'>━━</span> VLCC Tanker (Tanker)<br>
    <span style='color:#8E44AD'>━━</span> Capesize Bulk (Tahil)<br>
    <span style='color:#27AE60'>━━</span> Kuru Yuk Handy<br>
    <span style='color:#3498DB'>━━</span> LNG Carrier<br>
    <hr style='border-color:#1ABC9C;margin:8px 0'>
    <span style='color:#00FF88'>●</span> Verimli
    <span style='color:#FFFF00'>●</span> Normal
    <span style='color:#FF8800'>●</span> Dikkat
    <span style='color:#FF0000'>●</span> Kritik
</div>
"""
harita.get_root().html.add_child(folium.Element(panel_html))

harita.save("batimetrix_gemi_harita.html")
print(f"\nHarita kaydedildi: batimetrix_gemi_harita.html")
print(f"Toplam yillik tasarruf potansiyeli: ${toplam_tasarruf_usd:,.0f}")
print("\nTarayicida ac: batimetrix_gemi_harita.html")