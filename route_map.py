import folium
import torch
import torch.nn as nn
import numpy as np

print("=== Batimetrix Guzergah Harita Simulatoru ===")

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

def tahmin(lat, lon, depth, ssh, swh, speed=12.0, draft=8.5):
    inp = torch.tensor([[
        (lat+70)/150, (lon+180)/360, depth/6000,
        (ssh+2)/4, swh/20, speed/25, draft/22
    ]]).float()
    with torch.no_grad():
        return model(inp).item()

# --- Istanbul - Trabzon Guzergahi ---
route = [
    ("Istanbul Bogazi",    41.10, 29.05,  35, 0.05, 0.4),
    ("Karadeniz Girisi",   41.30, 29.50, 120, 0.07, 0.8),
    ("Bati Karadeniz",     41.80, 30.50, 650, 0.08, 1.2),
    ("Zonguldak Aciklari", 41.60, 31.80, 850, 0.09, 1.5),
    ("Orta Karadeniz",     42.10, 32.50,1100, 0.08, 1.8),
    ("Sinop Burnu",        42.00, 35.10, 950, 0.10, 1.6),
    ("Samsun Aciklari",    41.70, 36.20, 800, 0.11, 1.9),
    ("Ordu Aciklari",      41.10, 37.80, 750, 0.09, 1.7),
    ("Giresun Aciklari",   41.00, 38.50, 700, 0.08, 1.5),
    ("Trabzon Aciklari",   41.00, 39.50, 650, 0.10, 1.8),
    ("Trabzon Limani",     41.00, 39.73, 200, 0.06, 0.8),
]

print(f"\nGuzergah analiz ediliyor: {len(route)} nokta")

# --- Harita Olustur ---
harita = folium.Map(
    location=[41.5, 34.5],
    zoom_start=7,
    tiles="CartoDB dark_matter"
)

# Renk fonksiyonu
def drag_renk(drag):
    if drag < 0.15:   return "#00FF88"   # Yesil - verimli
    elif drag < 0.25: return "#FFFF00"   # Sari - normal
    elif drag < 0.40: return "#FF8800"   # Turuncu - dikkat
    else:             return "#FF0000"   # Kirmizi - kritik

results = []
coordinates = []

for name, lat, lon, depth, ssh, swh in route:
    drag = tahmin(lat, lon, depth, ssh, swh)
    savings = max(0, (0.5 - drag) * 30)
    color = drag_renk(drag)
    results.append((name, lat, lon, drag, savings, color))
    coordinates.append([lat, lon])

    # Nokta ekle
    popup_html = f"""
    <div style='font-family: Arial; width: 200px;'>
        <h4 style='color: {color}; margin: 0;'>{name}</h4>
        <hr style='margin: 5px 0;'>
        <b>Drag Skoru:</b> {drag:.4f}<br>
        <b>Yakit Tasarrufu:</b> %{savings:.1f}<br>
        <b>Derinlik:</b> {depth}m<br>
        <b>Dalga:</b> {swh}m<br>
        <b>Durum:</b> {'✅ Verimli' if drag < 0.25 else '⚠️ Dikkat'}
    </div>
    """
    folium.CircleMarker(
        location=[lat, lon],
        radius=12,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"{name} | Drag: {drag:.3f} | Tasarruf: %{savings:.1f}"
    ).add_to(harita)

# Guzergah cizgisi
folium.PolyLine(
    coordinates,
    color="#4488FF",
    weight=3,
    opacity=0.7,
    tooltip="Istanbul - Trabzon Guzergahi"
).add_to(harita)

# Baslangic ve bitis isaretleri
folium.Marker(
    coordinates[0],
    popup="KALKIS: Istanbul Bogazi",
    icon=folium.Icon(color="green", icon="ship", prefix="fa")
).add_to(harita)

folium.Marker(
    coordinates[-1],
    popup="VARIS: Trabzon Limani",
    icon=folium.Icon(color="red", icon="anchor", prefix="fa")
).add_to(harita)

# Bilgi kutusu
toplam_tasarruf = sum(s[4] for s in results) / len(results)
ortalama_drag   = sum(s[3] for s in results) / len(results)

bilgi_html = f"""
<div style='position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background: rgba(0,0,0,0.85); color: white; padding: 15px;
     border-radius: 10px; border: 2px solid #4488FF; font-family: Arial;'>
    <h3 style='margin: 0 0 10px 0; color: #4488FF;'>🚢 BATIMETRIX</h3>
    <b>Guzergah:</b> Istanbul → Trabzon<br>
    <b>Mesafe:</b> ~740 km<br>
    <b>Analiz Noktasi:</b> {len(route)}<br>
    <b>Ort. Drag:</b> {ortalama_drag:.4f}<br>
    <b>Ort. Tasarruf:</b> %{toplam_tasarruf:.1f}<br>
    <hr style='border-color: #4488FF;'>
    <span style='color:#00FF88'>●</span> Verimli (&lt;0.15)<br>
    <span style='color:#FFFF00'>●</span> Normal (0.15-0.25)<br>
    <span style='color:#FF8800'>●</span> Dikkat (0.25-0.40)<br>
    <span style='color:#FF0000'>●</span> Kritik (&gt;0.40)
</div>
"""
harita.get_root().html.add_child(folium.Element(bilgi_html))

# Kaydet
harita.save("batimetrix_map.html")
print(f"\nHarita saved: batimetrix_map.html")
print(f"Ortalama drag    : {ortalama_drag:.4f}")
print(f"Ortalama savings: %{toplam_tasarruf:.1f}")
print("\nOpen in browser: batimetrix_map.html")
