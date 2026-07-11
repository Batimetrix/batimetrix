import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("=== Batimetrix Grafik Gorselleştirme ===")

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

def tahmin(lat, lon, depth, ssh, swh, speed, draft):
    inp = torch.tensor([[
        (lat+70)/150, (lon+180)/360, depth/6000,
        (ssh+2)/4, swh/20, speed/25, draft/22
    ]]).float()
    with torch.no_grad():
        return model(inp).item()

# Koyu tema
plt.style.use('dark_background')
TEAL  = '#1ABC9C'
NAVY  = '#0A1628'
GOLD  = '#F39C12'
RED   = '#E74C3C'
BLUE  = '#3498DB'
GREEN = '#27AE60'
GRAY  = '#7F8C8D'

fig = plt.figure(figsize=(18, 14), facecolor='#0A1628')
fig.suptitle('BATIMETRIX — Hidrodinamik Sürüklenme Analizi',
             fontsize=20, color=TEAL, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ============================================================
# GRAFIK 1: Istanbul-Trabzon Güzergah Drag
# ============================================================
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#0D1F35')

route = [
    ("Istanbul\nBogazi",   41.10, 29.05,  35, 0.05, 0.4, 12, 8.5),
    ("Kara.Giris",         41.30, 29.50, 120, 0.07, 0.8, 12, 8.5),
    ("Bati KB",            41.80, 30.50, 650, 0.08, 1.2, 12, 8.5),
    ("Zonguldak",          41.60, 31.80, 850, 0.08, 1.5, 12, 8.5),
    ("Orta KB",            42.10, 32.50,1100, 0.09, 1.8, 12, 8.5),
    ("Sinop",              42.00, 35.10, 950, 0.10, 1.6, 12, 8.5),
    ("Samsun",             41.70, 36.20, 800, 0.11, 1.9, 12, 8.5),
    ("Ordu",               41.10, 37.80, 750, 0.09, 1.7, 12, 8.5),
    ("Giresun",            41.00, 38.50, 700, 0.08, 1.5, 12, 8.5),
    ("Trabzon\nAciklari",  41.00, 39.50, 650, 0.10, 1.8, 12, 8.5),
    ("Trabzon\nLimani",    41.00, 39.73, 200, 0.06, 0.8, 12, 8.5),
]

isimler = [g[0] for g in route]
draglar = [tahmin(*g[1:]) for g in route]
tasarruflar = [max(0, (0.5 - d) * 30) for d in draglar]
renkler = [GREEN if d < 0.15 else GOLD if d < 0.25 else RED for d in draglar]

x = range(len(isimler))
bars = ax1.bar(x, draglar, color=renkler, alpha=0.85, width=0.6, zorder=3)
ax1.plot(x, draglar, color=TEAL, linewidth=2, marker='o',
         markersize=6, zorder=4, label='Drag Skoru')
ax1.axhline(y=0.15, color=GREEN, linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=0.25, color=GOLD,  linestyle='--', alpha=0.5, linewidth=1)
ax1.fill_between(x, draglar, alpha=0.1, color=TEAL)

ax1.set_title('İstanbul → Trabzon Güzergahı — Drag Skoru',
              color='white', fontsize=12, pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(isimler, fontsize=7.5, color='white', rotation=0)
ax1.set_ylabel('Drag Skoru', color=TEAL, fontsize=10)
ax1.set_ylim(0, 0.35)
ax1.tick_params(colors='white')
ax1.grid(axis='y', alpha=0.2, color=GRAY)
ax1.spines['bottom'].set_color(TEAL)
ax1.spines['left'].set_color(TEAL)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for i, (d, t) in enumerate(zip(draglar, tasarruflar)):
    ax1.text(i, d + 0.005, f'%{t:.1f}', ha='center',
             va='bottom', fontsize=7, color=GOLD, fontweight='bold')

# ============================================================
# GRAFIK 2: Gemi Tipi Karsilastirma
# ============================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#0D1F35')

vessel_types = {
    "VLCC\nTanker":    (22.0, 15.0),
    "LNG\nCarrier":    (12.0, 19.5),
    "Panamax\nKont.":  (13.5, 20.0),
    "Capesize\nBulk":  (18.0, 14.5),
    "Handy\nBulk":     (10.0, 14.0),
    "RoRo\nGemisi":    ( 8.0, 22.0),
}

gemi_isimler = list(vessel_types.keys())
gemi_draglar = []
for name, (draft, speed) in vessel_types.items():
    d = tahmin(42.1, 31.5, 920, 0.08, 1.5, speed, draft)
    gemi_draglar.append(d)

gemi_renkler = [BLUE, TEAL, GOLD, RED, GREEN, '#9B59B6']
wedges, texts, autotexts = ax2.pie(
    gemi_draglar,
    labels=gemi_isimler,
    colors=gemi_renkler,
    autopct='%1.1f%%',
    pctdistance=0.75,
    startangle=90,
    textprops={'fontsize': 7.5, 'color': 'white'},
    wedgeprops={'linewidth': 2, 'edgecolor': '#0A1628'}
)
for at in autotexts:
    at.set_fontsize(7)
    at.set_color('white')
    at.set_fontweight('bold')

ax2.set_title('Gemi Tipi\nDrag Dağılımı\n(Karadeniz - Sakin)',
              color='white', fontsize=10, pad=10)

# ============================================================
# GRAFIK 3: Dalga Yüksekliği vs Drag
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#0D1F35')

swh_dizi = np.linspace(0, 12, 50)
drag_sakin  = [tahmin(42.1, 31.5,  920, 0.05, s, 12, 8.5) for s in swh_dizi]
drag_firtina = [tahmin(42.1, 31.5, 1100, 0.20, s, 10, 8.5) for s in swh_dizi]
drag_sig     = [tahmin(41.1, 29.0,   35, 0.05, s, 8, 8.5)  for s in swh_dizi]

ax3.plot(swh_dizi, drag_sakin,   color=TEAL,  linewidth=2.5, label='Derin Su (920m)')
ax3.plot(swh_dizi, drag_firtina, color=GOLD,  linewidth=2.5, label='Orta Derin (1100m)')
ax3.plot(swh_dizi, drag_sig,     color=RED,   linewidth=2.5, label='Sığ Su (35m)')
ax3.axhline(y=0.5, color=GRAY, linestyle=':', alpha=0.5)
ax3.fill_between(swh_dizi, drag_sakin, alpha=0.1, color=TEAL)

ax3.set_title('Dalga Yüksekliği vs Drag', color='white', fontsize=10)
ax3.set_xlabel('SWH (m)', color=TEAL, fontsize=9)
ax3.set_ylabel('Drag Skoru', color=TEAL, fontsize=9)
ax3.legend(fontsize=7, loc='upper left')
ax3.tick_params(colors='white')
ax3.grid(alpha=0.2, color=GRAY)
ax3.spines['bottom'].set_color(TEAL)
ax3.spines['left'].set_color(TEAL)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ============================================================
# GRAFIK 4: Derinlik vs Drag
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#0D1F35')

derinlik_dizi = np.linspace(5, 3000, 100)
drag_yavash  = [tahmin(42.1, 31.5, d, 0.08, 1.5, 10, 8.5) for d in derinlik_dizi]
drag_normal  = [tahmin(42.1, 31.5, d, 0.08, 1.5, 14, 8.5) for d in derinlik_dizi]
drag_hizli   = [tahmin(42.1, 31.5, d, 0.08, 1.5, 20, 8.5) for d in derinlik_dizi]

ax4.plot(derinlik_dizi, drag_yavash, color=GREEN, linewidth=2.5, label='10 knot')
ax4.plot(derinlik_dizi, drag_normal, color=GOLD,  linewidth=2.5, label='14 knot')
ax4.plot(derinlik_dizi, drag_hizli,  color=RED,   linewidth=2.5, label='20 knot')
ax4.axvline(x=50,  color=GRAY, linestyle=':', alpha=0.5, label='Sığ Su Sınırı')
ax4.fill_betweenx([0, 1], 0, 50, alpha=0.05, color=RED)

ax4.set_title('Derinlik vs Drag\n(Hız Etkisi)', color='white', fontsize=10)
ax4.set_xlabel('Su Derinliği (m)', color=TEAL, fontsize=9)
ax4.set_ylabel('Drag Skoru', color=TEAL, fontsize=9)
ax4.legend(fontsize=7)
ax4.set_xlim(0, 3000)
ax4.tick_params(colors='white')
ax4.grid(alpha=0.2, color=GRAY)
ax4.spines['bottom'].set_color(TEAL)
ax4.spines['left'].set_color(TEAL)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ============================================================
# GRAFIK 5: Hız vs Drag (Gemi Tipleri)
# ============================================================
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#0D1F35')

hiz_dizi = np.linspace(6, 24, 50)
profiller = [
    ("VLCC (Tastak 22m)",  22.0, RED),
    ("Panamax (13.5m)",    13.5, GOLD),
    ("Handy (10m)",        10.0, GREEN),
    ("RoRo (8m)",           8.0, BLUE),
]
for name, draft, color in profiller:
    draglar_hiz = [tahmin(42.1, 31.5, 920, 0.08, 1.5, h, draft) for h in hiz_dizi]
    ax5.plot(hiz_dizi, draglar_hiz, color=color, linewidth=2.5, label=name)

ax5.set_title('Hız vs Drag\n(Gemi Tipi Etkisi)', color='white', fontsize=10)
ax5.set_xlabel('Hız (knot)', color=TEAL, fontsize=9)
ax5.set_ylabel('Drag Skoru', color=TEAL, fontsize=9)
ax5.legend(fontsize=7)
ax5.tick_params(colors='white')
ax5.grid(alpha=0.2, color=GRAY)
ax5.spines['bottom'].set_color(TEAL)
ax5.spines['left'].set_color(TEAL)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ============================================================
# GRAFIK 6: Yıllık Tasarruf Potansiyeli
# ============================================================
ax6 = fig.add_subplot(gs[2, :2])
ax6.set_facecolor('#0D1F35')

gemi_isim2 = ["VLCC\nTanker", "Capesize\nBulk", "LNG\nCarrier",
               "Panamax\nKont.", "RoRo\nGemisi", "Handy\nBulk", "Ozgun\nGemi"]
dwt_ler    = [300000, 180000, 80000, 65000, 25000, 35000, 8000]
hiz_ler    = [15.0, 14.5, 19.5, 20.0, 22.0, 14.0, 12.0]
tastak_ler = [22.0, 18.0, 12.0, 13.5, 8.0, 10.0, 8.5]
renkler2   = [RED, '#8E44AD', BLUE, GOLD, '#E67E22', GREEN, TEAL]

tasarruf_ler = []
for dwt, speed, draft in zip(dwt_ler, hiz_ler, tastak_ler):
    drag = tahmin(42.1, 31.5, 920, 0.08, 1.5, speed, draft)
    gunluk = dwt * 0.000012 * (speed ** 2.5)
    yillik = gunluk * 280 * 650
    ratio   = max(0, (0.5 - drag) * 0.30)
    tasarruf_ler.append(yillik * ratio / 1_000_000)

x2 = range(len(gemi_isim2))
bars2 = ax6.bar(x2, tasarruf_ler, color=renkler2, alpha=0.85,
                width=0.6, zorder=3)
ax6.plot(x2, tasarruf_ler, color='white', linewidth=1.5,
         marker='D', markersize=6, zorder=4)

for i, val in enumerate(tasarruf_ler):
    ax6.text(i, val + 0.3, f'${val:.1f}M', ha='center',
             va='bottom', fontsize=9, color='white', fontweight='bold')

ax6.set_title('Gemi Tipi Başına Yıllık Tasarruf Potansiyeli (Batimetrix %20 Fiyatlandırma)',
              color='white', fontsize=11, pad=10)
ax6.set_xticks(x2)
ax6.set_xticklabels(gemi_isim2, fontsize=9, color='white')
ax6.set_ylabel('Yıllık Tasarruf (Milyon $)', color=TEAL, fontsize=10)
ax6.tick_params(colors='white')
ax6.grid(axis='y', alpha=0.2, color=GRAY)
ax6.spines['bottom'].set_color(TEAL)
ax6.spines['left'].set_color(TEAL)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

# ============================================================
# GRAFIK 7: Özet Kart
# ============================================================
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor('#0D1F35')
ax7.axis('off')

ozet_metin = [
    ("MODEL", "1.657.025 Parametre", TEAL),
    ("VERİ", "296.526 NASA SWOT", BLUE),
    ("GÜZERGAH", "İstanbul → Trabzon", GREEN),
    ("ORT. DRAG", "0.0791", GOLD),
    ("ORT. TASARRUF", "%11.0", GREEN),
    ("EN İYİ GEMİ", "VLCC Tanker", RED),
    ("YILLIK KAZAN", "$15.7M / gemi", GOLD),
    ("BATIMETRIX FİYAT", "%20 savings", TEAL),
    ("","",""),
    ("github.com/", "Batimetrix/batimetrix", GRAY),
]

y_pos = 0.95
for baslik, deger, color in ozet_metin:
    if not baslik:
        y_pos -= 0.04
        continue
    ax7.text(0.05, y_pos, baslik + ":", fontsize=8,
             color=GRAY, transform=ax7.transAxes, fontweight='bold')
    ax7.text(0.05, y_pos - 0.06, deger, fontsize=9,
             color=color, transform=ax7.transAxes, fontweight='bold')
    y_pos -= 0.13

ax7.set_title('Özet', color='white', fontsize=11, pad=10)
ax7.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
              edgecolor=TEAL, linewidth=2, transform=ax7.transAxes))

plt.savefig("batimetrix_grafik.png", dpi=150, bbox_inches='tight',
            facecolor='#0A1628', edgecolor='none')
print("\nGrafik saved: batimetrix_grafik.png")
print("Gorsellestirme completed!")
