import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os

print("=== Batimetrix Guclu Model ONNX Export ===")

class GucluPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.giris = nn.Sequential(
            nn.Linear(7, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.katmanlar = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.05),
            ) for _ in range(6)
        ])
        self.cikis = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.giris(x)
        for katman in self.katmanlar:
            h = h + katman(h)
        return self.cikis(h)

model = GucluPINN()
model.load_state_dict(torch.load("batimetrix_guclu.pt", weights_only=True))
model.eval()
print(f"Model yuklendi: {sum(p.numel() for p in model.parameters()):,} parametre")

print("ONNX export yapiliyor...")
dummy = torch.zeros(1, 7)
torch.onnx.export(
    model, dummy, "batimetrix_guclu.onnx",
    export_params=True,
    opset_version=17,
    input_names=["features"],
    output_names=["drag_score"],
    dynamic_axes={
        "features":   {0: "batch_size"},
        "drag_score": {0: "batch_size"},
    }
)

boyut = os.path.getsize("batimetrix_guclu.onnx") / 1024
print(f"batimetrix_guclu.onnx olusturuldu! ({boyut:.1f} KB)")

print("\nONNX dogrulaniyor...")
onnx_model = onnx.load("batimetrix_guclu.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX gecerli!")

print("\nPyTorch vs ONNX karsilastirma...")
test = np.random.rand(5, 7).astype(np.float32)
with torch.no_grad():
    pt_out = model(torch.from_numpy(test)).numpy()
sess = ort.InferenceSession("batimetrix_guclu.onnx")
ort_out = sess.run(["drag_score"], {"features": test})[0]
fark = np.abs(pt_out - ort_out).max()
print(f"Maksimum sapma: {fark:.2e}")

if fark < 1e-3:
    print("Dogrulama BASARILI!")
else:
    print("UYARI: Sapma yuksek")

print("\n=== FINAL KARSILASTIRMA ===")
print(f"{'Model':<25} {'Parametre':>12} {'Boyut':>10}")
print("-" * 50)

eski = os.path.getsize("batimetrix.onnx") / 1024 if os.path.exists("batimetrix.onnx") else 0
print(f"{'Eski (batimetrix.onnx)':<25} {'151,681':>12} {eski:>9.1f}KB")
print(f"{'Yeni (guclu.onnx)':<25} {'1,657,025':>12} {boyut:>9.1f}KB")

print("\n=== KARADENIZ SON TEST (ONNX) ===")
def tahmin_onnx(lat, lon, derinlik, ssh, swh, hiz, tastak):
    inp = np.array([[
        (lat+70)/150, (lon+180)/360, derinlik/6000,
        (ssh+2)/4, swh/20, hiz/25, tastak/22
    ]], dtype=np.float32)
    return sess.run(["drag_score"], {"features": inp})[0][0][0]

senaryolar = [
    ("Karadeniz - Sakin",   42.1,  31.5,  920, 0.08, 1.2, 12.0, 8.5),
    ("Istanbul Bogazi",     41.1,  29.0,   35, 0.05, 0.3,  8.0, 7.0),
    ("Atlantik Firtina",    52.0, -20.0, 3200, 0.45, 8.5, 10.0, 10.0),
    ("Kutup Buz Denizi",    75.0,  15.0,  380, 0.35, 6.5,  6.0, 9.0),
]

print(f"\n{'Senaryo':<25} {'Drag':>8} {'Tasarruf':>10} {'Durum'}")
print("-" * 55)
for isim, la, lo, de, ss, sw, sp, dr in senaryolar:
    drag = tahmin_onnx(la, lo, de, ss, sw, sp, dr)
    tasarruf = max(0, (0.5 - drag) * 30)
    durum = "Verimli" if drag < 0.3 else "Dikkat"
    print(f"{isim:<25} {drag:>8.4f} %{tasarruf:>8.1f}  [{durum}]")

print("\nbatimetrix_guclu.onnx Rust tarafina hazir!")
print("ONNX export tamamlandi!")