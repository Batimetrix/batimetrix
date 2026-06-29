import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

print("=== Batimetrix ONNX Export ===")

# --- Model mimarisi (train_test.py ile AYNI) ---
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

# --- Modeli yukle ---
model = PINN()
model.load_state_dict(torch.load("batimetrix_model.pt", weights_only=True))
model.eval()
print("PyTorch modeli yuklendi!")

# --- ONNX export ---
print("ONNX formatina donusturuluyor...")
dummy = torch.zeros(1, 7)

torch.onnx.export(
    model,
    dummy,
    "batimetrix.onnx",
    export_params=True,
    opset_version=17,
    input_names=["features"],
    output_names=["drag_score"],
    dynamic_axes={
        "features":   {0: "batch_size"},
        "drag_score": {0: "batch_size"},
    }
)
print("batimetrix.onnx olusturuldu!")

# --- Boyut bilgisi ---
import os
boyut = os.path.getsize("batimetrix.onnx") / 1024
print(f"Model boyutu: {boyut:.1f} KB")

# --- Dogrulama ---
print("\nONNX dogrulaniyor...")
onnx_model = onnx.load("batimetrix.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX gecerli!")

# --- PyTorch vs ONNX karsilastirma ---
print("\nPyTorch vs ONNX karsilastirma:")
test = np.random.rand(5, 7).astype(np.float32)

with torch.no_grad():
    pt_out = model(torch.from_numpy(test)).numpy()

sess = ort.InferenceSession("batimetrix.onnx")
ort_out = sess.run(["drag_score"], {"features": test})[0]

fark = np.abs(pt_out - ort_out).max()
print(f"Maksimum sapma: {fark:.2e}")

if fark < 1e-4:
    print("Dogrulama BASARILI - Rust icin hazir!")
else:
    print("UYARI: Sapma yuksek")

# --- Son test ---
print("\n=== Karadeniz Son Testi (ONNX ile) ===")
senaryolar = [
    ("Istanbul Bogazi", 41.2, 29.1, 850),
    ("Orta Karadeniz",  42.2, 32.0, 1200),
    ("Trabzon aciklari",41.0, 39.5, 950),
]

for isim, lat, lon, derinlik in senaryolar:
    inp = np.array([[
        lat/90, (lon+180)/360, derinlik/4000,
        0.54, 0.10, 0.48, 0.34
    ]], dtype=np.float32)
    drag = sess.run(["drag_score"], {"features": inp})[0][0][0]
    durum = "Verimli" if drag < 0.3 else "Dikkat"
    print(f"{isim:<20} Drag: {drag:.4f} [{durum}]")

print("\nbatimetrix.onnx Rust tarafina kopyalanmaya hazir!")
print("ONNX export tamamlandi!")