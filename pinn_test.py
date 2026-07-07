import torch
import torch.nn as nn

print("=== Batimetrix PINN Testi ===")
print(f"PyTorch versiyon: {torch.__version__}")

class KucukPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = KucukPINN()
test_input = torch.tensor([[0.47, 0.58, 0.21, 0.56, 0.12, 0.48, 0.34]])
drag_skoru = model(test_input).item()

print(f"Koordinat: 42.1 Kuzey, 31.5 Dogu (Karadeniz)")
print(f"Suruklenme direnci skoru: {drag_skoru:.4f}")
if drag_skoru < 0.5:
    print("Durum: Dusuk direnc - gemi verimli ilerleyebilir")
else:
    print("Durum: Yuksek direnc - pervane optimizasyonu onerilir")
print("\nBatimetrix ilk test completed!")