import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Along-track current coefficient measured from real AIS vs NASA OSCAR
# (Florida Straits 0.59, Cape Hatteras 0.82; see README Early Validation)
K_CURRENT = 0.7

np.random.seed(2026)
torch.manual_seed(2026)
n = 100_000
lat = np.random.uniform(-70, 80, n)
lon = np.random.uniform(-180, 180, n)
depth = np.random.exponential(800, n).clip(5, 6000)
ssh = np.random.normal(0, 0.2, n).clip(-2, 2)
swh = np.random.exponential(2.0, n).clip(0, 20)
speed = np.random.uniform(4, 25, n)          # speed through water (kn)
draft = np.random.uniform(3, 22, n)
cur = np.random.normal(0, 1.0, n).clip(-4, 4)  # along-track current (kn)

speed_ms = speed * 0.5144
Re = (speed_ms * draft / 1.35e-6).clip(1e6, 1e10)
Cf = 0.075 / (np.log10(Re) - 2) ** 2
wave_f = 1.0 + 0.20 * (swh / 3.0) ** 1.8
depth_f = 1.0 + 0.5 * np.exp(-depth / 30.0)
drag_raw = Cf * speed_ms ** 2 * draft * wave_f * depth_f
sog = np.clip(speed + K_CURRENT * cur, 2.0, None)
eff = drag_raw * speed / sog                  # drag per nm over ground
y = (eff / np.percentile(eff, 99)).clip(0, 1)
y = np.clip(y + np.random.normal(0, 0.01, n), 0, 1).astype(np.float32)

def norm(la, lo, de, ss, sw, sp, dr, cu):
    return np.column_stack([(la + 70) / 150, (lo + 180) / 360, de / 6000,
                            (ss + 2) / 4, sw / 20, sp / 25, dr / 22,
                            (cu + 4) / 8]).astype(np.float32)

X_t = torch.from_numpy(norm(lat, lon, depth, ssh, swh, speed, draft, cur))
y_t = torch.from_numpy(y)

class PINNv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(8, 512), nn.LayerNorm(512), nn.GELU())
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512),
                                     nn.GELU(), nn.Dropout(0.05)) for _ in range(6)])
        self.output_layer = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 32),
                                          nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        h = self.input_layer(x)
        for k in self.layers:
            h = h + k(h)
        return self.output_layer(h)

model = PINNv3()
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
EPOCHS = 60
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
mse = nn.MSELoss()
batch, best, wait, t0 = 2048, float('inf'), 0, time.time()
step = torch.zeros(8)
step[7] = 0.5 / 8   # +0.5 kn of following current

for epoch in range(1, EPOCHS + 1):
    model.train()
    idx = torch.randperm(len(X_t))
    total = 0.0
    for i in range(len(X_t) // batch):
        b = idx[i * batch:(i + 1) * batch]
        xb, yb = X_t[b], y_t[b]
        opt.zero_grad()
        pred = model(xb).squeeze()
        # Physics: more following current must not increase drag per nm
        phys = torch.relu(model(xb + step).squeeze() - pred).mean()
        phys = phys + torch.relu(xb[:, 4] * 0.3 - pred).mean()
        loss = mse(pred, yb) + 0.05 * phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    sched.step()
    avg = total / (len(X_t) // batch)
    if avg < best:
        best, wait = avg, 0
        torch.save(model.state_dict(), 'batimetrix_v3_current.pt')
    else:
        wait += 1
    if epoch % 5 == 0:
        print(f'Epoch {epoch:3d}/{EPOCHS} | loss {avg:.6f} | best {best:.6f} | {time.time() - t0:.0f}s', flush=True)
    if wait >= 10:
        print(f'Early stop at epoch {epoch}')
        break

model.load_state_dict(torch.load('batimetrix_v3_current.pt', weights_only=True))
model.eval()
print('\nSame ship, 12 kn, Florida Straits, calm sea:')
for c in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    x = torch.from_numpy(norm(np.array([25.5]), np.array([-80.0]), np.array([600.0]),
                              np.array([0.0]), np.array([1.0]), np.array([12.0]),
                              np.array([10.0]), np.array([c])))
    with torch.no_grad():
        print(f'  current {c:+.1f} kn -> drag per nm {model(x).item():.4f}')
print(f'\nSaved: batimetrix_v3_current.pt | total {time.time() - t0:.0f}s')
