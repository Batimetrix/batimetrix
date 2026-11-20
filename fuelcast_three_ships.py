import numpy as np, pandas as pd

def design(v, hs):
    return np.column_stack([np.ones_like(v), v, v ** 2, v ** 3, hs])

def cv_err(v, hs, y, folds):
    idx, sq = np.arange(len(y)), 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        b = np.linalg.lstsq(design(v[tr], hs[tr]), y[tr], rcond=None)[0]
        sq += np.sum((design(v[f], hs[f]) @ b - y[f]) ** 2)
    return 100 * np.sqrt(sq / len(y)) / y.mean()

for name in ['CPS_Poseidon', 'CPS_Triton', 'OSS_Ceto']:
    d = pd.read_parquet(f'fuelcast/{name}.parquet')
    head = 'Ship_Heading' if 'Ship_Heading' in d.columns else 'Ship_Bearing'
    need = ['Ship_SpeedOverGround', head, 'Weather_OceanCurrentVelocity',
            'Weather_OceanCurrentDirection', 'Weather_WaveHeight', 'Consumer_Total_MomentaryFuel']
    miss = [c for c in need if c not in d.columns]
    if miss:
        print(f'\n{name}: missing {miss}')
        continue
    u = d.dropna(subset=need)
    u = u[(u['Ship_SpeedOverGround'] > 3) & (u['Consumer_Total_MomentaryFuel'] > 0)].reset_index(drop=True)
    sog, hs, y = u['Ship_SpeedOverGround'].values, u['Weather_WaveHeight'].values, u['Consumer_Total_MomentaryFuel'].values
    al = (u['Weather_OceanCurrentVelocity'] * np.cos(np.deg2rad(u['Weather_OceanCurrentDirection'] - u[head]))).values
    folds = np.array_split(np.arange(len(u)), 5)
    print(f'\n{name} | rows {len(u):,} | SOG median {np.median(sog):.2f} | |current| median {np.median(np.abs(al)):.2f} | heading col {head}')
    base = cv_err(sog, hs, y, folds)
    for k in [0.0, 0.7, 1.0]:
        e = cv_err(sog - k * al, hs, y, folds)
        print(f'  k = {k:.1f}: fuel error {e:.2f}%  (change vs k=0: {100 * (e - base) / base:+.1f}%)')
