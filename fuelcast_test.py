import numpy as np, pandas as pd
d = pd.read_parquet('fuelcast/CPS_Poseidon.parquet')
need = ['Ship_SpeedOverGround', 'Ship_SpeedThroughWater', 'Ship_Heading', 'Ship_Bearing',
        'Weather_OceanCurrentVelocity', 'Weather_OceanCurrentDirection', 'Weather_WaveHeight']
targets = [c for c in ['Propeller_Total_ShaftPower', 'Consumer_Total_MomentaryFuel'] if c in d.columns]
print('targets:', targets)
u = d.dropna(subset=need + targets).copy()
u = u[(u['Ship_SpeedThroughWater'] > 3) & (u['Ship_SpeedOverGround'] > 3)]
u['diff'] = u['Ship_SpeedOverGround'] - u['Ship_SpeedThroughWater']
u = u[u['diff'].abs() < 2.5].reset_index(drop=True)
print(f"Underway rows: {len(u):,} | SOG-STW mean {u['diff'].mean():+.3f} "
      f"median {u['diff'].median():+.3f} sd {u['diff'].std():.3f}")

print('\n--- Test 1: ship-measured current (SOG-STW) vs Open-Meteo along-track current ---')
for ref in ['Ship_Heading', 'Ship_Bearing']:
    al = u['Weather_OceanCurrentVelocity'] * np.cos(np.deg2rad(u['Weather_OceanCurrentDirection'] - u[ref]))
    r = np.corrcoef(al, u['diff'])[0, 1]
    s = np.polyfit(al, u['diff'], 1)[0]
    print(f'{ref:<13} r {r:+.3f} | slope {s:+.3f}')

u['along'] = u['Weather_OceanCurrentVelocity'] * np.cos(
    np.deg2rad(u['Weather_OceanCurrentDirection'] - u['Ship_Heading']))
folds = np.array_split(np.arange(len(u)), 5)

def design(v, hs):
    return np.column_stack([np.ones_like(v), v, v ** 2, v ** 3, hs])

def fit_pred(v_tr, hs_tr, y_tr, v_te, hs_te):
    b = np.linalg.lstsq(design(v_tr, hs_tr), y_tr, rcond=None)[0]
    return design(v_te, hs_te) @ b

def evaluate(tgt, mode):
    y, hs = u[tgt].values, u['Weather_WaveHeight'].values
    sog, stw, al = u['Ship_SpeedOverGround'].values, u['Ship_SpeedThroughWater'].values, u['along'].values
    idx, sq, ks = np.arange(len(u)), 0.0, []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if mode == 'A':
            v = sog
        elif mode == 'C':
            v = stw
        else:
            best = None
            for k in np.arange(-1.5, 1.51, 0.05):
                vv = sog - k * al
                e = np.mean((fit_pred(vv[tr], hs[tr], y[tr], vv[tr], hs[tr]) - y[tr]) ** 2)
                if best is None or e < best[0]:
                    best = (e, k)
            ks.append(round(best[1], 2))
            v = sog - best[1] * al
        p = fit_pred(v[tr], hs[tr], y[tr], v[f], hs[f])
        sq += np.sum((p - y[f]) ** 2)
    return 100 * np.sqrt(sq / len(y)) / y.mean(), ks

print('\n--- Test 2: cross-validated error (% of mean), 5 time blocks ---')
for tgt in targets:
    a, _ = evaluate(tgt, 'A')
    b, ks = evaluate(tgt, 'B')
    c, _ = evaluate(tgt, 'C')
    share = 100 * (a - b) / (a - c) if a != c else float('nan')
    print(f'{tgt}')
    print(f'  A  SOG only            {a:.2f}%')
    print(f'  B  SOG - k*current     {b:.2f}%   k per block {ks}')
    print(f'  C  measured STW        {c:.2f}%')
    print(f'  Share of possible improvement achieved by B: {share:.0f}%')
