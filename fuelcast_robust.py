import numpy as np, pandas as pd
exec(open('fuelcast_test.py', encoding='utf-8-sig').read().split("print('\\n--- Test 2")[0])
y, hs = u['Consumer_Total_MomentaryFuel'].values, u['Weather_WaveHeight'].values
sog, al = u['Ship_SpeedOverGround'].values, u['along'].values
def cv(v):
    sq = 0.0
    for f in folds:
        tr = np.setdiff1d(np.arange(len(u)), f)
        sq += np.sum((fit_pred(v[tr], hs[tr], y[tr], v[f], hs[f]) - y[f]) ** 2)
    return 100 * np.sqrt(sq / len(y)) / y.mean()
print('\n--- Robustness: fuel error with fixed k (no fitting) ---')
for k in [0.0, 0.7, 1.0]:
    print(f'  k = {k:.1f}: {cv(sog - k * al):.2f}%')
p = u['Propeller_Total_ShaftPower']
print('\n--- Propeller power diagnostics (underway rows) ---')
print(f'  zeros: {(p == 0).mean() * 100:.1f}% | median {p.median():,.0f} | max {p.max():,.0f}')
print(f'  corr with STW: {np.corrcoef(p, u["Ship_SpeedThroughWater"])[0, 1]:+.2f} | '
      f'corr with fuel: {np.corrcoef(p, y)[0, 1]:+.2f}')
