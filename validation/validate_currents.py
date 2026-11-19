import io, os, sys, requests, zstandard as zstd, numpy as np, pandas as pd, xarray as xr, earthaccess

days = sys.argv[1].split(',')
PORTS = [(25.77, -80.17), (26.09, -80.12), (26.77, -80.05), (24.56, -81.80)]
cols = ['mmsi', 'latitude', 'longitude', 'sog', 'cog', 'vessel_type', 'length', 'status']
earthaccess.login(strategy="environment")
os.makedirs('results', exist_ok=True)
rows, pooled = [], []

def red(a, b):
    return round(100 * (1 - b.var() / a.var()), 1)

for day in days:
    day = day.strip()
    url = f'https://noaaocm.blob.core.windows.net/ais/csv2/csv2025/ais-{day}.csv.zst'
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(r.raw), encoding='utf-8', errors='replace')
        parts = []
        for ch in pd.read_csv(text, chunksize=500_000, low_memory=False, usecols=cols):
            for c in cols[1:]:
                ch[c] = pd.to_numeric(ch[c], errors='coerce')
            parts.append(ch[ch['latitude'].between(23.5, 27.5) & ch['longitude'].between(-81.5, -79.0)])
    df = pd.concat(parts, ignore_index=True)
    big = df[df['vessel_type'].between(70, 89) & (df['length'] >= 100) & df['sog'].between(8, 25)
             & df['cog'].between(0, 360) & ((df['status'] == 0) | df['status'].isna())].copy()
    far = np.ones(len(big), dtype=bool)
    for plat, plon in PORTS:
        dkm = np.hypot((big['latitude'] - plat) * 111, (big['longitude'] - plon) * 111 * np.cos(np.deg2rad(plat)))
        far &= (dkm > 15).values
    big = big[far]

    res = earthaccess.search_data(short_name='OSCAR_L4_OC_NRT_V2.0', temporal=(day, day))
    f = earthaccess.download(res[:1], local_path='oscar')[0]
    od = xr.open_dataset(f, engine='h5netcdf', decode_times=False)
    olat, olon = od['lat'].values, od['lon'].values
    U, V = od['u'].values[0], od['v'].values[0]
    ii = np.clip(np.rint(((big['longitude'].values % 360) - olon[0]) / (olon[1] - olon[0])).astype(int), 0, len(olon) - 1)
    jj = np.clip(np.rint((big['latitude'].values - olat[0]) / (olat[1] - olat[0])).astype(int), 0, len(olat) - 1)
    cog = np.deg2rad(big['cog'].values)
    big['cur_kn'] = (U[ii, jj] * np.sin(cog) + V[ii, jj] * np.cos(cog)) * 1.94384
    big = big.dropna(subset=['cur_kn'])
    big['stw'] = big['sog'] - big['cur_kn']
    od.close()
    os.remove(f)

    nb = big[(big['cog'] >= 315) | (big['cog'] <= 45)]
    sb = big[big['cog'].between(135, 225)]
    w = big[big.groupby('mmsi')['sog'].transform('size') >= 20].copy()
    w['sog_anom'] = w['sog'] - w.groupby('mmsi')['sog'].transform('mean')
    w['cur_anom'] = w['cur_kn'] - w.groupby('mmsi')['cur_kn'].transform('mean')
    w['stw_anom'] = w['sog_anom'] - w['cur_anom']
    w['day'] = day
    slope = np.polyfit(w['cur_anom'], w['sog_anom'], 1)[0]
    rr = np.corrcoef(w['cur_anom'], w['sog_anom'])[0, 1]
    g = w.groupby('mmsi')
    per_vessel = (1 - g['stw_anom'].var() / g['sog_anom'].var()) * 100
    rows.append({'day': day, 'vessels': w['mmsi'].nunique(), 'positions': len(w),
                 'slope': round(slope, 2), 'r': round(rr, 2),
                 'gap_sog': round(nb['sog'].median() - sb['sog'].median(), 2),
                 'gap_stw': round(nb['stw'].median() - sb['stw'].median(), 2),
                 'sd_sog': round(w['sog_anom'].std(), 2), 'sd_stw': round(w['stw_anom'].std(), 2),
                 'var_red_full': red(w['sog_anom'], w['stw_anom']),
                 'vessel_median_red': round(per_vessel.median(), 1)})
    pooled.append(w[['day', 'sog_anom', 'cur_anom', 'stw_anom']])
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
P = pd.concat(pooled, ignore_index=True)
# Leave-one-day-out: scale factor k fitted on the other days, tested on the held-out day
cv = []
for d in out['day']:
    tr, te = P[P['day'] != d], P[P['day'] == d]
    k = np.polyfit(tr['cur_anom'], tr['sog_anom'], 1)[0]
    cv.append(red(te['sog_anom'], te['sog_anom'] - k * te['cur_anom']))
out['var_red_cv'] = cv
out.to_csv('results/validation_days.csv', index=False)
s_all = np.polyfit(P['cur_anom'], P['sog_anom'], 1)[0]
r_all = np.corrcoef(P['cur_anom'], P['sog_anom'])[0, 1]
pd.set_option('display.width', 250)
print('\n' + out.to_string(index=False))
print(f'\nALL DAYS: positions {len(P):,} | slope {s_all:.2f} | r {r_all:.2f} | r^2 {r_all ** 2:.2f}')
print(f'North-south gap: SOG {out["gap_sog"].mean():.2f} kn -> STW {out["gap_stw"].mean():.2f} kn (mean of days)')
print(f'Wi




cd "C:\Users\ACER\OneDrive\Masaüstü\Batimetrix\shore"
.\venv\Scripts\Activate.ps1
git pull origin main
@'
import io, os, sys, requests, zstandard as zstd, numpy as np, pandas as pd, xarray as xr, earthaccess

days = sys.argv[1].split(',')
PORTS = [(25.77, -80.17), (26.09, -80.12), (26.77, -80.05), (24.56, -81.80)]
cols = ['mmsi', 'latitude', 'longitude', 'sog', 'cog', 'vessel_type', 'length', 'status']
earthaccess.login(strategy="environment")
os.makedirs('results', exist_ok=True)
rows, pooled = [], []

def red(a, b):
    return round(100 * (1 - b.var() / a.var()), 1)

for day in days:
    day = day.strip()
    url = f'https://noaaocm.blob.core.windows.net/ais/csv2/csv2025/ais-{day}.csv.zst'
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(r.raw), encoding='utf-8', errors='replace')
        parts = []
        for ch in pd.read_csv(text, chunksize=500_000, low_memory=False, usecols=cols):
            for c in cols[1:]:
                ch[c] = pd.to_numeric(ch[c], errors='coerce')
            parts.append(ch[ch['latitude'].between(23.5, 27.5) & ch['longitude'].between(-81.5, -79.0)])
    df = pd.concat(parts, ignore_index=True)
    big = df[df['vessel_type'].between(70, 89) & (df['length'] >= 100) & df['sog'].between(8, 25)
             & df['cog'].between(0, 360) & ((df['status'] == 0) | df['status'].isna())].copy()
    far = np.ones(len(big), dtype=bool)
    for plat, plon in PORTS:
        dkm = np.hypot((big['latitude'] - plat) * 111, (big['longitude'] - plon) * 111 * np.cos(np.deg2rad(plat)))
        far &= (dkm > 15).values
    big = big[far]

    res = earthaccess.search_data(short_name='OSCAR_L4_OC_NRT_V2.0', temporal=(day, day))
    f = earthaccess.download(res[:1], local_path='oscar')[0]
    od = xr.open_dataset(f, engine='h5netcdf', decode_times=False)
    olat, olon = od['lat'].values, od['lon'].values
    U, V = od['u'].values[0], od['v'].values[0]
    ii = np.clip(np.rint(((big['longitude'].values % 360) - olon[0]) / (olon[1] - olon[0])).astype(int), 0, len(olon) - 1)
    jj = np.clip(np.rint((big['latitude'].values - olat[0]) / (olat[1] - olat[0])).astype(int), 0, len(olat) - 1)
    cog = np.deg2rad(big['cog'].values)
    big['cur_kn'] = (U[ii, jj] * np.sin(cog) + V[ii, jj] * np.cos(cog)) * 1.94384
    big = big.dropna(subset=['cur_kn'])
    big['stw'] = big['sog'] - big['cur_kn']
    od.close()
    os.remove(f)

    nb = big[(big['cog'] >= 315) | (big['cog'] <= 45)]
    sb = big[big['cog'].between(135, 225)]
    w = big[big.groupby('mmsi')['sog'].transform('size') >= 20].copy()
    w['sog_anom'] = w['sog'] - w.groupby('mmsi')['sog'].transform('mean')
    w['cur_anom'] = w['cur_kn'] - w.groupby('mmsi')['cur_kn'].transform('mean')
    w['stw_anom'] = w['sog_anom'] - w['cur_anom']
    w['day'] = day
    slope = np.polyfit(w['cur_anom'], w['sog_anom'], 1)[0]
    rr = np.corrcoef(w['cur_anom'], w['sog_anom'])[0, 1]
    g = w.groupby('mmsi')
    per_vessel = (1 - g['stw_anom'].var() / g['sog_anom'].var()) * 100
    rows.append({'day': day, 'vessels': w['mmsi'].nunique(), 'positions': len(w),
                 'slope': round(slope, 2), 'r': round(rr, 2),
                 'gap_sog': round(nb['sog'].median() - sb['sog'].median(), 2),
                 'gap_stw': round(nb['stw'].median() - sb['stw'].median(), 2),
                 'sd_sog': round(w['sog_anom'].std(), 2), 'sd_stw': round(w['stw_anom'].std(), 2),
                 'var_red_full': red(w['sog_anom'], w['stw_anom']),
                 'vessel_median_red': round(per_vessel.median(), 1)})
    pooled.append(w[['day', 'sog_anom', 'cur_anom', 'stw_anom']])
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
P = pd.concat(pooled, ignore_index=True)
# Leave-one-day-out: scale factor k fitted on the other days, tested on the held-out day
cv = []
for d in out['day']:
    tr, te = P[P['day'] != d], P[P['day'] == d]
    k = np.polyfit(tr['cur_anom'], tr['sog_anom'], 1)[0]
    cv.append(red(te['sog_anom'], te['sog_anom'] - k * te['cur_anom']))
out['var_red_cv'] = cv
out.to_csv('results/validation_days.csv', index=False)
s_all = np.polyfit(P['cur_anom'], P['sog_anom'], 1)[0]
r_all = np.corrcoef(P['cur_anom'], P['sog_anom'])[0, 1]
pd.set_option('display.width', 250)
print('\n' + out.to_string(index=False))
print(f'\nALL DAYS: positions {len(P):,} | slope {s_all:.2f} | r {r_all:.2f} | r^2 {r_all ** 2:.2f}')
print(f'North-south gap: SOG {out["gap_sog"].mean():.2f} kn -> STW {out["gap_stw"].mean():.2f} kn (mean of days)')
print(f'Within-vessel variance reduction: full current {red(P["sog_anom"], P["stw_anom"])}% | '
      f'cross-validated scaled current {np.mean(cv):.1f}% (range {min(cv)} to {max(cv)})')
