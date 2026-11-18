import io, os, sys, requests, zstandard as zstd, numpy as np, pandas as pd, xarray as xr, earthaccess

days = sys.argv[1].split(',')
PORTS = [(25.77, -80.17), (26.09, -80.12), (26.77, -80.05), (24.56, -81.80)]
cols = ['mmsi', 'latitude', 'longitude', 'sog', 'cog', 'vessel_type', 'length', 'status']
earthaccess.login(strategy="environment")
os.makedirs('results', exist_ok=True)
rows, pooled = [], []

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
    od.close()
    os.remove(f)

    nb = big[(big['cog'] >= 315) | (big['cog'] <= 45)]
    sb = big[big['cog'].between(135, 225)]
    w = big[big.groupby('mmsi')['sog'].transform('size') >= 20].copy()
    w['sog_anom'] = w['sog'] - w.groupby('mmsi')['sog'].transform('mean')
    w['cur_anom'] = w['cur_kn'] - w.groupby('mmsi')['cur_kn'].transform('mean')
    slope = np.polyfit(w['cur_anom'], w['sog_anom'], 1)[0]
    rr = np.corrcoef(w['cur_anom'], w['sog_anom'])[0, 1]
    rows.append({'day': day, 'vessels': w['mmsi'].nunique(), 'positions': len(w),
                 'north_speed': round(nb['sog'].median(), 2), 'north_current': round(nb['cur_kn'].median(), 2),
                 'south_speed': round(sb['sog'].median(), 2), 'south_current': round(sb['cur_kn'].median(), 2),
                 'slope': round(slope, 2), 'r': round(rr, 2)})
    pooled.append(w[['sog_anom', 'cur_anom']])
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv('results/validation_days.csv', index=False)
P = pd.concat(pooled)
s_all = np.polyfit(P['cur_anom'], P['sog_anom'], 1)[0]
r_all = np.corrcoef(P['cur_anom'], P['sog_anom'])[0, 1]
print('\n' + out.to_string(index=False))
print(f'\nALL DAYS: positions {len(P):,} | slope {s_all:.2f} | r {r_all:.2f} | r^2 {r_all ** 2:.2f}')
print(f'Slope range across days: {out["slope"].min()} to {out["slope"].max()}')
