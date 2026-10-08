import ast, json, glob, datetime
import numpy as np
import xarray as xr

# 1) ROUTES sozlugunu app2.py'den guvenli sekilde oku (calistirmadan)
src = open('app2.py', encoding='utf-8').read()
tree = ast.parse(src)
ROUTES = None
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'ROUTES' for t in node.targets):
        ROUTES = ast.literal_eval(node.value)
        break
if ROUTES is None:
    raise SystemExit('ROUTES not found')
print('Routes found:', len(ROUTES))

# 2) En son NASA-SSH dosyasini ac
f = sorted(glob.glob('ssh_data/*.nc'))[-1]
ds = xr.open_dataset(open(f, 'rb'), engine='h5netcdf')
ssha = ds['ssha'].values
counts = ds['counts'].values
lats = ds['latitude'].values
lons = ds['longitude'].values

g, omega, R = 9.81, 7.2921e-5, 6371000.0
dlat = np.deg2rad(0.5) * R

def sample(lat, lon):
    lon360 = lon % 360
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon360)))
    nearest = False
    if np.isnan(ssha[i, j]):
        found = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, (j + dj) % len(lons)
                if 0 <= ii < len(lats) and not np.isnan(ssha[ii, jj]):
                    found = (ii, jj); break
            if found: break
        if not found:
            return None
        i, j = found
        nearest = True
    out = {'ssha': round(float(ssha[i, j]), 4),
           'counts': int(counts[i, j]),
           'nearest_cell': nearest,
           'u': None, 'v': None, 'speed': None}
    if abs(lat) > 5 and 1 <= i < len(lats) - 1:
        fc = 2 * omega * np.sin(np.deg2rad(lat))
        dlon = np.deg2rad(0.5) * R * np.cos(np.deg2rad(lat))
        n, s = ssha[i + 1, j], ssha[i - 1, j]
        e, w = ssha[i, (j + 1) % len(lons)], ssha[i, (j - 1) % len(lons)]
        if not np.isnan([n, s, e, w]).any():
            u = -(g / fc) * (n - s) / (2 * dlat)
            v = (g / fc) * (e - w) / (2 * dlon)
            out['u'] = round(float(u), 4)
            out['v'] = round(float(v), 4)
            out['speed'] = round(float((u ** 2 + v ** 2) ** 0.5), 4)
    return out

# 3) Tum rota noktalarini hesapla
points = {}
total = with_ssh = with_cur = 0
for rkey, route in ROUTES.items():
    for wp in route.get('waypoints', []):
        key = f"{wp['lat']:.2f},{wp['lon']:.2f}"
        if key in points:
            continue
        total += 1
        r = sample(wp['lat'], wp['lon'])
        points[key] = r
        if r:
            with_ssh += 1
            if r['speed'] is not None:
                with_cur += 1

meta = {
    'source': 'NASA_SSH_REF_SIMPLE_GRID_V11 (NASA PO.DAAC)',
    'file': f.replace('\\', '/').split('/')[-1],
    'data_start': ds.attrs.get('time_coverage_start'),
    'data_end': ds.attrs.get('time_coverage_end'),
    'grid_resolution_deg': 0.5,
    'generated_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'current_note': 'Anomaly geostrophic current derived from SSHA gradients (not total current).',
}
json.dump({'meta': meta, 'points': points}, open('ssh_live.json', 'w', encoding='utf-8'), indent=1)

print('Unique waypoints:', total)
print('With SSH data   :', with_ssh)
print('With current    :', with_cur)
print('Saved ssh_live.json')
