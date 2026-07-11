import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("=== SWOT NetCDF Gercek Veri Okuma ===")

# --- NetCDF Ac ---
print("swot_blacksea.nc opening...")
ds = xr.open_dataset(
    r"C:\Users\ACER\OneDrive\Masaüstü\Batimetrix\shore\swot_blacksea.nc"
)

print("\nDosya icerigi:")
print(f"Degiskenler: {list(ds.data_vars)}")
print(f"Koordinatlar: {list(ds.coords)}")
print(f"Boyutlar: {dict(ds.dims)}")

# --- SSH Verisini Cek ---
print("\nSSH verisi aranıyor...")
ssh_var = None
for v in ds.data_vars:
    if "ssh" in v.lower() or "height" in v.lower() or "sea" in v.lower():
        print(f"Bulundu: {v} — {ds[v].dims} — {ds[v].shape}")
        ssh_var = v
        break

if ssh_var is None:
    print("Tum degiskenler:")
    for v in ds.data_vars:
        print(f"  {v}: {ds[v].dims} {ds[v].shape}")
    ssh_var = list(ds.data_vars)[0]
    print(f"\nIlk degisken kullanilacak: {ssh_var}")

# --- Koordinatları Cek ---
lat_var = next((v for v in ds.coords if "lat" in v.lower()), None)
lon_var = next((v for v in ds.coords if "lon" in v.lower()), None)

print(f"\nKoordinatlar: lat={lat_var}, lon={lon_var}")

ssh_data = ds[ssh_var].values.flatten()
if lat_var:
    lat_data = ds[lat_var].values.flatten()
    lon_data = ds[lon_var].values.flatten()
else:
    lat_data = np.zeros_like(ssh_data)
    lon_data = np.zeros_like(ssh_data)

# NaN temizle
mask = ~np.isnan(ssh_data) & ~np.isnan(lat_data) & ~np.isnan(lon_data)
ssh_clean = ssh_data[mask]
lat_clean = lat_data[mask]
lon_clean = lon_data[mask]

print(f"\nTemiz veri noktalari: {len(ssh_clean):,}")
print(f"SSH aralik: {ssh_clean.min():.4f} - {ssh_clean.max():.4f} m")
print(f"Lat aralik: {lat_clean.min():.2f} - {lat_clean.max():.2f}")
print(f"Lon aralik: {lon_clean.min():.2f} - {lon_clean.max():.2f}")

# Karadeniz filtrele
kb_mask = (lat_clean > 41) & (lat_clean < 43) & \
          (lon_clean > 29) & (lon_clean < 35)
ssh_kb = ssh_clean[kb_mask]
lat_kb = lat_clean[kb_mask]
lon_kb = lon_clean[kb_mask]

print(f"\nKaradeniz noktalari: {len(ssh_kb):,}")
if len(ssh_kb) > 0:
    print(f"KB SSH aralik: {ssh_kb.min():.4f} - {ssh_kb.max():.4f} m")
    print(f"KB SSH ortalama: {ssh_kb.mean():.4f} m")

print("\nGercek NASA SWOT verisi basariyla okundu!")
ds.close()
