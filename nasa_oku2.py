import netCDF4 as nc
import numpy as np

print("=== SWOT NetCDF Gercek Veri Okuma ===")

DOSYA = "swot_karadeniz.nc"

print(f"Aciliyor: {DOSYA}")
ds = nc.Dataset(DOSYA, "r")

print("\nDegiskenler:")
for v in ds.variables:
    var = ds.variables[v]
    print(f"  {v}: {var.shape} — {getattr(var, 'long_name', '')}")

print("\nKoordinatlar:")
for d in ds.dimensions:
    print(f"  {d}: {len(ds.dimensions[d])}")

# SSH bul
ssh_var = None
for v in ds.variables:
    if any(x in v.lower() for x in ["ssh", "sea_surface", "height", "ssha"]):
        ssh_var = v
        print(f"\nSSH degiskeni bulundu: {v}")
        break

if ssh_var:
    ssh = ds.variables[ssh_var][:]
    print(f"SSH sekli: {ssh.shape}")
    print(f"SSH aralik: {float(np.nanmin(ssh)):.4f} - {float(np.nanmax(ssh)):.4f} m")
    print(f"SSH ortalama: {float(np.nanmean(ssh)):.4f} m")
    print(f"Gecerli nokta sayisi: {int(np.sum(~np.isnan(ssh.data))):,}")

ds.close()
print("\nGercek NASA SWOT verisi basariyla okundu!")