import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

TOKEN = "BURAYA_TOKEN"

print("=== SWOT OPeNDAP Gercek Veri Cekim ===")

# OPeNDAP URL - alt kume cekim
OPENDAP_URL = (
    "https://opendap.earthdata.nasa.gov/collections/C3233945000-POCLOUD/"
    "granules/SWOT_L2_LR_SSH_Basic_052_078_20260622T015415_20260622T024543_PID0_01"
)

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {TOKEN}"})

# Once metadata kontrol
print("OPeNDAP metadata checking...")
r = session.get(OPENDAP_URL + ".das", timeout=30)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    print("OPeNDAP erisim basarili!")
    print("\nDosya icerigi (ilk 500 karakter):")
    print(r.text[:500])
elif r.status_code == 401:
    print("Token hatasi - token'i kontrol et")
elif r.status_code == 404:
    print("Dosya bulunamadi - URL degismis olabilir")
    print("Alternatif deneniyor...")
    
    # Alternatif OPeNDAP URL
    ALT_URL = (
        "https://opendap.earthdata.nasa.gov/providers/POCLOUD/"
        "collections/SWOT_L2_LR_SSH_Basic_D/granules/"
        "SWOT_L2_LR_SSH_Basic_052_078_20260622T015415_20260622T024543_PID0_01"
    )
    r2 = session.get(ALT_URL + ".das", timeout=30)
    print(f"Alternatif status: {r2.status_code}")
    if r2.status_code == 200:
        print(r2.text[:300])
else:
    print(f"Beklenmedik status: {r.status_code}")
    print(r.text[:200])
