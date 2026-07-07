import requests
import os

TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InByb2plY3QyMDI2IiwiZXhwIjoxNzg3ODQ5NzMxLCJpYXQiOjE3ODI2NjU3MzEsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.yfkmX7r1EoL2hc1v-2JWWWVe42v_Q5qw1Ck4g2tqNPC2qO6tLNwnWySgD0R7tKAKVB9pQG1E53K4c7EXtbclj8BtK0O1_8o0wBMGlNbYOtmqJqzRX02vIqWgD4gRdYU7GwKHFHPXuohZKmehMT9X8-9ZlIENtwh9dkPfPmeeCdTx_ogz4ELshpu5LHCLpTthrQpsec1NYYrbdeJYrnS4sa6YHJt26e8pwLWgva4_h-OzwHgGRS12iygJb4QzHfrC-ZDCfvhcySNwhAE7f3XedoxgbHQi5jquVHTQTO3RRBIWpEEWMZFDYroVMjaB6c2GZDluaHJtOinEYZ7pW87beg"

print("=== SWOT NetCDF Direkt Indirme ===")

NC_URL = (
    "https://archive.swot.podaac.earthdata.nasa.gov/"
    "podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/"
    "SWOT_L2_LR_SSH_Basic_052_078_20260622T015415_"
    "20260622T024543_PID0_01.nc"
)

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "User-Agent": "Batimetrix/1.0"
})

print("Indirme basliyor...")
r = session.get(NC_URL, allow_redirects=True, stream=True, timeout=60)
print(f"Final URL: {r.url[:80]}...")
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type', 'bilinmiyor')}")

boyut = int(r.headers.get('Content-Length', 0))
print(f"Dosya boyutu: {boyut/1024/1024:.1f} MB")

if r.status_code == 200 and boyut > 0:
    print("Indiriliyor...")
    with open("swot_karadeniz.nc", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    gercek_boyut = os.path.getsize("swot_karadeniz.nc") / 1024 / 1024
    print(f"Kaydedildi: swot_karadeniz.nc ({gercek_boyut:.1f} MB)")
else:
    print(f"Indirme basarisiz: {r.status_code}")
    print(r.text[:300])