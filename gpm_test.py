import requests
import json

TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InByb2plY3QyMDI2IiwiZXhwIjoxNzg3ODQ5NzMxLCJpYXQiOjE3ODI2NjU3MzEsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.yfkmX7r1EoL2hc1v-2JWWWVe42v_Q5qw1Ck4g2tqNPC2qO6tLNwnWySgD0R7tKAKVB9pQG1E53K4c7EXtbclj8BtK0O1_8o0wBMGlNbYOtmqJqzRX02vIqWgD4gRdYU7GwKHFHPXuohZKmehMT9X8-9ZlIENtwh9dkPfPmeeCdTx_ogz4ELshpu5LHCLpTthrQpsec1NYYrbdeJYrnS4sa6YHJt26e8pwLWgva4_h-OzwHgGRS12iygJb4QzHfrC-ZDCfvhcySNwhAE7f3XedoxgbHQi5jquVHTQTO3RRBIWpEEWMZFDYroVMjaB6c2GZDluaHJtOinEYZ7pW87beg"
headers = {"Authorization": f"Bearer {TOKEN}"}

print("=== Batimetrix — GPM Firtina Verisi ===")
print("Global Precipitation Measurement uydusu baglaniyor...")

# --- GPM Granul Ara ---
url = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Karadeniz bölgesi için GPM IMERG verisi
params = {
    "short_name": "GPM_3IMERGHH",  # GPM IMERG Half-Hourly
    "bounding_box": "29,41,35,43",  # Karadeniz
    "page_size": 3,
    "sort_key": "-start_date",
}

print("\nGPM IMERG verisi aranıyor...")
r = requests.get(url, headers=headers, params=params, timeout=30)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    data = r.json()
    granules = data.get("feed", {}).get("entry", [])
    print(f"{len(granules)} GPM granulu found!")
    for g in granules:
        print(f"  - {g.get('title', 'isimsiz')}")
        print(f"    Zaman: {g.get('time_start', '?')} → {g.get('time_end', '?')}")
else:
    print(f"Hata: {r.status_code}")
    # Alternatif koleksiyon dene
    print("\nAlternatif GPM koleksiyonu deneniyor...")
    params["short_name"] = "GPM_3IMERGDF"  # GPM IMERG Daily Final
    r2 = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"Status: {r2.status_code}")
    if r2.status_code == 200:
        granules = r2.json().get("feed", {}).get("entry", [])
        print(f"{len(granules)} GPM granulu found!")
        for g in granules:
            print(f"  - {g.get('title', 'isimsiz')}")

# --- GPM Koleksiyon Listesi ---
print("\n=== Mevcut GPM Koleksiyonlari ===")
kol_url = "https://cmr.earthdata.nasa.gov/search/collections.json"
kol_params = {
    "keyword": "GPM precipitation",
    "page_size": 5,
}
r3 = requests.get(kol_url, params=kol_params, timeout=30)
if r3.status_code == 200:
    koleksiyonlar = r3.json().get("feed", {}).get("entry", [])
    for k in koleksiyonlar:
        print(f"  [{k.get('short_name', '?')}] {k.get('title', '?')[:60]}")

print("\nGPM test completed!")
