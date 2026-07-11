import requests
import numpy as np

TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InByb2plY3QyMDI2IiwiZXhwIjoxNzg3ODQ5NzMxLCJpYXQiOjE3ODI2NjU3MzEsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.yfkmX7r1EoL2hc1v-2JWWWVe42v_Q5qw1Ck4g2tqNPC2qO6tLNwnWySgD0R7tKAKVB9pQG1E53K4c7EXtbclj8BtK0O1_8o0wBMGlNbYOtmqJqzRX02vIqWgD4gRdYU7GwKHFHPXuohZKmehMT9X8-9ZlIENtwh9dkPfPmeeCdTx_ogz4ELshpu5LHCLpTthrQpsec1NYYrbdeJYrnS4sa6YHJt26e8pwLWgva4_h-OzwHgGRS12iygJb4QzHfrC-ZDCfvhcySNwhAE7f3XedoxgbHQi5jquVHTQTO3RRBIWpEEWMZFDYroVMjaB6c2GZDluaHJtOinEYZ7pW87beg"
headers = {"Authorization": f"Bearer {TOKEN}"}

print("=== SWOT NetCDF File Check ===")

# Real NetCDF link
NC_URL = (
    "https://archive.swot.podaac.earthdata.nasa.gov/"
    "podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/"
    "SWOT_L2_LR_SSH_Basic_052_078_20260622T015415_"
    "20260622T024543_PID0_01.nc"
)

print(f"Checking URL...")
r = requests.head(NC_URL, headers=headers, timeout=30)
print(f"HTTP Status: {r.status_code}")

if "Content-Length" in r.headers:
    boyut_mb = int(r.headers["Content-Length"]) / 1024 / 1024
    print(f"File size: {boyut_mb:.1f} MB")
    if boyut_mb < 50:
        print("Small file - downloadable!")
    elif boyut_mb < 200:
        print("Medium size - download may take 5-10 minutes")
    else:
        print("Large file - will fetch subset via OPeNDAP")
else:
    print("Boyut bilgisi alinamadi")
    print("Headers:", dict(r.headers))
