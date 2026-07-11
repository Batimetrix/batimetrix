import requests

TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InByb2plY3QyMDI2IiwiZXhwIjoxNzg3ODQ5NzMxLCJpYXQiOjE3ODI2NjU3MzEsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.yfkmX7r1EoL2hc1v-2JWWWVe42v_Q5qw1Ck4g2tqNPC2qO6tLNwnWySgD0R7tKAKVB9pQG1E53K4c7EXtbclj8BtK0O1_8o0wBMGlNbYOtmqJqzRX02vIqWgD4gRdYU7GwKHFHPXuohZKmehMT9X8-9ZlIENtwh9dkPfPmeeCdTx_ogz4ELshpu5LHCLpTthrQpsec1NYYrbdeJYrnS4sa6YHJt26e8pwLWgva4_h-OzwHgGRS12iygJb4QzHfrC-ZDCfvhcySNwhAE7f3XedoxgbHQi5jquVHTQTO3RRBIWpEEWMZFDYroVMjaB6c2GZDluaHJtOinEYZ7pW87beg"
headers = {"Authorization": f"Bearer {TOKEN}"}

print("=== SWOT Granul Link Bulucu ===")

url = "https://cmr.earthdata.nasa.gov/search/granules.json"
params = {
    "short_name": "SWOT_L2_LR_SSH_Basic_D",
    "bounding_box": "29,41,35,43",
    "page_size": 1,
    "sort_key": "-start_date",
}

r = requests.get(url, headers=headers, params=params, timeout=30)
granules = r.json().get("feed", {}).get("entry", [])

if granules:
    g = granules[0]
    print(f"Granul: {g.get('title')}")
    print(f"Zaman: {g.get('time_start')}")
    print("\nTum linkler:")
    for link in g.get("links", []):
        href = link.get("href", "")
        tip  = link.get("type", "")
        print(f"  [{tip}] {href}")
