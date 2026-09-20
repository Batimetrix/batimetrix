# BATIMETRIX V3 PRO - 7 dil + Leaflet harita + Chart.js analiz
from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import numpy as np
import math

app = Flask(__name__)

# --- Security: rate limiting ---
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app,
                      default_limits=["120 per minute"],
                      storage_uri="memory://")
except ImportError:
    class _NoLimiter:
        def limit(self, *_a, **_k):
            def deco(f): return f
            return deco
    limiter = _NoLimiter()

# --- Security: response headers ---
@app.after_request
def _secure_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp

# --- Security: clean error responses (stack trace sizdirma) ---
@app.errorhandler(404)
def _e404(e): return jsonify({"error": "not found"}), 404

@app.errorhandler(429)
def _e429(e): return jsonify({"error": "rate limit exceeded, slow down"}), 429

@app.errorhandler(500)
def _e500(e): return jsonify({"error": "server error"}), 500


class GucluPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.giris = nn.Sequential(nn.Linear(7,512), nn.LayerNorm(512), nn.GELU())
        self.katmanlar = nn.ModuleList([
            nn.Sequential(nn.Linear(512,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.05))
            for _ in range(6)
        ])
        self.cikis = nn.Sequential(
            nn.Linear(512,128), nn.GELU(),
            nn.Linear(128,32), nn.GELU(),
            nn.Linear(32,1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.giris(x)
        for k in self.katmanlar: h = h + k(h)
        return self.cikis(h)

model = GucluPINN()
model.load_state_dict(torch.load("batimetrix_swot_real.pt", weights_only=True))
model.eval()
print("Model yuklendi! BATIMETRIX V3 PRO")

CII_REF = {
    "Aframax Tanker": {"a": 5247.0, "c": 0.61, "d": [0.82,0.93,1.08,1.28]},
    "Suezmax Tanker": {"a": 5247.0, "c": 0.61, "d": [0.82,0.93,1.08,1.28]},
    "MR Product Tanker": {"a": 5247.0, "c": 0.61, "d": [0.82,0.93,1.08,1.28]},
    "ULCC Tanker": {"a": 5247.0, "c": 0.61, "d": [0.82,0.93,1.08,1.28]},
    "Supramax Bulk": {"a": 4745.0, "c": 0.622, "d": [0.86,0.94,1.06,1.18]},
    "Panamax Bulk": {"a": 4745.0, "c": 0.622, "d": [0.86,0.94,1.06,1.18]},
    "VLOC Valemax": {"a": 4745.0, "c": 0.622, "d": [0.86,0.94,1.06,1.18], "cap": 279000},
    "ULCV Container": {"a": 1984.0, "c": 0.489, "d": [0.83,0.94,1.07,1.19]},
    "Feeder Container": {"a": 1984.0, "c": 0.489, "d": [0.83,0.94,1.07,1.19]},
    "VLCC Tanker":       {"a":5247.0,"c":0.610, "d": [0.82,0.93,1.08,1.28]},
    "Panamax Container": {"a":1984.0,"c":0.489, "d": [0.83,0.94,1.07,1.19]},
    "Capesize Bulk":     {"a":4745.0,"c":0.622, "d": [0.86,0.94,1.06,1.18]},
    "LNG Carrier":       {"a":9.827, "c":0.000, "d": [0.89,0.98,1.06,1.13]},
    "Handy Bulk":        {"a":588.0, "c":0.3885, "d": [0.86,0.94,1.06,1.18]},
    "Black Sea Cargo":   {"a":588.0, "c":0.3885, "d": [0.83,0.94,1.06,1.19]},
}
VESSEL_PROFILE = {
    "Aframax Tanker": {"dwt":110000,"fuel":45},
    "Suezmax Tanker": {"dwt":160000,"fuel":55},
    "MR Product Tanker": {"dwt":50000,"fuel":32},
    "ULCC Tanker": {"dwt":400000,"fuel":95},
    "Supramax Bulk": {"dwt":58000,"fuel":32},
    "Panamax Bulk": {"dwt":75000,"fuel":32},
    "VLOC Valemax": {"dwt":400000,"fuel":65},
    "ULCV Container": {"dwt":220000,"fuel":150},
    "Feeder Container": {"dwt":20000,"fuel":22},
    "VLCC Tanker":       {"dwt":300000,"fuel":75},
    "Panamax Container": {"dwt": 65000,"fuel": 80},
    "Capesize Bulk":     {"dwt":180000,"fuel": 40},
    "LNG Carrier":       {"dwt":100000,"fuel": 65},
    "Handy Bulk":        {"dwt": 35000,"fuel": 25},
    "Black Sea Cargo":   {"dwt":  8000,"fuel": 12},
}

ROUTES = {
    "istanbul_trabzon": {
        "name": "Istanbul → Trabzon",
        "waypoints": [
            {"name":"Istanbul Strait","lat":41.10,"lon":29.05,"depth":35,"ssh":0.05},
            {"name":"BS Entrance","lat":41.30,"lon":29.50,"depth":120,"ssh":0.07},
            {"name":"West BS","lat":41.80,"lon":30.50,"depth":650,"ssh":0.08},
            {"name":"Zonguldak","lat":41.60,"lon":31.80,"depth":850,"ssh":0.08},
            {"name":"Sinop","lat":42.00,"lon":35.10,"depth":950,"ssh":0.10},
            {"name":"Samsun","lat":41.70,"lon":36.20,"depth":800,"ssh":0.11},
            {"name":"Trabzon","lat":41.00,"lon":39.73,"depth":200,"ssh":0.06},
        ]
    },
    "istanbul_novorossiysk": {
        "name": "Istanbul → Novorossiysk",
        "waypoints": [
            {"name":"Istanbul Strait","lat":41.10,"lon":29.05,"depth":35,"ssh":0.05},
            {"name":"West BS","lat":41.80,"lon":30.50,"depth":650,"ssh":0.08},
            {"name":"Central BS","lat":42.10,"lon":33.00,"depth":1100,"ssh":0.09},
            {"name":"Novorossiysk","lat":44.72,"lon":37.77,"depth":120,"ssh":0.06},
        ]
    },
    "odessa_istanbul": {
        "name": "Odessa → Istanbul",
        "waypoints": [
            {"name":"Odessa","lat":46.48,"lon":30.73,"depth":80,"ssh":0.07},
            {"name":"West BS","lat":44.00,"lon":31.00,"depth":800,"ssh":0.10},
            {"name":"Central BS","lat":42.50,"lon":32.00,"depth":1100,"ssh":0.09},
            {"name":"Istanbul Strait","lat":41.10,"lon":29.05,"depth":35,"ssh":0.05},
        ]
    },
    "batumi_constanta": {
        "name": "Batumi → Constanta",
        "waypoints": [
            {"name":"Batumi","lat":41.65,"lon":41.64,"depth":150,"ssh":0.07},
            {"name":"East BS","lat":42.00,"lon":38.00,"depth":900,"ssh":0.09},
            {"name":"Central BS","lat":42.20,"lon":33.00,"depth":1100,"ssh":0.08},
            {"name":"Constanta","lat":44.17,"lon":28.65,"depth":60,"ssh":0.06},
        ]
    },
    "karadeniz_sakin": {
        "name": "Black Sea — Calm",
        "waypoints": [
            {"name":"Point 1","lat":41.50,"lon":30.00,"depth":600,"ssh":0.06},
            {"name":"Point 2","lat":41.80,"lon":32.00,"depth":900,"ssh":0.08},
            {"name":"Point 3","lat":42.10,"lon":34.00,"depth":1100,"ssh":0.09},
            {"name":"Point 4","lat":42.00,"lon":36.00,"depth":950,"ssh":0.10},
        ]
    },
    "atlantik": {
        "name": "North Atlantic — Storm",
        "waypoints": [
            {"name":"Biscay","lat":45.00,"lon":-5.00,"depth":2800,"ssh":0.25},
            {"name":"Mid-Atlantic","lat":48.00,"lon":-15.00,"depth":3500,"ssh":0.35},
            {"name":"Deep Atlantic","lat":50.00,"lon":-25.00,"depth":4200,"ssh":0.40},
            {"name":"N.Atlantic","lat":52.00,"lon":-30.00,"depth":3800,"ssh":0.38},
        ]
    },
    "shanghai_rotterdam": {
        "name": "Shanghai → Rotterdam",
        "waypoints": [
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Indian Ocean","lat":5.0,"lon":75.0,"depth":3800,"ssh":0.2},
            {"name":"Suez","lat":29.97,"lon":32.55,"depth":20,"ssh":0.05},
            {"name":"Mediterranean","lat":36.0,"lon":15.0,"depth":2500,"ssh":0.12},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "shanghai_losangeles": {
        "name": "Shanghai → Los Angeles",
        "waypoints": [
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
            {"name":"Pacific West","lat":35.0,"lon":150.0,"depth":5500,"ssh":0.3},
            {"name":"Mid-Pacific","lat":38.0,"lon":180.0,"depth":5800,"ssh":0.35},
            {"name":"Pacific East","lat":35.0,"lon":-140.0,"depth":4500,"ssh":0.28},
            {"name":"Los Angeles","lat":33.74,"lon":-118.27,"depth":20,"ssh":0.06},
        ]
    },
    "singapore_shanghai": {
        "name": "Singapore → Shanghai",
        "waypoints": [
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
            {"name":"South China Sea","lat":10.0,"lon":110.0,"depth":4000,"ssh":0.15},
            {"name":"Taiwan Strait","lat":24.0,"lon":119.0,"depth":60,"ssh":0.12},
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
        ]
    },
    "rotterdam_newyork": {
        "name": "Rotterdam → New York",
        "waypoints": [
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
            {"name":"English Channel","lat":50.0,"lon":-1.0,"depth":50,"ssh":0.1},
            {"name":"North Atlantic","lat":48.0,"lon":-30.0,"depth":4200,"ssh":0.35},
            {"name":"Grand Banks","lat":44.0,"lon":-50.0,"depth":100,"ssh":0.25},
            {"name":"New York","lat":40.6,"lon":-74.05,"depth":20,"ssh":0.06},
        ]
    },
    "santos_rotterdam": {
        "name": "Santos → Rotterdam",
        "waypoints": [
            {"name":"Santos","lat":-23.98,"lon":-46.3,"depth":15,"ssh":0.07},
            {"name":"South Atlantic","lat":-15.0,"lon":-30.0,"depth":4500,"ssh":0.22},
            {"name":"Equator Atlantic","lat":0.0,"lon":-25.0,"depth":4000,"ssh":0.18},
            {"name":"Canary Islands","lat":28.0,"lon":-18.0,"depth":3500,"ssh":0.15},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "dubai_singapore": {
        "name": "Dubai → Singapore",
        "waypoints": [
            {"name":"Dubai","lat":25.27,"lon":55.3,"depth":20,"ssh":0.05},
            {"name":"Arabian Sea","lat":18.0,"lon":62.0,"depth":3500,"ssh":0.16},
            {"name":"Indian Ocean","lat":8.0,"lon":75.0,"depth":3800,"ssh":0.18},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "mumbai_suez": {
        "name": "Mumbai → Suez",
        "waypoints": [
            {"name":"Mumbai","lat":18.94,"lon":72.83,"depth":15,"ssh":0.06},
            {"name":"Arabian Sea","lat":18.0,"lon":62.0,"depth":3500,"ssh":0.16},
            {"name":"Gulf of Aden","lat":12.5,"lon":45.0,"depth":2000,"ssh":0.14},
            {"name":"Red Sea","lat":20.0,"lon":38.0,"depth":1800,"ssh":0.12},
            {"name":"Suez","lat":29.97,"lon":32.55,"depth":20,"ssh":0.05},
        ]
    },
    "durban_singapore": {
        "name": "Durban → Singapore",
        "waypoints": [
            {"name":"Durban","lat":-29.87,"lon":31.03,"depth":20,"ssh":0.1},
            {"name":"Indian Ocean South","lat":-20.0,"lon":55.0,"depth":4200,"ssh":0.22},
            {"name":"Indian Ocean","lat":-5.0,"lon":80.0,"depth":4000,"ssh":0.18},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "panama_losangeles": {
        "name": "Panama → Los Angeles",
        "waypoints": [
            {"name":"Panama Canal","lat":9.08,"lon":-79.68,"depth":20,"ssh":0.06},
            {"name":"Pacific Central","lat":15.0,"lon":-95.0,"depth":3800,"ssh":0.2},
            {"name":"Baja California","lat":25.0,"lon":-112.0,"depth":3000,"ssh":0.16},
            {"name":"Los Angeles","lat":33.74,"lon":-118.27,"depth":20,"ssh":0.06},
        ]
    },
    "tokyo_losangeles": {
        "name": "Tokyo → Los Angeles",
        "waypoints": [
            {"name":"Tokyo","lat":35.65,"lon":139.84,"depth":20,"ssh":0.07},
            {"name":"Pacific NW","lat":40.0,"lon":160.0,"depth":5500,"ssh":0.32},
            {"name":"Mid-Pacific","lat":42.0,"lon":-175.0,"depth":5800,"ssh":0.35},
            {"name":"Pacific NE","lat":38.0,"lon":-140.0,"depth":4500,"ssh":0.28},
            {"name":"Los Angeles","lat":33.74,"lon":-118.27,"depth":20,"ssh":0.06},
        ]
    },
    "hamburg_newyork": {
        "name": "Hamburg → New York",
        "waypoints": [
            {"name":"Hamburg","lat":53.55,"lon":9.99,"depth":15,"ssh":0.04},
            {"name":"North Sea","lat":55.0,"lon":3.0,"depth":50,"ssh":0.1},
            {"name":"North Atlantic","lat":52.0,"lon":-30.0,"depth":4000,"ssh":0.35},
            {"name":"New York","lat":40.6,"lon":-74.05,"depth":20,"ssh":0.06},
        ]
    },
    "melbourne_shanghai": {
        "name": "Melbourne → Shanghai",
        "waypoints": [
            {"name":"Melbourne","lat":-37.84,"lon":144.92,"depth":20,"ssh":0.12},
            {"name":"Coral Sea","lat":-18.0,"lon":155.0,"depth":3500,"ssh":0.2},
            {"name":"Philippine Sea","lat":10.0,"lon":130.0,"depth":5000,"ssh":0.22},
            {"name":"East China Sea","lat":28.0,"lon":125.0,"depth":150,"ssh":0.12},
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
        ]
    },
    "jeddah_rotterdam": {
        "name": "Jeddah → Rotterdam",
        "waypoints": [
            {"name":"Jeddah","lat":21.49,"lon":39.19,"depth":30,"ssh":0.06},
            {"name":"Red Sea","lat":25.0,"lon":36.0,"depth":1800,"ssh":0.12},
            {"name":"Suez","lat":29.97,"lon":32.55,"depth":20,"ssh":0.05},
            {"name":"Mediterranean","lat":36.0,"lon":15.0,"depth":2500,"ssh":0.12},
            {"name":"Gibraltar","lat":36.14,"lon":-5.35,"depth":300,"ssh":0.08},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "busan_losangeles": {
        "name": "Busan → Los Angeles",
        "waypoints": [
            {"name":"Busan","lat":35.1,"lon":129.04,"depth":20,"ssh":0.07},
            {"name":"Pacific NW","lat":42.0,"lon":160.0,"depth":5500,"ssh":0.32},
            {"name":"Mid-Pacific","lat":44.0,"lon":-175.0,"depth":5800,"ssh":0.35},
            {"name":"Los Angeles","lat":33.74,"lon":-118.27,"depth":20,"ssh":0.06},
        ]
    },
    "gibraltar_piraeus": {
        "name": "Gibraltar → Piraeus",
        "waypoints": [
            {"name":"Gibraltar","lat":36.14,"lon":-5.35,"depth":300,"ssh":0.08},
            {"name":"Alboran Sea","lat":36.5,"lon":-2.0,"depth":1500,"ssh":0.1},
            {"name":"Sardinia","lat":38.5,"lon":8.0,"depth":2800,"ssh":0.14},
            {"name":"Ionian Sea","lat":37.5,"lon":18.0,"depth":3000,"ssh":0.13},
            {"name":"Piraeus","lat":37.94,"lon":23.65,"depth":40,"ssh":0.06},
        ]
    },
    "hongkong_singapore": {
        "name": "Hong Kong → Singapore",
        "waypoints": [
            {"name":"Hong Kong","lat":22.3,"lon":114.17,"depth":25,"ssh":0.08},
            {"name":"South China Sea","lat":15.0,"lon":113.0,"depth":4000,"ssh":0.15},
            {"name":"Natuna Sea","lat":4.0,"lon":108.0,"depth":80,"ssh":0.1},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "newyork_santos": {
        "name": "New York → Santos",
        "waypoints": [
            {"name":"New York","lat":40.6,"lon":-74.05,"depth":20,"ssh":0.06},
            {"name":"Caribbean","lat":20.0,"lon":-65.0,"depth":4000,"ssh":0.18},
            {"name":"Equator Atlantic","lat":0.0,"lon":-40.0,"depth":4200,"ssh":0.16},
            {"name":"Santos","lat":-23.98,"lon":-46.3,"depth":15,"ssh":0.07},
        ]
    },
    "capetown_singapore": {
        "name": "Cape Town → Singapore",
        "waypoints": [
            {"name":"Cape Town","lat":-33.91,"lon":18.42,"depth":30,"ssh":0.12},
            {"name":"Indian Ocean SW","lat":-25.0,"lon":50.0,"depth":4500,"ssh":0.24},
            {"name":"Indian Ocean","lat":-5.0,"lon":80.0,"depth":4000,"ssh":0.18},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "yokohama_singapore": {
        "name": "Yokohama → Singapore",
        "waypoints": [
            {"name":"Yokohama","lat":35.44,"lon":139.64,"depth":20,"ssh":0.07},
            {"name":"Philippine Sea","lat":20.0,"lon":130.0,"depth":5000,"ssh":0.22},
            {"name":"South China Sea","lat":10.0,"lon":115.0,"depth":4000,"ssh":0.15},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "antwerp_newyork": {
        "name": "Antwerp → New York",
        "waypoints": [
            {"name":"Antwerp","lat":51.24,"lon":4.42,"depth":20,"ssh":0.04},
            {"name":"English Channel","lat":50.0,"lon":-2.0,"depth":50,"ssh":0.1},
            {"name":"North Atlantic","lat":47.0,"lon":-35.0,"depth":4200,"ssh":0.35},
            {"name":"New York","lat":40.6,"lon":-74.05,"depth":20,"ssh":0.06},
        ]
    },
    "valencia_suez": {
        "name": "Valencia → Suez",
        "waypoints": [
            {"name":"Valencia","lat":39.44,"lon":-0.32,"depth":25,"ssh":0.06},
            {"name":"Mediterranean W","lat":38.0,"lon":5.0,"depth":2500,"ssh":0.12},
            {"name":"Mediterranean E","lat":35.0,"lon":20.0,"depth":3000,"ssh":0.13},
            {"name":"Suez","lat":29.97,"lon":32.55,"depth":20,"ssh":0.05},
        ]
    },
    "qingdao_rotterdam": {
        "name": "Qingdao → Rotterdam",
        "waypoints": [
            {"name":"Qingdao","lat":36.07,"lon":120.38,"depth":20,"ssh":0.06},
            {"name":"East China Sea","lat":30.0,"lon":124.0,"depth":120,"ssh":0.1},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Suez","lat":29.97,"lon":32.55,"depth":20,"ssh":0.05},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "colombo_singapore": {
        "name": "Colombo → Singapore",
        "waypoints": [
            {"name":"Colombo","lat":6.94,"lon":79.84,"depth":20,"ssh":0.08},
            {"name":"Bay of Bengal","lat":8.0,"lon":88.0,"depth":3500,"ssh":0.16},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "houston_rotterdam": {
        "name": "Houston → Rotterdam",
        "waypoints": [
            {"name":"Houston","lat":29.73,"lon":-94.98,"depth":15,"ssh":0.06},
            {"name":"Gulf of Mexico","lat":26.0,"lon":-88.0,"depth":2000,"ssh":0.14},
            {"name":"North Atlantic","lat":40.0,"lon":-45.0,"depth":4500,"ssh":0.3},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "seattle_tokyo": {
        "name": "Seattle → Tokyo",
        "waypoints": [
            {"name":"Seattle","lat":47.6,"lon":-122.33,"depth":20,"ssh":0.07},
            {"name":"Pacific NE","lat":48.0,"lon":-150.0,"depth":4500,"ssh":0.28},
            {"name":"Pacific NW","lat":45.0,"lon":175.0,"depth":5500,"ssh":0.32},
            {"name":"Tokyo","lat":35.65,"lon":139.84,"depth":20,"ssh":0.07},
        ]
    },
    "piraeus_alexandria": {
        "name": "Piraeus → Alexandria",
        "waypoints": [
            {"name":"Piraeus","lat":37.94,"lon":23.65,"depth":40,"ssh":0.06},
            {"name":"Aegean Sea","lat":36.0,"lon":26.0,"depth":1000,"ssh":0.1},
            {"name":"East Med","lat":33.0,"lon":28.0,"depth":2500,"ssh":0.12},
            {"name":"Alexandria","lat":31.2,"lon":29.92,"depth":20,"ssh":0.05},
        ]
    },
    "vancouver_shanghai": {
        "name": "Vancouver → Shanghai",
        "waypoints": [
            {"name":"Vancouver","lat":49.29,"lon":-123.11,"depth":25,"ssh":0.07},
            {"name":"Pacific NE","lat":50.0,"lon":-155.0,"depth":4500,"ssh":0.28},
            {"name":"Pacific NW","lat":45.0,"lon":170.0,"depth":5500,"ssh":0.32},
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
        ]
    },
    "istanbul_piraeus": {
        "name": "Istanbul → Piraeus",
        "waypoints": [
            {"name":"Istanbul Strait","lat":41.1,"lon":29.05,"depth":35,"ssh":0.05},
            {"name":"Dardanelles","lat":40.2,"lon":26.4,"depth":60,"ssh":0.07},
            {"name":"Aegean Sea","lat":39.0,"lon":25.0,"depth":1000,"ssh":0.1},
            {"name":"Piraeus","lat":37.94,"lon":23.65,"depth":40,"ssh":0.06},
        ]
    },
    "kobe_singapore": {
        "name": "Kobe → Singapore",
        "waypoints": [
            {"name":"Kobe","lat":34.68,"lon":135.2,"depth":20,"ssh":0.07},
            {"name":"East China Sea","lat":28.0,"lon":125.0,"depth":150,"ssh":0.12},
            {"name":"South China Sea","lat":12.0,"lon":115.0,"depth":4000,"ssh":0.15},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "algeciras_newyork": {
        "name": "Algeciras → New York",
        "waypoints": [
            {"name":"Algeciras","lat":36.13,"lon":-5.45,"depth":300,"ssh":0.08},
            {"name":"Atlantic Mid","lat":38.0,"lon":-25.0,"depth":4000,"ssh":0.3},
            {"name":"Grand Banks","lat":42.0,"lon":-48.0,"depth":150,"ssh":0.25},
            {"name":"New York","lat":40.6,"lon":-74.05,"depth":20,"ssh":0.06},
        ]
    },
    "dalian_singapore": {
        "name": "Dalian → Singapore",
        "waypoints": [
            {"name":"Dalian","lat":38.92,"lon":121.63,"depth":20,"ssh":0.06},
            {"name":"Yellow Sea","lat":34.0,"lon":123.0,"depth":80,"ssh":0.1},
            {"name":"East China Sea","lat":28.0,"lon":124.0,"depth":120,"ssh":0.11},
            {"name":"South China Sea","lat":12.0,"lon":114.0,"depth":4000,"ssh":0.15},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "felixstowe_singapore": {
        "name": "Felixstowe → Singapore",
        "waypoints": [
            {"name":"Felixstowe","lat":51.96,"lon":1.31,"depth":20,"ssh":0.05},
            {"name":"Gibraltar","lat":36.14,"lon":-5.35,"depth":300,"ssh":0.08},
            {"name":"Suez","lat":29.97,"lon":32.55,"depth":20,"ssh":0.05},
            {"name":"Indian Ocean","lat":8.0,"lon":70.0,"depth":3800,"ssh":0.18},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "longbeach_yokohama": {
        "name": "Long Beach → Yokohama",
        "waypoints": [
            {"name":"Long Beach","lat":33.75,"lon":-118.19,"depth":20,"ssh":0.06},
            {"name":"Pacific E","lat":38.0,"lon":-140.0,"depth":4500,"ssh":0.28},
            {"name":"Mid-Pacific","lat":42.0,"lon":-175.0,"depth":5800,"ssh":0.35},
            {"name":"Yokohama","lat":35.44,"lon":139.64,"depth":20,"ssh":0.07},
        ]
    },
    "genoa_alexandria": {
        "name": "Genoa → Alexandria",
        "waypoints": [
            {"name":"Genoa","lat":44.41,"lon":8.93,"depth":25,"ssh":0.06},
            {"name":"Tyrrhenian Sea","lat":40.0,"lon":12.0,"depth":3000,"ssh":0.13},
            {"name":"Ionian Sea","lat":36.0,"lon":18.0,"depth":3500,"ssh":0.14},
            {"name":"Alexandria","lat":31.2,"lon":29.92,"depth":20,"ssh":0.05},
        ]
    },
    "porthedland_qingdao": {
        "name": "Port Hedland → Qingdao (Iron Ore)",
        "waypoints": [
            {"name":"Port Hedland","lat":-20.31,"lon":118.58,"depth":15,"ssh":0.06},
            {"name":"Timor Sea","lat":-11.0,"lon":123.0,"depth":2500,"ssh":0.14},
            {"name":"Makassar Strait","lat":-2.0,"lon":118.0,"depth":2000,"ssh":0.12},
            {"name":"Philippine Sea","lat":10.0,"lon":125.0,"depth":4500,"ssh":0.2},
            {"name":"Taiwan Strait","lat":24.0,"lon":119.0,"depth":60,"ssh":0.12},
            {"name":"Qingdao","lat":36.07,"lon":120.38,"depth":20,"ssh":0.06},
        ]
    },
    "pontamadeira_qingdao": {
        "name": "Ponta da Madeira → Qingdao (Vale Ore)",
        "waypoints": [
            {"name":"Ponta da Madeira","lat":-2.56,"lon":-44.37,"depth":25,"ssh":0.08},
            {"name":"Equator Atlantic","lat":0.0,"lon":-30.0,"depth":4200,"ssh":0.16},
            {"name":"Cape of Good Hope","lat":-35.0,"lon":20.0,"depth":3000,"ssh":0.28},
            {"name":"Indian Ocean","lat":-25.0,"lon":60.0,"depth":4500,"ssh":0.22},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Qingdao","lat":36.07,"lon":120.38,"depth":20,"ssh":0.06},
        ]
    },
    "newcastle_tokyo": {
        "name": "Newcastle → Tokyo (Coal)",
        "waypoints": [
            {"name":"Newcastle AU","lat":-32.92,"lon":151.78,"depth":15,"ssh":0.1},
            {"name":"Coral Sea","lat":-20.0,"lon":155.0,"depth":3500,"ssh":0.2},
            {"name":"Philippine Sea","lat":10.0,"lon":140.0,"depth":5500,"ssh":0.24},
            {"name":"Tokyo","lat":35.65,"lon":139.84,"depth":20,"ssh":0.07},
        ]
    },
    "neworleans_shanghai": {
        "name": "New Orleans → Shanghai (Grain)",
        "waypoints": [
            {"name":"New Orleans","lat":29.95,"lon":-90.07,"depth":15,"ssh":0.05},
            {"name":"Gulf of Mexico","lat":25.0,"lon":-88.0,"depth":2000,"ssh":0.14},
            {"name":"Panama Canal","lat":9.08,"lon":-79.68,"depth":20,"ssh":0.06},
            {"name":"Pacific Central","lat":15.0,"lon":-110.0,"depth":3800,"ssh":0.2},
            {"name":"Mid-Pacific","lat":25.0,"lon":-160.0,"depth":5500,"ssh":0.32},
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
        ]
    },
    "rastanura_ningbo": {
        "name": "Ras Tanura → Ningbo (VLCC Crude)",
        "waypoints": [
            {"name":"Ras Tanura","lat":26.64,"lon":50.16,"depth":20,"ssh":0.04},
            {"name":"Hormuz Strait","lat":26.57,"lon":56.25,"depth":60,"ssh":0.08},
            {"name":"Arabian Sea","lat":18.0,"lon":62.0,"depth":3500,"ssh":0.16},
            {"name":"Indian Ocean","lat":5.0,"lon":80.0,"depth":3800,"ssh":0.18},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Ningbo","lat":29.87,"lon":121.55,"depth":20,"ssh":0.06},
        ]
    },
    "raslaffan_tokyo": {
        "name": "Ras Laffan → Tokyo (LNG)",
        "waypoints": [
            {"name":"Ras Laffan","lat":25.92,"lon":51.55,"depth":20,"ssh":0.04},
            {"name":"Hormuz Strait","lat":26.57,"lon":56.25,"depth":60,"ssh":0.08},
            {"name":"Arabian Sea","lat":18.0,"lon":62.0,"depth":3500,"ssh":0.16},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"South China Sea","lat":12.0,"lon":115.0,"depth":4000,"ssh":0.15},
            {"name":"Tokyo","lat":35.65,"lon":139.84,"depth":20,"ssh":0.07},
        ]
    },
    "sabinepass_rotterdam": {
        "name": "Sabine Pass → Rotterdam (US LNG)",
        "waypoints": [
            {"name":"Sabine Pass","lat":29.73,"lon":-93.87,"depth":15,"ssh":0.05},
            {"name":"Gulf of Mexico","lat":26.0,"lon":-88.0,"depth":2000,"ssh":0.14},
            {"name":"Florida Strait","lat":24.5,"lon":-80.0,"depth":800,"ssh":0.16},
            {"name":"North Atlantic","lat":40.0,"lon":-45.0,"depth":4500,"ssh":0.3},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "bonny_rotterdam": {
        "name": "Bonny → Rotterdam (Crude)",
        "waypoints": [
            {"name":"Bonny","lat":4.45,"lon":7.17,"depth":20,"ssh":0.07},
            {"name":"Gulf of Guinea","lat":2.0,"lon":3.0,"depth":3000,"ssh":0.14},
            {"name":"Equator Atlantic","lat":0.0,"lon":-5.0,"depth":4000,"ssh":0.16},
            {"name":"Canary Islands","lat":28.0,"lon":-18.0,"depth":3500,"ssh":0.15},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "singapore_rotterdam_cape": {
        "name": "Singapore → Rotterdam (Cape Route)",
        "waypoints": [
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
            {"name":"Indian Ocean","lat":-5.0,"lon":80.0,"depth":4000,"ssh":0.18},
            {"name":"Cape of Good Hope","lat":-35.0,"lon":20.0,"depth":3000,"ssh":0.3},
            {"name":"South Atlantic","lat":-15.0,"lon":-5.0,"depth":4500,"ssh":0.22},
            {"name":"Canary Islands","lat":28.0,"lon":-18.0,"depth":3500,"ssh":0.15},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "murmansk_shanghai": {
        "name": "Murmansk → Shanghai (Arctic NSR)",
        "waypoints": [
            {"name":"Murmansk","lat":68.97,"lon":33.05,"depth":30,"ssh":0.06},
            {"name":"Kara Sea","lat":75.0,"lon":65.0,"depth":150,"ssh":0.1},
            {"name":"Laptev Sea","lat":76.0,"lon":125.0,"depth":50,"ssh":0.09},
            {"name":"Bering Strait","lat":65.8,"lon":-169.0,"depth":50,"ssh":0.12},
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
        ]
    },
    "hormuz_transit": {
        "name": "Kuwait → Arabian Sea (Hormuz)",
        "waypoints": [
            {"name":"Kuwait","lat":29.07,"lon":48.13,"depth":15,"ssh":0.04},
            {"name":"Persian Gulf","lat":27.0,"lon":52.0,"depth":60,"ssh":0.06},
            {"name":"Hormuz Strait","lat":26.57,"lon":56.25,"depth":60,"ssh":0.08},
            {"name":"Gulf of Oman","lat":24.0,"lon":58.0,"depth":2000,"ssh":0.12},
            {"name":"Arabian Sea","lat":20.0,"lon":62.0,"depth":3500,"ssh":0.16},
        ]
    },
    "dover_strait": {
        "name": "Rotterdam → Felixstowe (Dover)",
        "waypoints": [
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
            {"name":"North Sea","lat":52.5,"lon":3.0,"depth":30,"ssh":0.06},
            {"name":"Dover Strait","lat":51.0,"lon":1.5,"depth":35,"ssh":0.07},
            {"name":"Felixstowe","lat":51.96,"lon":1.31,"depth":20,"ssh":0.05},
        ]
    },
    "babelmandeb": {
        "name": "Jeddah → Djibouti (Bab el-Mandeb)",
        "waypoints": [
            {"name":"Jeddah","lat":21.49,"lon":39.19,"depth":30,"ssh":0.06},
            {"name":"Red Sea South","lat":17.0,"lon":41.0,"depth":1500,"ssh":0.11},
            {"name":"Bab el-Mandeb","lat":12.6,"lon":43.3,"depth":180,"ssh":0.09},
            {"name":"Djibouti","lat":11.6,"lon":43.15,"depth":20,"ssh":0.06},
        ]
    },
    "tangermed_rotterdam": {
        "name": "Tanger Med → Rotterdam",
        "waypoints": [
            {"name":"Tanger Med","lat":35.89,"lon":-5.5,"depth":20,"ssh":0.07},
            {"name":"Gibraltar","lat":36.14,"lon":-5.35,"depth":300,"ssh":0.08},
            {"name":"Atlantic Iberia","lat":40.0,"lon":-10.0,"depth":3000,"ssh":0.18},
            {"name":"Biscay","lat":45.0,"lon":-5.0,"depth":2800,"ssh":0.25},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "callao_shanghai": {
        "name": "Callao → Shanghai (Copper)",
        "waypoints": [
            {"name":"Callao","lat":-12.05,"lon":-77.15,"depth":20,"ssh":0.08},
            {"name":"Pacific South","lat":-10.0,"lon":-100.0,"depth":4000,"ssh":0.2},
            {"name":"Mid-Pacific","lat":0.0,"lon":-140.0,"depth":4800,"ssh":0.24},
            {"name":"Philippine Sea","lat":15.0,"lon":135.0,"depth":5000,"ssh":0.22},
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
        ]
    },
    "lagos_rotterdam": {
        "name": "Lagos → Rotterdam",
        "waypoints": [
            {"name":"Lagos","lat":6.44,"lon":3.4,"depth":15,"ssh":0.07},
            {"name":"Gulf of Guinea","lat":2.0,"lon":0.0,"depth":3000,"ssh":0.14},
            {"name":"Equator Atlantic","lat":5.0,"lon":-15.0,"depth":4000,"ssh":0.16},
            {"name":"Canary Islands","lat":28.0,"lon":-18.0,"depth":3500,"ssh":0.15},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "gdansk_rotterdam": {
        "name": "Gdansk → Rotterdam (Baltic)",
        "waypoints": [
            {"name":"Gdansk","lat":54.4,"lon":18.66,"depth":15,"ssh":0.04},
            {"name":"Baltic Sea","lat":55.0,"lon":15.0,"depth":80,"ssh":0.06},
            {"name":"Danish Straits","lat":55.5,"lon":11.0,"depth":30,"ssh":0.05},
            {"name":"North Sea","lat":55.0,"lon":5.0,"depth":40,"ssh":0.08},
            {"name":"Rotterdam","lat":51.95,"lon":4.14,"depth":25,"ssh":0.04},
        ]
    },
    "hochiminh_losangeles": {
        "name": "Ho Chi Minh → Los Angeles",
        "waypoints": [
            {"name":"Ho Chi Minh","lat":10.77,"lon":106.7,"depth":15,"ssh":0.08},
            {"name":"South China Sea","lat":12.0,"lon":112.0,"depth":4000,"ssh":0.15},
            {"name":"Philippine Sea","lat":18.0,"lon":130.0,"depth":5000,"ssh":0.22},
            {"name":"Mid-Pacific","lat":30.0,"lon":-175.0,"depth":5800,"ssh":0.34},
            {"name":"Los Angeles","lat":33.74,"lon":-118.27,"depth":20,"ssh":0.06},
        ]
    },
    "chittagong_singapore": {
        "name": "Chittagong → Singapore",
        "waypoints": [
            {"name":"Chittagong","lat":22.3,"lon":91.8,"depth":10,"ssh":0.07},
            {"name":"Bay of Bengal","lat":15.0,"lon":90.0,"depth":3000,"ssh":0.16},
            {"name":"Andaman Sea","lat":8.0,"lon":96.0,"depth":1500,"ssh":0.12},
            {"name":"Malacca Strait","lat":2.5,"lon":101.0,"depth":40,"ssh":0.1},
            {"name":"Singapore","lat":1.29,"lon":103.85,"depth":25,"ssh":0.08},
        ]
    },
    "manzanillo_shanghai": {
        "name": "Manzanillo → Shanghai",
        "waypoints": [
            {"name":"Manzanillo","lat":19.05,"lon":-104.32,"depth":20,"ssh":0.06},
            {"name":"Pacific East","lat":18.0,"lon":-120.0,"depth":3500,"ssh":0.2},
            {"name":"Mid-Pacific","lat":25.0,"lon":-160.0,"depth":5500,"ssh":0.32},
            {"name":"East China Sea","lat":29.0,"lon":123.0,"depth":100,"ssh":0.1},
            {"name":"Shanghai","lat":31.23,"lon":121.47,"depth":15,"ssh":0.06},
        ]
    },
}

def predict_drag(lat,lon,depth,ssh,swh,speed,draft):
    inp = torch.tensor([[
        (lat+70)/150,(lon+180)/360,depth/6000,
        (ssh+2)/4,swh/20,speed/25,draft/22
    ]]).float()
    with torch.no_grad():
        return model(inp).item()

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>BATIMETRIX V3 | Maritime AI Platform</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js" onload="window.leafletReady=true;if(window.pendingMapInit){initMap();}"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{
  --bg:#030B14;--panel:#071525;--panel2:#0A1C2E;
  --line:#0F2A42;--teal:#00E5B0;--teal2:#00B894;--teal-dim:#00E5B015;
  --amber:#FFC107;--coral:#FF4757;--sky:#3498DB;--purple:#9B59B6;
  --ink:#ECF0F1;--mute:#4A6FA5;--mute2:#2C3E50;
  --grad:linear-gradient(135deg,#00E5B0,#3498DB);
}
*{margin:0;padding:0;box-sizing:border-box;scrollbar-width:thin;scrollbar-color:var(--teal) var(--bg)}
body{background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 120% 80% at 50% -20%,#001830 0%,transparent 60%);pointer-events:none;z-index:0}

/* HEADER */
header{position:sticky;top:0;z-index:1000;background:rgba(3,11,20,.95);backdrop-filter:blur(20px);border-bottom:1px solid var(--line);padding:0 24px;height:64px;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:12px}
.sonar{width:36px;height:36px;border-radius:50%;border:1.5px solid var(--teal);position:relative;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.sonar::before{content:'';position:absolute;inset:-3px;border-radius:50%;border:1.5px solid var(--teal);opacity:0;animation:ping 2.5s ease-out infinite}
.sonar::after{content:'';width:7px;height:7px;border-radius:50%;background:var(--teal);box-shadow:0 0 10px var(--teal)}
@keyframes ping{0%{transform:scale(1);opacity:.7}100%{transform:scale(2.4);opacity:0}}
.logo{font-family:'Space Grotesk';font-size:20px;font-weight:700;letter-spacing:3px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tagline{font-size:9px;color:var(--mute);letter-spacing:3px;text-transform:uppercase;margin-top:1px}
.hdr-right{display:flex;align-items:center;gap:12px}
.clock{font-family:'JetBrains Mono';font-size:12px;color:var(--teal);background:var(--panel);border:1px solid var(--line);padding:7px 12px;border-radius:6px;letter-spacing:1.5px;white-space:nowrap}
.lang-sel{background:var(--panel);border:1px solid var(--line);color:var(--ink);padding:7px 10px;border-radius:6px;font-size:12px;cursor:pointer;outline:none;font-family:'Inter'}
.lang-sel:focus{border-color:var(--teal)}
.v-badge{background:var(--teal-dim);border:1px solid var(--teal);color:var(--teal);padding:4px 10px;border-radius:20px;font-size:10px;font-family:'JetBrains Mono';letter-spacing:1px}

/* WORLD CLOCK TICKER */
.wct-wrap{background:#050F1C;border-bottom:1px solid var(--line);overflow:hidden;white-space:nowrap;padding:5px 0;position:relative}
.wct-track{display:inline-block;white-space:nowrap;animation:wctScroll 90s linear infinite}
.wct-wrap:hover .wct-track{animation-play-state:paused}
.wct-item{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--mute);margin:0 14px;letter-spacing:.5px}
.wct-item b{color:var(--teal);font-weight:600}
.wct-item.wct-hl b{color:#FFC107}
.wct-item .wct-off{color:#2C3E50;font-size:9px;margin-left:4px}
@keyframes wctScroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
/* PRINT / PDF STYLES */
@media print {
  body{background:#fff!important;color:#000!important}
  .sidebar,.tabs,.btn-run,.wct-wrap,.sat-bar,footer,
  .loading,#empty_state,.tab{display:none!important}
  .layout{display:block!important}
  .content{width:100%!important;padding:0!important}
  .card{border:1px solid #ccc!important;background:#fff!important;break-inside:avoid}
  .kpi{background:#f5f5f5!important;border:1px solid #ccc!important}
  .kpi-val{color:#000!important}
  .chart-card{break-inside:avoid}
  #print_header{display:block!important}
  #btn_pdf{display:none}
  .results{display:block!important}
  #fleet_tab,#analysis_tab,#cii_tab,#table_tab,#map_tab{display:block!important}
}

/* SAT BAR */
.satbar{display:flex;gap:8px;padding:12px 24px;border-bottom:1px solid var(--line);background:var(--panel);overflow-x:auto}
.sat{display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--bg);border:1px solid var(--line);border-radius:8px;white-space:nowrap;flex-shrink:0}
.sat-dot{width:7px;height:7px;border-radius:50%;background:var(--teal);box-shadow:0 0 6px var(--teal);animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.sat-name{font-family:'JetBrains Mono';font-size:11px;font-weight:600;color:var(--teal)}
.sat-desc{font-size:10px;color:var(--mute)}

/* MAIN LAYOUT */
.main{display:grid;grid-template-columns:340px 1fr;min-height:calc(100vh - 100px);position:relative;z-index:1}
@media(max-width:900px){.main{grid-template-columns:1fr}}

/* SIDEBAR */
.sidebar{background:var(--panel);border-right:1px solid var(--line);padding:20px;overflow-y:auto;max-height:calc(100vh - 100px);position:sticky;top:64px}
.section-title{font-family:'Space Grotesk';font-size:11px;font-weight:600;color:var(--teal);letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.section-title::before{content:'';width:3px;height:14px;background:var(--teal);border-radius:2px}
.fg{margin-bottom:14px}
label{font-size:10px;color:var(--mute);display:block;margin-bottom:6px;letter-spacing:1.5px;text-transform:uppercase;font-weight:500}
select,input[type=number]{width:100%;background:var(--bg);border:1px solid var(--line);color:var(--ink);padding:10px 12px;border-radius:8px;font-size:13px;font-family:'Inter';outline:none;transition:all .2s}
select:focus,input:focus{border-color:var(--teal);box-shadow:0 0 0 2px var(--teal-dim)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btn-run{width:100%;padding:13px;margin-top:6px;background:var(--grad);border:none;border-radius:9px;color:#001a0f;font-family:'Space Grotesk';font-size:13px;font-weight:700;letter-spacing:2px;cursor:pointer;text-transform:uppercase;transition:all .2s;position:relative;overflow:hidden}
.btn-run::before{content:'';position:absolute;inset:0;background:rgba(255,255,255,.1);opacity:0;transition:opacity .2s}
.btn-run:hover::before{opacity:1}
.btn-run:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(0,229,176,.3)}
.divider{height:1px;background:var(--line);margin:18px 0}

/* RESULTS PANEL */
.content{padding:20px;overflow-y:auto}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
@media(max-width:1300px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s}
.kpi:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.3)}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--kc)}
.kpi.drag{--kc:var(--coral)}.kpi.fuel{--kc:var(--teal)}.kpi.cash{--kc:var(--amber)}.kpi.co2{--kc:var(--sky)}
.kpi-icon{font-size:20px;margin-bottom:8px}
.kpi-val{font-family:'JetBrains Mono';font-size:26px;font-weight:700;color:var(--kc);letter-spacing:-1px;line-height:1}
.kpi-lbl{font-size:10px;color:var(--mute);margin-top:6px;letter-spacing:1.5px;text-transform:uppercase}
.kpi-sub{font-size:10px;color:var(--mute);margin-top:4px}

/* TABS */
.tabs{display:flex;gap:4px;margin-bottom:16px;background:var(--panel);padding:4px;border-radius:10px;border:1px solid var(--line)}
.tab{flex:1;padding:9px;text-align:center;font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;cursor:pointer;border-radius:7px;transition:all .2s;color:var(--mute);font-family:'Space Grotesk'}
.tab.active{background:var(--teal);color:#001a0f}

/* MAP */
#map{height:420px;border-radius:12px;border:1px solid var(--line);overflow:hidden}
.leaflet-container{background:#071525 !important}

/* CHARTS */
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
@media(max-width:1100px){.charts-grid{grid-template-columns:1fr}}
.chart-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
.chart-title{font-family:'Space Grotesk';font-size:11px;font-weight:600;color:var(--teal);letter-spacing:2px;text-transform:uppercase;margin-bottom:14px}
.chart-wrap{position:relative;height:200px}

/* CII */
.cii-grid{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;margin:16px 0}
.cii-card{background:var(--bg);border:2px solid var(--cc);border-radius:12px;padding:20px;text-align:center;transition:all .2s}
.cii-label{font-size:9px;color:var(--mute);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}
.cii-grade{font-family:'Space Grotesk';font-size:52px;font-weight:700;color:var(--cc);line-height:1}
.cii-val{font-family:'JetBrains Mono';font-size:10px;color:var(--mute);margin-top:6px}
.cii-arrow{font-size:24px;color:var(--teal);animation:nudge 1.8s ease-in-out infinite;text-align:center}
@keyframes nudge{0%,100%{transform:translateX(0)}50%{transform:translateX(5px)}}
.cA{--cc:#22C55E}.cB{--cc:#84CC16}.cC{--cc:var(--amber)}.cD{--cc:#FB923C}.cE{--cc:var(--coral)}
.cii-msg{text-align:center;font-size:12px;padding:10px;border-radius:8px;margin-top:8px}

/* TABLE */
.route-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:16px}
.route-table th{background:var(--bg);color:var(--mute);padding:10px 12px;text-align:left;font-size:9px;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid var(--line)}
.route-table td{padding:11px 12px;border-bottom:1px solid var(--line);font-family:'JetBrains Mono';font-size:11px}
.route-table td:first-child{font-family:'Inter';font-size:12px;font-weight:500}
.route-table tr:hover td{background:#0A1826}
.wp-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}
.wg{background:var(--teal);box-shadow:0 0 5px var(--teal)}
.wy{background:var(--amber);box-shadow:0 0 5px var(--amber)}
.wr{background:var(--coral);box-shadow:0 0 5px var(--coral)}
.badge{padding:3px 8px;border-radius:4px;font-size:9px;font-weight:600;letter-spacing:1px;font-family:'Space Grotesk'}
.badge-green{background:#22C55E20;color:#22C55E;border:1px solid #22C55E40}
.badge-yellow{background:#FFC10720;color:#FFC107;border:1px solid #FFC10740}
.badge-red{background:#FF475720;color:#FF4757;border:1px solid #FF475740}

/* LOADING */
.loading{display:none;text-align:center;padding:80px 20px}
.radar-wrap{width:72px;height:72px;margin:0 auto 20px;border-radius:50%;border:2px solid var(--line);position:relative;overflow:hidden}
.radar-wrap::before{content:'';position:absolute;inset:0;background:conic-gradient(from 0deg,transparent 0deg,var(--teal) 40deg,transparent 90deg);animation:sweep 1.4s linear infinite}
@keyframes sweep{to{transform:rotate(360deg)}}
.load-txt{font-family:'JetBrains Mono';font-size:12px;color:var(--teal);letter-spacing:2px;animation:pulse 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.load-sub{font-size:11px;color:var(--mute);margin-top:8px}

.results{display:none;animation:fadeUp .4s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* PANEL CARD */
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.card-title{font-family:'Space Grotesk';font-size:11px;font-weight:600;color:var(--teal);letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.card-title::before{content:'';width:3px;height:14px;background:var(--teal);border-radius:2px}

/* PROGRESS */
.prog-row{margin-bottom:14px}
.prog-head{display:flex;justify-content:space-between;font-size:11px;color:var(--mute);margin-bottom:6px}
.prog-head b{font-family:'JetBrains Mono';color:var(--ink)}
.prog-track{height:6px;background:var(--bg);border-radius:3px;overflow:hidden;border:1px solid var(--line)}
.prog-fill{height:100%;border-radius:3px;transition:width 1.2s cubic-bezier(.22,1,.36,1)}

/* STATS ROW */
.stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.stat-item{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;text-align:center}
.stat-val{font-family:'JetBrains Mono';font-size:18px;font-weight:700;color:var(--teal)}
.stat-lbl{font-size:9px;color:var(--mute);margin-top:4px;letter-spacing:1px;text-transform:uppercase}

/* FOOTER */
footer{background:var(--panel);border-top:1px solid var(--line);padding:16px 24px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;font-family:'JetBrains Mono';font-size:10px;color:var(--mute);position:relative;z-index:1}
footer a{color:var(--teal);text-decoration:none}
.footer-dots{display:flex;gap:6px;align-items:center}
.footer-dot{width:6px;height:6px;border-radius:50%;background:var(--teal);animation:blink 2s infinite}
.footer-dot:nth-child(2){animation-delay:.3s}
.footer-dot:nth-child(3){animation-delay:.6s}

/* EMPTY STATE */
.empty{text-align:center;padding:60px 20px;color:var(--mute)}
.empty-icon{font-size:48px;margin-bottom:16px;opacity:.3}
.empty-txt{font-size:13px;letter-spacing:1px}
</style>


</head>
<body>

<header>
  <div class="brand">
    <div class="sonar"></div>
    <div>
      <div class="logo">BATIMETRIX</div>
      <div class="tagline" data-i18n="tagline">Proactive Hydrodynamic Intelligence</div>
    </div>
  </div>
  <div class="hdr-right">
    <div class="clock" id="clock">--:--:-- UTC</div>
    <select class="lang-sel" id="lang_sel" onchange="setLang(this.value)">
      <option value="en">🇬🇧 EN</option>
      <option value="tr">🇹🇷 TR</option>
      <option value="el">🇬🇷 EL</option>
      <option value="zh">🇨🇳 ZH</option>
      <option value="ru">🇷🇺 RU</option>
      <option value="es">🇪🇸 ES</option>
      <option value="fr">🇫🇷 FR</option>
    </select>
    <div class="v-badge">V3 PRO</div>
  </div>
</header>
<div class="wct-wrap"><div class="wct-track" id="wct_track"></div></div>

<div class="satbar">
  <div class="sat"><div class="sat-dot"></div><div class="sat-name">SWOT</div><div class="sat-desc" data-i18n="sat_swot">Sea Surface Height</div></div>
  <div class="sat"><div class="sat-dot"></div><div class="sat-name">GPM</div><div class="sat-desc" data-i18n="sat_gpm">Storm Prediction</div></div>
  <div class="sat"><div class="sat-dot"></div><div class="sat-name">MODIS</div><div class="sat-desc" data-i18n="sat_modis">SST Viscosity</div></div>
  <div class="sat"><div class="sat-dot"></div><div class="sat-name">GEBCO</div><div class="sat-desc" data-i18n="sat_gebco">Bathymetry 2026</div></div>
  <div class="sat"><div class="sat-dot"></div><div class="sat-name">PINN</div><div class="sat-desc">1,657,025 params</div></div>
  <div class="sat"><div class="sat-dot"></div><div class="sat-name">IMO CII</div><div class="sat-desc">MEPC.354(78)</div></div>
</div>

<div class="main">
  <div class="sidebar">
    <div class="section-title" data-i18n="mission_params">Mission Parameters</div>

    <div class="fg">
      <label data-i18n="vessel_type">Vessel Type</label>
      <select id="vessel">
        <option value="Black Sea Cargo">Black Sea Cargo</option>
        <option value="Handy Bulk">Handy Bulk Carrier</option>
        <option value="Panamax Container">Panamax Container</option>
        <option value="Capesize Bulk">Capesize Bulk</option>
        <option value="LNG Carrier">LNG Carrier</option>
        <option value="VLCC Tanker">VLCC Tanker</option>
        <option value="Aframax Tanker">Aframax Tanker</option>
        <option value="Suezmax Tanker">Suezmax Tanker</option>
        <option value="MR Product Tanker">MR Product Tanker</option>
        <option value="ULCC Tanker">ULCC Tanker</option>
        <option value="Supramax Bulk">Supramax Bulk</option>
        <option value="Panamax Bulk">Panamax Bulk</option>
        <option value="VLOC Valemax">VLOC Valemax</option>
        <option value="ULCV Container">ULCV Container</option>
        <option value="Feeder Container">Feeder Container</option>
      </select>
    </div>

    <div class="fg">
      <label data-i18n="route_lbl">Route</label>
      <select id="route" onchange="previewRoute()">
        <option value="istanbul_trabzon">Istanbul → Trabzon</option>
        <option value="istanbul_novorossiysk">Istanbul → Novorossiysk</option>
        <option value="odessa_istanbul">Odessa → Istanbul</option>
        <option value="batumi_constanta">Batumi → Constanta</option>
        <option value="karadeniz_sakin">Black Sea — Calm</option>
        <option value="atlantik">North Atlantic — Storm</option>
        <option value="shanghai_rotterdam">Shanghai → Rotterdam</option>
        <option value="shanghai_losangeles">Shanghai → Los Angeles</option>
        <option value="singapore_shanghai">Singapore → Shanghai</option>
        <option value="rotterdam_newyork">Rotterdam → New York</option>
        <option value="santos_rotterdam">Santos → Rotterdam</option>
        <option value="dubai_singapore">Dubai → Singapore</option>
        <option value="mumbai_suez">Mumbai → Suez</option>
        <option value="durban_singapore">Durban → Singapore</option>
        <option value="panama_losangeles">Panama → Los Angeles</option>
        <option value="tokyo_losangeles">Tokyo → Los Angeles</option>
        <option value="hamburg_newyork">Hamburg → New York</option>
        <option value="melbourne_shanghai">Melbourne → Shanghai</option>
        <option value="jeddah_rotterdam">Jeddah → Rotterdam</option>
        <option value="busan_losangeles">Busan → Los Angeles</option>
        <option value="gibraltar_piraeus">Gibraltar → Piraeus</option>
        <option value="hongkong_singapore">Hong Kong → Singapore</option>
        <option value="newyork_santos">New York → Santos</option>
        <option value="capetown_singapore">Cape Town → Singapore</option>
        <option value="yokohama_singapore">Yokohama → Singapore</option>
        <option value="antwerp_newyork">Antwerp → New York</option>
        <option value="valencia_suez">Valencia → Suez</option>
        <option value="qingdao_rotterdam">Qingdao → Rotterdam</option>
        <option value="colombo_singapore">Colombo → Singapore</option>
        <option value="houston_rotterdam">Houston → Rotterdam</option>
        <option value="seattle_tokyo">Seattle → Tokyo</option>
        <option value="piraeus_alexandria">Piraeus → Alexandria</option>
        <option value="vancouver_shanghai">Vancouver → Shanghai</option>
        <option value="istanbul_piraeus">Istanbul → Piraeus</option>
        <option value="kobe_singapore">Kobe → Singapore</option>
        <option value="algeciras_newyork">Algeciras → New York</option>
        <option value="dalian_singapore">Dalian → Singapore</option>
        <option value="felixstowe_singapore">Felixstowe → Singapore</option>
        <option value="longbeach_yokohama">Long Beach → Yokohama</option>
        <option value="genoa_alexandria">Genoa → Alexandria</option>
        <option value="porthedland_qingdao">Port Hedland → Qingdao (Iron Ore)</option>
        <option value="pontamadeira_qingdao">Ponta da Madeira → Qingdao (Vale Ore)</option>
        <option value="newcastle_tokyo">Newcastle → Tokyo (Coal)</option>
        <option value="neworleans_shanghai">New Orleans → Shanghai (Grain)</option>
        <option value="rastanura_ningbo">Ras Tanura → Ningbo (VLCC Crude)</option>
        <option value="raslaffan_tokyo">Ras Laffan → Tokyo (LNG)</option>
        <option value="sabinepass_rotterdam">Sabine Pass → Rotterdam (US LNG)</option>
        <option value="bonny_rotterdam">Bonny → Rotterdam (Crude)</option>
        <option value="singapore_rotterdam_cape">Singapore → Rotterdam (Cape Route)</option>
        <option value="murmansk_shanghai">Murmansk → Shanghai (Arctic NSR)</option>
        <option value="hormuz_transit">Kuwait → Arabian Sea (Hormuz)</option>
        <option value="dover_strait">Rotterdam → Felixstowe (Dover)</option>
        <option value="babelmandeb">Jeddah → Djibouti (Bab el-Mandeb)</option>
        <option value="tangermed_rotterdam">Tanger Med → Rotterdam</option>
        <option value="callao_shanghai">Callao → Shanghai (Copper)</option>
        <option value="lagos_rotterdam">Lagos → Rotterdam</option>
        <option value="gdansk_rotterdam">Gdansk → Rotterdam (Baltic)</option>
        <option value="hochiminh_losangeles">Ho Chi Minh → Los Angeles</option>
        <option value="chittagong_singapore">Chittagong → Singapore</option>
        <option value="manzanillo_shanghai">Manzanillo → Shanghai</option>
      </select>
    </div>

    <div class="row2">
      <div class="fg"><label data-i18n="speed_lbl">Speed (kn)</label><input type="number" id="speed" value="12" min="5" max="25" step="0.5"></div>
      <div class="fg"><label data-i18n="draft_lbl">Draft (m)</label><input type="number" id="draft" value="8.5" min="3" max="22" step="0.5"></div>
    </div>
    <div class="row2">
      <div class="fg"><label data-i18n="wave_lbl">Wave Ht (m)</label><input type="number" id="swh" value="1.2" min="0" max="12" step="0.1"></div>
      <div class="fg"><label data-i18n="temp_lbl">Sea Temp °C</label><input type="number" id="sst" value="22" min="0" max="32"></div>
    </div>
    <div class="fg"><label data-i18n="voyage_lbl">Annual Voyage Days</label><input type="number" id="days" value="280" min="50" max="365"></div>

    <button class="btn-run" onclick="runAnalysis()" data-i18n="run_btn">⚡ Run Analysis</button>
    <button id="btn_pdf" onclick="downloadPDF()"
      style="width:100%;padding:10px;margin-top:8px;background:transparent;display:none;border:1px solid #00E5B0;border-radius:9px;color:#00E5B0;font-size:12px;font-weight:600;letter-spacing:1px;cursor:pointer;text-transform:uppercase">
      Download PDF Report
    </button>

    <div class="divider"></div>

    <div class="section-title">Live Stats</div>
    <div class="stats-row">
      <div class="stat-item"><div class="stat-val" id="stat_drag">—</div><div class="stat-lbl">Drag</div></div>
      <div class="stat-item"><div class="stat-val" id="stat_sav">—</div><div class="stat-lbl">Savings</div></div>
      <div class="stat-item"><div class="stat-val" id="stat_cii">—</div><div class="stat-lbl">CII</div></div>
    </div>
  </div>

  <div class="content">
    <div class="loading" id="loading">
      <div class="radar-wrap"></div>
      <div class="load-txt" data-i18n="analyzing">RUNNING PINN INFERENCE...</div>
      <div class="load-sub">SWOT + GPM + MODIS + GEBCO → Neural Network</div>
    </div>

    <div id="print_header" style="display:none;padding:20px 0;border-bottom:2px solid #000;margin-bottom:20px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:24px;font-weight:900;letter-spacing:2px">BATIMETRIX</div>
      <div style="font-size:11px;color:#666">Proactive Hydrodynamic Drag Intelligence Report</div>
    </div>
    <div style="text-align:right;font-size:11px;color:#666">
      <div>Generated: <span id="print_date"></span></div>
      <div>NASA SWOT + GPM + MODIS + GEBCO 2026</div>
      <div>PINN Model — 1,657,025 parameters</div>
      <div>IMO CII MEPC.354(78)</div>
    </div>
  </div>
</div>
<div id="print_header" style="display:none;padding:20px 0;border-bottom:2px solid #000;margin-bottom:20px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:24px;font-weight:900;letter-spacing:2px">BATIMETRIX</div>
      <div style="font-size:11px;color:#666">Proactive Hydrodynamic Drag Intelligence Report</div>
    </div>
    <div style="text-align:right;font-size:11px;color:#666">
      <div>Generated: <span id="print_date"></span></div>
      <div>NASA SWOT + GPM + MODIS + GEBCO 2026</div>
      <div>PINN Model — 1,657,025 parameters</div>
      <div>IMO CII MEPC.354(78)</div>
    </div>
  </div>
</div>
<div id="empty_state">
      <div class="empty">
        <div class="empty-icon">🛰️</div>
        <div class="empty-txt" data-i18n="empty_txt">Select vessel, route and run analysis</div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="card-title" data-i18n="map_title">Route Map</div>
        <div id="map"></div>
      </div>
    </div>

    <div class="results" id="results">
      <div class="kpis">
        <div class="kpi drag">
          <div class="kpi-icon">🌊</div>
          <div class="kpi-val" id="kpi_drag">—</div>
          <div class="kpi-lbl" data-i18n="drag_score">Drag Score</div>
          <div class="kpi-sub">Hydrodynamic resistance index</div>
        </div>
        <div class="kpi fuel">
          <div class="kpi-icon">⚡</div>
          <div class="kpi-val" id="kpi_fuel">—</div>
          <div class="kpi-lbl" data-i18n="fuel_savings">Fuel Savings</div>
          <div class="kpi-sub">vs. unoptimized voyage</div>
        </div>
        <div class="kpi cash">
          <div class="kpi-icon">💰</div>
          <div class="kpi-val" id="kpi_cash">—</div>
          <div class="kpi-lbl" data-i18n="annual_savings">Annual Savings</div>
          <div class="kpi-sub">USD per year</div>
        </div>
        <div class="kpi co2">
          <div class="kpi-icon">🌱</div>
          <div class="kpi-val" id="kpi_co2">—</div>
          <div class="kpi-lbl" data-i18n="co2_cut">CO2 Cut / yr</div>
          <div class="kpi-sub">metric tons annually</div>
        </div>
      </div>

      <div class="tabs">
        <div class="tab active" onclick="showTab('map_tab',this)" data-i18n="tab_map">🗺️ Route Map</div>
        <div class="tab" onclick="showTab('analysis_tab',this)" data-i18n="tab_analysis">📊 Analysis</div>
        <div class="tab" onclick="showTab('cii_tab',this)" data-i18n="tab_cii">⚖️ CII Rating</div>
        <div class="tab" onclick="showTab('table_tab',this)" data-i18n="tab_table">📋 Telemetry</div>
        <div class="tab" onclick="showTab('fleet_tab',this)">&#128674; Fleet</div>
        <div class="tab" onclick="showTab('ssh_tab',this)">&#127754; Ocean Intel</div>
        <div class="tab" onclick="showTab('globe_tab',this)">&#127758; 3D Globe</div>
        <div class="tab" onclick="showTab('compare_tab',this)">&#9878; Compare</div>
        <div class="tab" onclick="showTab('ets_tab',this)">&#127758; EU ETS</div>
        <div class="tab" onclick="showTab('speed_tab',this)">&#9889; Speed AI</div>
      </div>

      <!-- MAP TAB -->
      <div id="map_tab">
        <div class="card">
          <div class="card-title" data-i18n="map_title">Route Map — Live Drag Overlay</div>
          <div id="map"></div>
        </div>
        <div class="card">
          <div class="card-title" data-i18n="drag_profile">Drag Profile</div>
          <div class="prog-row">
            <div class="prog-head"><span data-i18n="drag_score">Drag Score</span><b id="prog_drag">—</b></div>
            <div class="prog-track"><div class="prog-fill" id="fill_drag" style="width:0%;background:linear-gradient(90deg,var(--teal),var(--coral))"></div></div>
          </div>
          <div class="prog-row">
            <div class="prog-head"><span data-i18n="fuel_eff">Fuel Efficiency</span><b id="prog_eff">—</b></div>
            <div class="prog-track"><div class="prog-fill" id="fill_eff" style="width:0%;background:linear-gradient(90deg,var(--coral),var(--teal))"></div></div>
          </div>
        </div>
      </div>

      <!-- ANALYSIS TAB -->
      <div id="analysis_tab" style="display:none">

        <!-- FUEL COST CALCULATOR -->
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#9981; Real-Time Fuel Cost Calculator</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">BUNKER PRICE ($/ton)</label>
              <input type="number" id="fuel_price" value="650" min="200" max="1500" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:13px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">VOYAGE DAYS</label>
              <input type="number" id="fuel_days" value="20" min="1" max="120" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:13px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">FUEL CONSUMPTION (t/day)</label>
              <input type="number" id="fuel_consumption" value="80" min="5" max="300" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:13px">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">DRAG SAVINGS (%)</label>
              <input type="number" id="fuel_savings_pct" value="10" min="1" max="20" step="0.1" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:13px">
              <div style="font-size:9px;color:var(--mute);margin-top:2px">Auto-filled from analysis</div>
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">VOYAGES PER YEAR</label>
              <input type="number" id="fuel_voyages" value="12" min="1" max="50" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:13px">
            </div>
          </div>
          <button onclick="calcFuel()" style="width:100%;padding:10px;background:linear-gradient(135deg,#F39C12,#E67E22);border:none;border-radius:8px;color:white;font-size:13px;font-weight:700;cursor:pointer;margin-bottom:12px">
            &#9981; CALCULATE FUEL SAVINGS
          </button>
          <div id="fuel_results" style="display:none">
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px">
              <div style="background:var(--bg);border-radius:10px;padding:12px;text-align:center;border:1px solid #E74C3C33">
                <div style="font-size:22px;font-weight:900;color:#E74C3C" id="fuel_cost_without">—</div>
                <div style="font-size:9px;color:var(--mute)">WITHOUT BATIMETRIX</div>
              </div>
              <div style="background:var(--bg);border-radius:10px;padding:12px;text-align:center;border:1px solid #27AE6033">
                <div style="font-size:22px;font-weight:900;color:#27AE60" id="fuel_cost_with">—</div>
                <div style="font-size:9px;color:var(--mute)">WITH BATIMETRIX</div>
              </div>
              <div style="background:var(--bg);border-radius:10px;padding:12px;text-align:center;border:1px solid #F39C1233">
                <div style="font-size:22px;font-weight:900;color:#F39C12" id="fuel_saving_voyage">—</div>
                <div style="font-size:9px;color:var(--mute)">SAVED/VOYAGE</div>
              </div>
              <div style="background:var(--bg);border-radius:10px;padding:12px;text-align:center;border:1px solid var(--teal)33">
                <div style="font-size:22px;font-weight:900;color:var(--teal)" id="fuel_saving_year">—</div>
                <div style="font-size:9px;color:var(--mute)">SAVED/YEAR</div>
              </div>
            </div>
            <div id="fuel_breakdown" style="background:var(--bg);border-radius:10px;padding:14px;font-size:12px;line-height:2"></div>
            <div id="fuel_co2" style="background:#27AE6011;border:1px solid #27AE6033;border-radius:10px;padding:12px;margin-top:10px;font-size:12px;color:#27AE60;text-align:center"></div>
          </div>
        </div>


        <div class="charts-grid">
          <div class="chart-card">
            <div class="chart-title" data-i18n="drag_per_wp">Drag per Waypoint</div>
            <div class="chart-wrap"><canvas id="chart_drag"></canvas></div>
          </div>
          <div class="chart-card">
            <div class="chart-title" data-i18n="savings_per_wp">Savings per Waypoint</div>
            <div class="chart-wrap"><canvas id="chart_sav"></canvas></div>
          </div>
          <div class="chart-card">
            <div class="chart-title" data-i18n="depth_profile">Depth Profile</div>
            <div class="chart-wrap"><canvas id="chart_depth"></canvas></div>
          </div>
          <div class="chart-card">
            <div class="chart-title" data-i18n="drag_vs_depth">Drag vs Depth</div>
            <div class="chart-wrap"><canvas id="chart_scatter"></canvas></div>
          </div>
        </div>
      </div>

      <!-- CII TAB -->
      <div id="cii_tab" style="display:none">
        <div class="card">
          <div class="card-title" data-i18n="cii_title">IMO CII Rating 2026 — MEPC.354(78)</div>
          <div class="cii-grid">
            <div>
              <div class="cii-label" data-i18n="without_b">Without Batimetrix</div>
              <div class="cii-card cE" id="cii_before_card">
                <div class="cii-grade" id="cii_before">—</div>
                <div class="cii-val" id="cii_before_val">—</div>
              </div>
            </div>
            <div class="cii-arrow">→</div>
            <div>
              <div class="cii-label" data-i18n="with_b">With Batimetrix</div>
              <div class="cii-card cA" id="cii_after_card">
                <div class="cii-grade" id="cii_after">—</div>
                <div class="cii-val" id="cii_after_val">—</div>
              </div>
            </div>
          </div>
          <div class="cii-msg" id="cii_msg"></div>
        </div>
        <div class="charts-grid">
          <div class="chart-card">
            <div class="chart-title">CII Comparison</div>
            <div class="chart-wrap"><canvas id="chart_cii"></canvas></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Annual CO2 Reduction</div>
            <div class="chart-wrap"><canvas id="chart_co2"></canvas></div>
          </div>
        </div>
      </div>

      <!-- TABLE TAB -->
      <div id="table_tab" style="display:none">
        <div class="card">
          <div class="card-title" data-i18n="telemetry">Route Telemetry</div>
          <table class="route-table">
            <thead>
              <tr>
                <th data-i18n="th_waypoint">Waypoint</th>
                <th data-i18n="th_depth">Depth</th>
                <th>SSH (m)</th><th>WIND (m/s)</th><th>CHLOROPHYLL</th>
                <th>SWH (m)</th>
                <th>Drag</th>
                <th data-i18n="th_savings">Savings</th>
                <th data-i18n="th_status">Status</th>
              </tr>
            </thead>
            <tbody id="table_body"></tbody>
          </table>
        </div>
      </div>
    </div>

      <!-- FLEET DASHBOARD TAB -->
      <div id="fleet_tab" style="display:none">
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#128674; Fleet Dashboard — Live Drag Intelligence</div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px">
            <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid var(--line)">
              <div style="font-size:22px;font-weight:900;color:var(--teal)" id="fleet_vessels">5</div>
              <div style="font-size:10px;color:var(--mute);letter-spacing:1px">VESSELS</div>
            </div>
            <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid var(--line)">
              <div style="font-size:22px;font-weight:900;color:var(--amber)" id="fleet_savings">$4.2M</div>
              <div style="font-size:10px;color:var(--mute);letter-spacing:1px">ANNUAL SAVINGS</div>
            </div>
            <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid var(--line)">
              <div style="font-size:22px;font-weight:900;color:var(--teal)" id="fleet_co2">18,420t</div>
              <div style="font-size:10px;color:var(--mute);letter-spacing:1px">CO2 SAVED/YR</div>
            </div>
            <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid var(--line)">
              <div style="font-size:22px;font-weight:900;color:#27AE60" id="fleet_cii">3 × A</div>
              <div style="font-size:10px;color:var(--mute);letter-spacing:1px">CII IMPROVED</div>
            </div>
          </div>
          <table class="route-table">
            <thead>
              <tr>
                <th>VESSEL</th>
                <th>TYPE</th>
                <th>ROUTE</th>
                <th>DRAG</th>
                <th>SAVINGS</th>
                <th>ANNUAL $</th>
                <th>CII</th>
                <th>STATUS</th><th>WIND</th><th>CHLOROPHYLL</th><th>SALINITY</th><th>ICE (m)</th>
              </tr>
            </thead>
            <tbody id="fleet_tbody">
            </tbody>
          </table>
        </div>
        <div style="background:#27AE6011;border:1px solid #27AE6033;border-radius:8px;padding:10px 14px;margin-top:12px;font-size:11px;color:#7F8C8D">
          &#127807; <span style="color:#27AE60;font-weight:700">PACE OCI</span> chlorophyll-a data updated <span style="color:white">2026-09-16</span> — yesterday's ocean color. High chlorophyll = phytoplankton bloom = increased water viscosity = higher drag. Black Sea: 2.81 mg/m³ (elevated).
        </div>
        <div style="text-align:center;padding:16px;font-size:11px;color:var(--mute)">
          &#128161; Run analysis on individual vessels to populate fleet data. 
          Pilot fleet data shown for demonstration.
        </div>
      </div>


      <!-- SSH INTELLIGENCE TAB -->
      <div id="ssh_tab" style="display:none">
        <div style="background:#0A1628;border:1px solid #1B4F72;border-radius:10px;padding:12px 16px;margin-bottom:12px;display:flex;gap:20px;flex-wrap:wrap">
          <div style="font-size:11px;color:#7F8C8D">
            <span style="color:var(--teal);font-weight:700">SWOT</span> (NASA/CNES)
            <span style="color:#2C3E50"> | </span>
            21-day cycle
            <span style="color:#2C3E50"> | </span>
            Last: <span style="color:white">2025-05-03</span>
          </div>
          <div style="font-size:11px;color:#7F8C8D">
            <span style="color:#3498DB;font-weight:700">Sentinel-6</span> (NASA/ESA)
            <span style="color:#2C3E50"> | </span>
            10-day cycle
            <span style="color:#2C3E50"> | </span>
            Last: <span style="color:white">2026-01-16</span>
          </div>
          <div style="font-size:11px;color:#7F8C8D">
            <span style="color:#F39C12;font-weight:700">GEBCO 2026</span>
            <span style="color:#2C3E50"> | </span>
            Bathymetry
            <span style="color:#2C3E50"> | </span>
            Released: <span style="color:white">April 2026</span>
          </div>
          <div style="font-size:11px;color:#7F8C8D">
            <span style="color:#27AE60;font-weight:700">PACE OCI</span> (NASA)
            <span style="color:#2C3E50"> | </span>
            Ocean Color NRT
            <span style="color:#2C3E50"> | </span>
            Last: <span style="color:#1ABC9C;font-weight:700">2026-09-16</span> &#128308;
          </div>
          <div style="font-size:11px;color:#7F8C8D">
            <span style="color:#9B59B6;font-weight:700">CYGNSS</span> (NASA)
            <span style="color:#2C3E50"> | </span>
            8 satellites
            <span style="color:#2C3E50"> | </span>
            Last: <span style="color:white">2025-10-31</span>
          </div>
          <div style="font-size:11px;color:#7F8C8D">
            <span style="color:#3498DB;font-weight:700">ICESat-2</span> (NASA)
            <span style="color:#2C3E50"> | </span>
            Arctic Ocean Height
            <span style="color:#2C3E50"> | </span>
            Last: <span style="color:white">2026-05-18</span>
          </div>
          <div style="font-size:11px;color:#7F8C8D">
            <span style="color:#E74C3C;font-weight:700">Combined Coverage</span>
            <span style="color:#2C3E50"> | </span>
            Dual altimeter SSH fusion
          </div>
        </div>
        <div id="ssh_banner" style="background:#E74C3C11;border:1px solid #E74C3C;border-radius:12px;padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px">
          <div style="font-size:20px">&#128680;</div>
          <div>
            <div style="color:#E74C3C;font-weight:700;font-size:13px;letter-spacing:1px">ACTIVE SSH ANOMALIES DETECTED</div>
            <div style="color:#7F8C8D;font-size:11px;margin-top:2px">NASA SWOT baseline deviation — <span id="ssh_count">0</span> critical zones</div>
          </div>
          <div style="margin-left:auto;text-align:right">
            <div style="color:#E74C3C;font-size:20px;font-weight:900" id="ssh_max">+0.000m</div>
            <div style="color:#7F8C8D;font-size:10px">MAX DEVIATION</div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #E74C3C33">
            <div style="font-size:20px;font-weight:900;color:#E74C3C" id="ssh_crit">0</div>
            <div style="font-size:10px;color:var(--mute)">CRITICAL</div>
          </div>
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #F39C1233">
            <div style="font-size:20px;font-weight:900;color:#F39C12" id="ssh_warn">0</div>
            <div style="font-size:10px;color:var(--mute)">WARNING</div>
          </div>
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #27AE6033">
            <div style="font-size:20px;font-weight:900;color:#27AE60" id="ssh_norm">0</div>
            <div style="font-size:10px;color:var(--mute)">NORMAL</div>
          </div>
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid var(--teal)33">
            <div style="font-size:20px;font-weight:900;color:var(--teal)" id="ssh_total">16</div>
            <div style="font-size:10px;color:var(--mute)">MONITORED</div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">&#127758; SSH Anomaly Intelligence — 16 Global Maritime Zones (SWOT + Sentinel-6)</div>
          <table class="route-table">
            <thead>
              <tr>
                <th>ZONE</th><th>REGION</th><th>SSH (m)</th>
                <th>BASELINE</th><th>DEVIATION</th>
                <th>DRAG IMPACT</th><th>STATUS</th>
              </tr>
            </thead>
            <tbody id="ssh_tbody"></tbody>
          </table>
        </div>
      </div>


      <!-- 3D GLOBE TAB -->
      <div id="globe_tab" style="display:none" onmouseenter="initGlobe()">
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#127758; Thalassa 3D Globe — SSH Anomaly Intelligence + Route Optimizer</div>
          <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:center">
            <select id="globe_route_select" onchange="drawGlobeRoute()" style="flex:1;min-width:200px;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px 12px;border-radius:8px;font-size:12px">
              <option value="">— Select Route to Visualize —</option>
              <option value="istanbul_trabzon">Istanbul → Trabzon</option>
              <option value="istanbul_novorossiysk">Istanbul → Novorossiysk</option>
              <option value="odessa_istanbul">Odessa → Istanbul</option>
              <option value="batumi_constanta">Batumi → Constanta</option>
              <option value="shanghai_rotterdam">Shanghai → Rotterdam</option>
              <option value="rastanura_ningbo">Ras Tanura → Ningbo (VLCC)</option>
              <option value="singapore_rotterdam_cape">Singapore → Rotterdam (Cape)</option>
              <option value="murmansk_shanghai">Murmansk → Shanghai (Arctic NSR)</option>
              <option value="hormuz_transit">Hormuz Strait Transit</option>
              <option value="porthedland_qingdao">Port Hedland → Qingdao</option>
            </select>
            <button onclick="clearGlobeRoute()" style="background:transparent;border:1px solid #E74C3C;color:#E74C3C;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:12px">&#10005; Clear</button>
            <div id="globe_route_info" style="font-size:11px;color:var(--teal);font-weight:700"></div>
          </div>
          <div style="font-size:11px;color:#7F8C8D;margin-bottom:12px">
            Real-time SSH deviation visualization across 16 global maritime zones — 
            powered by NASA SWOT + Sentinel-6 + CYGNSS + PACE + SMAP + ICESat-2
          </div>
          <div id="globe_container" style="width:100%;height:550px;background:#000510;border-radius:12px;position:relative;overflow:hidden">
            <canvas id="globe_canvas" style="width:100%;height:100%"></canvas>
            <div id="globe_tooltip" style="position:absolute;display:none;background:#0D1F35;border:1px solid var(--teal);border-radius:8px;padding:10px 14px;font-size:11px;pointer-events:none;z-index:100;min-width:180px"></div>
            <div style="position:absolute;top:12px;left:12px;font-size:10px;color:#4A6FA5;letter-spacing:1px">
              THALASSA // SSH ANOMALY GLOBE // NASA SWOT+6SAT
            </div>
            <div style="position:absolute;bottom:12px;right:12px;display:flex;gap:8px;align-items:center">
              <div style="width:8px;height:8px;background:#E74C3C;border-radius:50%"></div>
              <span style="font-size:10px;color:#7F8C8D">HIGH</span>
              <div style="width:8px;height:8px;background:#F39C12;border-radius:50%"></div>
              <span style="font-size:10px;color:#7F8C8D">MID</span>
              <div style="width:8px;height:8px;background:#27AE60;border-radius:50%"></div>
              <span style="font-size:10px;color:#7F8C8D">NORMAL</span>
            </div>
          </div>
        </div>
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#127759; Satellite View — SSH Anomaly Overlay</div>
          <div id="sat_map" style="height:400px;border-radius:10px;overflow:hidden"></div>
        </div>

        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#127759; Satellite View — SSH Anomaly Overlay</div>
          <div id="sat_map" style="height:400px;border-radius:10px;overflow:hidden"></div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #E74C3C33">
            <div style="font-size:20px;font-weight:900;color:#E74C3C" id="g_crit">5</div>
            <div style="font-size:10px;color:var(--mute)">CRITICAL ZONES</div>
          </div>
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #F39C1233">
            <div style="font-size:20px;font-weight:900;color:#F39C12" id="g_warn">8</div>
            <div style="font-size:10px;color:var(--mute)">WARNING ZONES</div>
          </div>
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #27AE6033">
            <div style="font-size:20px;font-weight:900;color:#27AE60" id="g_norm">3</div>
            <div style="font-size:10px;color:var(--mute)">NORMAL ZONES</div>
          </div>
          <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid var(--teal)33">
            <div style="font-size:20px;font-weight:900;color:var(--teal)">9</div>
            <div style="font-size:10px;color:var(--mute)">SAT SOURCES</div>
          </div>
        </div>
      </div>


      <!-- ROUTE COMPARE TAB -->
      <div id="compare_tab" style="display:none">
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#9878; Route Comparison — Find the Most Efficient Path</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
            
            <!-- ROTA A -->
            <div>
              <div style="font-size:11px;color:var(--teal);font-weight:700;margin-bottom:8px;letter-spacing:1px">ROUTE A</div>
              <select id="compare_vessel_a" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px;margin-bottom:8px">
                <option value="Black Sea Cargo">Black Sea Cargo</option>
                <option value="Handy Bulk">Handy Bulk</option>
                <option value="Panamax Container">Panamax Container</option>
                <option value="Capesize Bulk">Capesize Bulk</option>
                <option value="LNG Carrier">LNG Carrier</option>
                <option value="VLCC Tanker">VLCC Tanker</option>
                <option value="Aframax Tanker">Aframax Tanker</option>
              </select>
              <select id="compare_route_a" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px;margin-bottom:8px">
                <option value="istanbul_trabzon">Istanbul → Trabzon</option>
                <option value="istanbul_novorossiysk">Istanbul → Novorossiysk</option>
                <option value="odessa_istanbul">Odessa → Istanbul</option>
                <option value="batumi_constanta">Batumi → Constanta</option>
                <option value="karadeniz_sakin">Black Sea — Calm</option>
                <option value="shanghai_rotterdam">Shanghai → Rotterdam</option>
                <option value="rastanura_ningbo">Ras Tanura → Ningbo</option>
                <option value="singapore_rotterdam_cape">Singapore → Rotterdam (Cape)</option>
                <option value="murmansk_shanghai">Murmansk → Shanghai (Arctic)</option>
                <option value="hormuz_transit">Hormuz Strait Transit</option>
              </select>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div>
                  <label style="font-size:10px;color:var(--mute)">SPEED (kn)</label>
                  <input type="number" id="compare_speed_a" value="12" min="5" max="25" style="width:100%;background:var(--bg);border:1px solid var(--line);color:white;padding:6px;border-radius:6px;font-size:12px">
                </div>
                <div>
                  <label style="font-size:10px;color:var(--mute)">DRAFT (m)</label>
                  <input type="number" id="compare_draft_a" value="8.5" min="3" max="22" step="0.5" style="width:100%;background:var(--bg);border:1px solid var(--line);color:white;padding:6px;border-radius:6px;font-size:12px">
                </div>
              </div>
            </div>

            <!-- ROTA B -->
            <div>
              <div style="font-size:11px;color:#E74C3C;font-weight:700;margin-bottom:8px;letter-spacing:1px">ROUTE B</div>
              <select id="compare_vessel_b" style="width:100%;background:var(--bg);border:1px solid #E74C3C;color:white;padding:8px;border-radius:8px;font-size:12px;margin-bottom:8px">
                <option value="Black Sea Cargo">Black Sea Cargo</option>
                <option value="Handy Bulk">Handy Bulk</option>
                <option value="Panamax Container">Panamax Container</option>
                <option value="Capesize Bulk">Capesize Bulk</option>
                <option value="LNG Carrier">LNG Carrier</option>
                <option value="VLCC Tanker">VLCC Tanker</option>
                <option value="Aframax Tanker">Aframax Tanker</option>
              </select>
              <select id="compare_route_b" style="width:100%;background:var(--bg);border:1px solid #E74C3C;color:white;padding:8px;border-radius:8px;font-size:12px;margin-bottom:8px">
                <option value="istanbul_novorossiysk" selected>Istanbul → Novorossiysk</option>
                <option value="istanbul_trabzon">Istanbul → Trabzon</option>
                <option value="odessa_istanbul">Odessa → Istanbul</option>
                <option value="batumi_constanta">Batumi → Constanta</option>
                <option value="karadeniz_sakin">Black Sea — Calm</option>
                <option value="shanghai_rotterdam">Shanghai → Rotterdam</option>
                <option value="rastanura_ningbo">Ras Tanura → Ningbo</option>
                <option value="singapore_rotterdam_cape">Singapore → Rotterdam (Cape)</option>
                <option value="murmansk_shanghai">Murmansk → Shanghai (Arctic)</option>
                <option value="hormuz_transit">Hormuz Strait Transit</option>
              </select>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div>
                  <label style="font-size:10px;color:var(--mute)">SPEED (kn)</label>
                  <input type="number" id="compare_speed_b" value="12" min="5" max="25" style="width:100%;background:var(--bg);border:1px solid var(--line);color:white;padding:6px;border-radius:6px;font-size:12px">
                </div>
                <div>
                  <label style="font-size:10px;color:var(--mute)">DRAFT (m)</label>
                  <input type="number" id="compare_draft_b" value="8.5" min="3" max="22" step="0.5" style="width:100%;background:var(--bg);border:1px solid var(--line);color:white;padding:6px;border-radius:6px;font-size:12px">
                </div>
              </div>
            </div>
          </div>

          <button onclick="runComparison()" style="width:100%;padding:12px;background:linear-gradient(135deg,#1ABC9C,#148F77);border:none;border-radius:10px;color:#001a0f;font-size:14px;font-weight:700;cursor:pointer;letter-spacing:1px;margin-bottom:16px">
            &#9878; RUN COMPARISON ANALYSIS
          </button>

          <!-- SONUÇLAR -->
          <div id="compare_results" style="display:none">
            <!-- Kazanan Banner -->
            <div id="compare_winner" style="border-radius:12px;padding:16px;text-align:center;margin-bottom:16px;font-size:16px;font-weight:700"></div>

            <!-- Metrik Karşılaştırma -->
            <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:8px;margin-bottom:16px;align-items:center">
              
              <!-- ROTA A Metrikler -->
              <div id="compare_metrics_a" style="background:var(--bg);border-radius:12px;padding:16px;border:2px solid var(--teal)">
                <div style="font-size:12px;color:var(--teal);font-weight:700;margin-bottom:12px;letter-spacing:1px">ROUTE A</div>
                <div id="compare_a_name" style="font-size:13px;font-weight:700;margin-bottom:12px;color:white"></div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                  <div style="text-align:center">
                    <div id="compare_a_drag" style="font-size:24px;font-weight:900;color:var(--teal)">—</div>
                    <div style="font-size:9px;color:var(--mute)">DRAG SCORE</div>
                  </div>
                  <div style="text-align:center">
                    <div id="compare_a_savings" style="font-size:24px;font-weight:900;color:var(--teal)">—</div>
                    <div style="font-size:9px;color:var(--mute)">SAVINGS</div>
                  </div>
                  <div style="text-align:center">
                    <div id="compare_a_cash" style="font-size:20px;font-weight:900;color:#F39C12">—</div>
                    <div style="font-size:9px;color:var(--mute)">ANNUAL $</div>
                  </div>
                  <div style="text-align:center">
                    <div id="compare_a_cii" style="font-size:20px;font-weight:900;color:#27AE60">—</div>
                    <div style="font-size:9px;color:var(--mute)">CII</div>
                  </div>
                </div>
              </div>

              <!-- VS -->
              <div style="text-align:center;font-size:20px;font-weight:900;color:var(--mute)">VS</div>

              <!-- ROTA B Metrikler -->
              <div id="compare_metrics_b" style="background:var(--bg);border-radius:12px;padding:16px;border:2px solid #E74C3C">
                <div style="font-size:12px;color:#E74C3C;font-weight:700;margin-bottom:12px;letter-spacing:1px">ROUTE B</div>
                <div id="compare_b_name" style="font-size:13px;font-weight:700;margin-bottom:12px;color:white"></div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                  <div style="text-align:center">
                    <div id="compare_b_drag" style="font-size:24px;font-weight:900;color:#E74C3C">—</div>
                    <div style="font-size:9px;color:var(--mute)">DRAG SCORE</div>
                  </div>
                  <div style="text-align:center">
                    <div id="compare_b_savings" style="font-size:24px;font-weight:900;color:#E74C3C">—</div>
                    <div style="font-size:9px;color:var(--mute)">SAVINGS</div>
                  </div>
                  <div style="text-align:center">
                    <div id="compare_b_cash" style="font-size:20px;font-weight:900;color:#F39C12">—</div>
                    <div style="font-size:9px;color:var(--mute)">ANNUAL $</div>
                  </div>
                  <div style="text-align:center">
                    <div id="compare_b_cii" style="font-size:20px;font-weight:900;color:#27AE60">—</div>
                    <div style="font-size:9px;color:var(--mute)">CII</div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Waypoint Karşılaştırma Tablosu -->
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
              <div>
                <div style="font-size:11px;color:var(--teal);margin-bottom:8px;font-weight:700">ROUTE A — WAYPOINTS</div>
                <table class="route-table">
                  <thead><tr><th>POINT</th><th>DRAG</th><th>STATUS</th></tr></thead>
                  <tbody id="compare_table_a"></tbody>
                </table>
              </div>
              <div>
                <div style="font-size:11px;color:#E74C3C;margin-bottom:8px;font-weight:700">ROUTE B — WAYPOINTS</div>
                <table class="route-table">
                  <thead><tr><th>POINT</th><th>DRAG</th><th>STATUS</th></tr></thead>
                  <tbody id="compare_table_b"></tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>


      <!-- EU ETS TAB -->
      <div id="ets_tab" style="display:none">

        <!-- Header Banner -->
        <div style="background:linear-gradient(135deg,#27AE6011,#1ABC9C11);border:1px solid #27AE60;border-radius:12px;padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px">
          <div style="font-size:24px">&#127776;</div>
          <div>
            <div style="color:#27AE60;font-weight:700;font-size:13px;letter-spacing:1px">EU ETS CARBON COST CALCULATOR — 2026</div>
            <div style="color:#7F8C8D;font-size:11px;margin-top:2px">100% compliance from Jan 2026 · CO2 + CH4 + N2O coverage · EUA price: ~€85/tonne</div>
          </div>
          <div style="margin-left:auto;text-align:right">
            <div style="color:#27AE60;font-size:20px;font-weight:900">€85</div>
            <div style="color:#7F8C8D;font-size:10px">EUA/tonne CO2</div>
          </div>
        </div>

        <!-- Inputs -->
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#9881; Carbon Cost Parameters</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">VESSEL TYPE</label>
              <select id="ets_vessel" onchange="updateETSDefaults()" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
                <option value="VLCC Tanker">VLCC Tanker</option>
                <option value="Capesize Bulk">Capesize Bulk</option>
                <option value="Panamax Container">Panamax Container</option>
                <option value="Aframax Tanker">Aframax Tanker</option>
                <option value="LNG Carrier">LNG Carrier</option>
                <option value="Panamax Bulk">Panamax Bulk</option>
                <option value="Black Sea Cargo" selected>Black Sea Cargo</option>
              </select>
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">FUEL CONSUMPTION (t/day)</label>
              <input type="number" id="ets_fuel" value="12" min="5" max="300" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">VOYAGE DAYS/YEAR</label>
              <input type="number" id="ets_days" value="280" min="50" max="365" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">EUA PRICE (€/tonne CO2)</label>
              <input type="number" id="ets_price" value="85" min="40" max="200" style="width:100%;background:var(--bg);border:1px solid #27AE60;color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">EUR/USD RATE</label>
              <input type="number" id="ets_fx" value="1.09" min="0.8" max="1.5" step="0.01" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">BATIMETRIX SAVINGS (%)</label>
              <input type="number" id="ets_savings" value="10" min="1" max="20" step="0.1" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">EU PORT EXPOSURE (%)</label>
              <input type="range" id="ets_exposure" value="60" min="0" max="100" oninput="document.getElementById('ets_exposure_val').textContent=this.value+'%'" style="width:100%">
              <div style="text-align:center;color:var(--teal);font-weight:700" id="ets_exposure_val">60%</div>
              <div style="font-size:9px;color:var(--mute)">% of voyages touching EU/EEA ports</div>
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">FUEL TYPE</label>
              <select id="ets_fuel_type" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
                <option value="3.151">VLSFO (3.151 tCO2/t)</option>
                <option value="3.114">HFO 380 (3.114 tCO2/t)</option>
                <option value="3.206">MGO (3.206 tCO2/t)</option>
                <option value="2.750">LNG (2.750 tCO2/t)</option>
              </select>
            </div>
          </div>
          <button onclick="calcETS()" style="width:100%;padding:12px;background:linear-gradient(135deg,#27AE60,#1E8449);border:none;border-radius:10px;color:white;font-size:14px;font-weight:700;cursor:pointer;letter-spacing:1px">
            &#127776; CALCULATE EU ETS CARBON COST
          </button>
        </div>

        <!-- Results -->
        <div id="ets_results" style="display:none">

          <!-- KPI Grid -->
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid #E74C3C33">
              <div style="font-size:24px;font-weight:900;color:#E74C3C" id="ets_total_cost">—</div>
              <div style="font-size:10px;color:var(--mute)">TOTAL ETS COST/YEAR</div>
            </div>
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid #27AE6033">
              <div style="font-size:24px;font-weight:900;color:#27AE60" id="ets_saved">—</div>
              <div style="font-size:10px;color:var(--mute)">SAVED WITH BATIMETRIX</div>
            </div>
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid #F39C1233">
              <div style="font-size:24px;font-weight:900;color:#F39C12" id="ets_co2">—</div>
              <div style="font-size:10px;color:var(--mute)">CO2 EMISSIONS/YEAR</div>
            </div>
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid var(--teal)33">
              <div style="font-size:24px;font-weight:900;color:var(--teal)" id="ets_co2_saved">—</div>
              <div style="font-size:10px;color:var(--mute)">CO2 REDUCED</div>
            </div>
          </div>

          <!-- Breakdown -->
          <div class="card" style="margin-bottom:16px">
            <div class="card-title">&#128202; Carbon Cost Breakdown</div>
            <div id="ets_breakdown" style="font-size:12px;line-height:2.2"></div>
          </div>

          <!-- Compliance Status -->
          <div class="card">
            <div class="card-title">&#9878; EU ETS Compliance Timeline 2024-2026</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:8px">
              <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #4A6FA5">
                <div style="font-size:20px;font-weight:900;color:#4A6FA5">40%</div>
                <div style="font-size:11px;color:white;margin:4px 0">2024</div>
                <div style="font-size:10px;color:var(--mute)">CO2 coverage</div>
                <div style="font-size:9px;color:#27AE60;margin-top:4px">✅ Completed</div>
              </div>
              <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #F39C12">
                <div style="font-size:20px;font-weight:900;color:#F39C12">70%</div>
                <div style="font-size:11px;color:white;margin:4px 0">2025</div>
                <div style="font-size:10px;color:var(--mute)">CO2 coverage</div>
                <div style="font-size:9px;color:#27AE60;margin-top:4px">✅ Completed</div>
              </div>
              <div style="background:var(--bg);border-radius:10px;padding:14px;text-align:center;border:1px solid #E74C3C">
                <div style="font-size:20px;font-weight:900;color:#E74C3C">100%</div>
                <div style="font-size:11px;color:white;margin:4px 0">2026</div>
                <div style="font-size:10px;color:var(--mute)">CO2 + CH4 + N2O</div>
                <div style="font-size:9px;color:#E74C3C;margin-top:4px">🔴 FULL COMPLIANCE</div>
              </div>
            </div>
            <div style="margin-top:12px;padding:12px;background:#E74C3C11;border:1px solid #E74C3C33;border-radius:8px;font-size:11px;color:#7F8C8D">
              ⚠️ <b style="color:#E74C3C">Non-compliance penalty:</b> €150,000–€1,000,000+ per vessel annually. 
              Port state detention risk for repeated violations. 
              Batimetrix reduces emissions → reduces ETS liability.
            </div>
          </div>
        </div>
      </div>


      <!-- SPEED OPTIMIZATION TAB -->
      <div id="speed_tab" style="display:none">

        <!-- Header -->
        <div style="background:linear-gradient(135deg,#F39C1211,#E67E2211);border:1px solid #F39C12;border-radius:12px;padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px">
          <div style="font-size:24px">&#9889;</div>
          <div>
            <div style="color:#F39C12;font-weight:700;font-size:13px;letter-spacing:1px">THALASSA SPEED OPTIMIZATION WIZARD</div>
            <div style="color:#7F8C8D;font-size:11px;margin-top:2px">NASA SSH anomaly-aware optimal speed calculation · CII + EU ETS + Fuel cost integrated</div>
          </div>
        </div>

        <!-- Inputs -->
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#9881; Voyage Parameters</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">VESSEL TYPE</label>
              <select id="speed_vessel" onchange="updateSpeedDefaults()" style="width:100%;background:var(--bg);border:1px solid #F39C12;color:white;padding:8px;border-radius:8px;font-size:12px">
                <option value="Black Sea Cargo">Black Sea Cargo</option>
                <option value="Handy Bulk">Handy Bulk</option>
                <option value="Panamax Container">Panamax Container</option>
                <option value="Capesize Bulk">Capesize Bulk</option>
                <option value="LNG Carrier">LNG Carrier</option>
                <option value="VLCC Tanker">VLCC Tanker</option>
                <option value="Aframax Tanker">Aframax Tanker</option>
              </select>
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">DISTANCE (nm)</label>
              <input type="number" id="speed_distance" value="1200" min="100" max="15000" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">MAX SPEED (kn)</label>
              <input type="number" id="speed_max" value="14" min="8" max="25" step="0.5" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">FUEL CONSUMPTION AT MAX (t/day)</label>
              <input type="number" id="speed_fuel_max" value="12" min="5" max="300" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">BUNKER PRICE ($/ton)</label>
              <input type="number" id="speed_bunker" value="650" min="200" max="1500" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">HIRE RATE ($/day)</label>
              <input type="number" id="speed_hire" value="15000" min="1000" max="200000" step="1000" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">SSH DRAG IMPACT (%)</label>
              <input type="number" id="speed_ssh_drag" value="8" min="0" max="30" step="0.5" style="width:100%;background:var(--bg);border:1px solid #F39C12;color:white;padding:8px;border-radius:8px;font-size:12px">
              <div style="font-size:9px;color:var(--mute);margin-top:2px">From NASA SWOT SSH analysis</div>
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">EUA PRICE (€/tonne CO2)</label>
              <input type="number" id="speed_eua" value="85" min="40" max="200" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <button onclick="calcSpeedOpt()" style="width:100%;padding:12px;background:linear-gradient(135deg,#F39C12,#E67E22);border:none;border-radius:10px;color:white;font-size:14px;font-weight:700;cursor:pointer;letter-spacing:1px">
            &#9889; FIND OPTIMAL SPEED
          </button>
        </div>

        <!-- Results -->
        <div id="speed_results" style="display:none">

          <!-- Optimal Speed Banner -->
          <div id="speed_winner_banner" style="border-radius:12px;padding:20px;text-align:center;margin-bottom:16px;border:2px solid #F39C12;background:#F39C1211">
            <div style="font-size:13px;color:#7F8C8D;margin-bottom:4px">THALASSA RECOMMENDED OPTIMAL SPEED</div>
            <div id="speed_optimal" style="font-size:48px;font-weight:900;color:#F39C12">— kn</div>
            <div id="speed_optimal_reason" style="font-size:12px;color:#7F8C8D;margin-top:4px"></div>
          </div>

          <!-- Speed Comparison Table -->
          <div class="card" style="margin-bottom:16px">
            <div class="card-title">&#128202; Speed vs Cost Analysis (NASA SSH-adjusted)</div>
            <table class="route-table">
              <thead>
                <tr>
                  <th>SPEED</th>
                  <th>VOYAGE TIME</th>
                  <th>FUEL COST</th>
                  <th>HIRE COST</th>
                  <th>ETS COST</th>
                  <th>TOTAL COST</th>
                  <th>STATUS</th>
                </tr>
              </thead>
              <tbody id="speed_table"></tbody>
            </table>
          </div>

          <!-- Savings vs Max Speed -->
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid #27AE6033">
              <div style="font-size:22px;font-weight:900;color:#27AE60" id="speed_saving">—</div>
              <div style="font-size:10px;color:var(--mute)">SAVED VS MAX SPEED</div>
            </div>
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid #F39C1233">
              <div style="font-size:22px;font-weight:900;color:#F39C12" id="speed_time_add">—</div>
              <div style="font-size:10px;color:var(--mute)">EXTRA VOYAGE TIME</div>
            </div>
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid var(--teal)33">
              <div style="font-size:22px;font-weight:900;color:var(--teal)" id="speed_co2_save">—</div>
              <div style="font-size:10px;color:var(--mute)">CO2 SAVED (tonnes)</div>
            </div>
          </div>
        </div>
      </div>


      <!-- SPEED OPTIMIZATION TAB -->
      <div id="speed_tab" style="display:none">

        <!-- Header -->
        <div style="background:linear-gradient(135deg,#F39C1211,#E67E2211);border:1px solid #F39C12;border-radius:12px;padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px">
          <div style="font-size:24px">&#9889;</div>
          <div>
            <div style="color:#F39C12;font-weight:700;font-size:13px;letter-spacing:1px">THALASSA SPEED OPTIMIZATION WIZARD</div>
            <div style="color:#7F8C8D;font-size:11px;margin-top:2px">NASA SSH anomaly-aware optimal speed calculation · CII + EU ETS + Fuel cost integrated</div>
          </div>
        </div>

        <!-- Inputs -->
        <div class="card" style="margin-bottom:16px">
          <div class="card-title">&#9881; Voyage Parameters</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">VESSEL TYPE</label>
              <select id="speed_vessel" onchange="updateSpeedDefaults()" style="width:100%;background:var(--bg);border:1px solid #F39C12;color:white;padding:8px;border-radius:8px;font-size:12px">
                <option value="Black Sea Cargo">Black Sea Cargo</option>
                <option value="Handy Bulk">Handy Bulk</option>
                <option value="Panamax Container">Panamax Container</option>
                <option value="Capesize Bulk">Capesize Bulk</option>
                <option value="LNG Carrier">LNG Carrier</option>
                <option value="VLCC Tanker">VLCC Tanker</option>
                <option value="Aframax Tanker">Aframax Tanker</option>
              </select>
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">DISTANCE (nm)</label>
              <input type="number" id="speed_distance" value="1200" min="100" max="15000" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">MAX SPEED (kn)</label>
              <input type="number" id="speed_max" value="14" min="8" max="25" step="0.5" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">FUEL CONSUMPTION AT MAX (t/day)</label>
              <input type="number" id="speed_fuel_max" value="12" min="5" max="300" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">BUNKER PRICE ($/ton)</label>
              <input type="number" id="speed_bunker" value="650" min="200" max="1500" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">HIRE RATE ($/day)</label>
              <input type="number" id="speed_hire" value="15000" min="1000" max="200000" step="1000" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:12px">
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">SSH DRAG IMPACT (%)</label>
              <input type="number" id="speed_ssh_drag" value="8" min="0" max="30" step="0.5" style="width:100%;background:var(--bg);border:1px solid #F39C12;color:white;padding:8px;border-radius:8px;font-size:12px">
              <div style="font-size:9px;color:var(--mute);margin-top:2px">From NASA SWOT SSH analysis</div>
            </div>
            <div>
              <label style="font-size:10px;color:var(--mute);display:block;margin-bottom:4px">EUA PRICE (€/tonne CO2)</label>
              <input type="number" id="speed_eua" value="85" min="40" max="200" style="width:100%;background:var(--bg);border:1px solid var(--teal);color:white;padding:8px;border-radius:8px;font-size:12px">
            </div>
          </div>
          <button onclick="calcSpeedOpt()" style="width:100%;padding:12px;background:linear-gradient(135deg,#F39C12,#E67E22);border:none;border-radius:10px;color:white;font-size:14px;font-weight:700;cursor:pointer;letter-spacing:1px">
            &#9889; FIND OPTIMAL SPEED
          </button>
        </div>

        <!-- Results -->
        <div id="speed_results" style="display:none">

          <!-- Optimal Speed Banner -->
          <div id="speed_winner_banner" style="border-radius:12px;padding:20px;text-align:center;margin-bottom:16px;border:2px solid #F39C12;background:#F39C1211">
            <div style="font-size:13px;color:#7F8C8D;margin-bottom:4px">THALASSA RECOMMENDED OPTIMAL SPEED</div>
            <div id="speed_optimal" style="font-size:48px;font-weight:900;color:#F39C12">— kn</div>
            <div id="speed_optimal_reason" style="font-size:12px;color:#7F8C8D;margin-top:4px"></div>
          </div>

          <!-- Speed Comparison Table -->
          <div class="card" style="margin-bottom:16px">
            <div class="card-title">&#128202; Speed vs Cost Analysis (NASA SSH-adjusted)</div>
            <table class="route-table">
              <thead>
                <tr>
                  <th>SPEED</th>
                  <th>VOYAGE TIME</th>
                  <th>FUEL COST</th>
                  <th>HIRE COST</th>
                  <th>ETS COST</th>
                  <th>TOTAL COST</th>
                  <th>STATUS</th>
                </tr>
              </thead>
              <tbody id="speed_table"></tbody>
            </table>
          </div>

          <!-- Savings vs Max Speed -->
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid #27AE6033">
              <div style="font-size:22px;font-weight:900;color:#27AE60" id="speed_saving">—</div>
              <div style="font-size:10px;color:var(--mute)">SAVED VS MAX SPEED</div>
            </div>
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid #F39C1233">
              <div style="font-size:22px;font-weight:900;color:#F39C12" id="speed_time_add">—</div>
              <div style="font-size:10px;color:var(--mute)">EXTRA VOYAGE TIME</div>
            </div>
            <div style="background:var(--bg);border-radius:12px;padding:16px;text-align:center;border:1px solid var(--teal)33">
              <div style="font-size:22px;font-weight:900;color:var(--teal)" id="speed_co2_save">—</div>
              <div style="font-size:10px;color:var(--mute)">CO2 SAVED (tonnes)</div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</div>

<footer>
  <div class="footer-dots"><div class="footer-dot"></div><div class="footer-dot"></div><div class="footer-dot"></div></div>
  <div>BATIMETRIX V3 PRO // PINN 1,657,025 params // NASA SWOT+GPM+MODIS+Sentinel-6+CYGNSS+PACE // GEBCO 2026 // IMO CII MEPC.354(78)</div>
  <a href="https://github.com/Batimetrix/batimetrix" target="_blank">github.com/Batimetrix/batimetrix</a>
</footer>

<script>
// ===== LANGUAGES =====
var LANG={
  en:{tagline:"Proactive Hydrodynamic Intelligence",sat_swot:"Sea Surface Height",sat_gpm:"Storm Prediction",sat_modis:"SST Viscosity",sat_gebco:"Bathymetry 2026",mission_params:"Mission Parameters",vessel_type:"Vessel Type",route_lbl:"Route",speed_lbl:"Speed (kn)",draft_lbl:"Draft (m)",wave_lbl:"Wave Ht (m)",temp_lbl:"Sea Temp °C",voyage_lbl:"Annual Voyage Days",run_btn:"⚡ Run Analysis",drag_score:"Drag Score",fuel_savings:"Fuel Savings",annual_savings:"Annual Savings",co2_cut:"CO2 Cut / yr",tab_map:"🗺️ Route Map",tab_analysis:"📊 Analysis",tab_cii:"⚖️ CII Rating",tab_table:"📋 Telemetry",map_title:"Route Map — Live Drag Overlay",drag_profile:"Drag Profile",drag_per_wp:"Drag per Waypoint",savings_per_wp:"Savings per Waypoint",depth_profile:"Depth Profile",drag_vs_depth:"Drag vs Depth",cii_title:"IMO CII Rating 2026",without_b:"Without Batimetrix",with_b:"With Batimetrix",telemetry:"Route Telemetry",th_waypoint:"Waypoint",th_depth:"Depth",th_savings:"Savings",th_status:"Status",fuel_eff:"Fuel Efficiency",analyzing:"RUNNING PINN INFERENCE...",empty_txt:"Select vessel, route and run analysis",cii_improved:"✅ CII rating improved with Batimetrix!",cii_same:"CII rating — further optimization possible",efficient:"EFFICIENT",nominal:"NOMINAL",high_drag:"HIGH DRAG"},
  tr:{tagline:"Proaktif Hidrodinamik Zeka",sat_swot:"Deniz Yüzey Yüksekliği",sat_gpm:"Fırtına Tahmini",sat_modis:"SST Viskozite",sat_gebco:"Batimetri 2026",mission_params:"Görev Parametreleri",vessel_type:"Gemi Tipi",route_lbl:"Güzergah",speed_lbl:"Hız (kn)",draft_lbl:"Taslak (m)",wave_lbl:"Dalga (m)",temp_lbl:"Deniz Sıcaklığı",voyage_lbl:"Yıllık Sefer Günü",run_btn:"⚡ Analizi Başlat",drag_score:"Sürüklenme Skoru",fuel_savings:"Yakıt Tasarrufu",annual_savings:"Yıllık Tasarruf",co2_cut:"CO2 Kesinti/yıl",tab_map:"🗺️ Harita",tab_analysis:"📊 Analiz",tab_cii:"⚖️ CII Notu",tab_table:"📋 Telemetri",map_title:"Güzergah Haritası",drag_profile:"Sürüklenme Profili",drag_per_wp:"Nokta Başı Drag",savings_per_wp:"Nokta Başı Tasarruf",depth_profile:"Derinlik Profili",drag_vs_depth:"Drag - Derinlik",cii_title:"IMO CII Notu 2026",without_b:"Batimetrix Olmadan",with_b:"Batimetrix İle",telemetry:"Güzergah Telemetrisi",th_waypoint:"Nokta",th_depth:"Derinlik",th_savings:"Tasarruf",th_status:"Durum",fuel_eff:"Yakıt Verimliliği",analyzing:"PINN ÇIKARIMI ÇALIŞIYOR...",empty_txt:"Gemi ve güzergah seçin, analizi başlatın",cii_improved:"✅ CII notu Batimetrix ile iyileşti!",cii_same:"CII notu — daha fazla optimizasyon mümkün",efficient:"VERİMLİ",nominal:"NORMAL",high_drag:"YÜKSEK DİRENÇ"},
  el:{tagline:"Προληπτική Υδροδυναμική Νοημοσύνη",sat_swot:"Ύψος Επιφάνειας",sat_gpm:"Πρόβλεψη Καταιγίδας",sat_modis:"Θερμοκρασία Θάλασσας",sat_gebco:"Βαθυμετρία 2026",mission_params:"Παράμετροι Αποστολής",vessel_type:"Τύπος Πλοίου",route_lbl:"Διαδρομή",speed_lbl:"Ταχύτητα (κόμβοι)",draft_lbl:"Βύθισμα (μ)",wave_lbl:"Ύψος Κύματος (μ)",temp_lbl:"Θερμ. Θάλασσας",voyage_lbl:"Ετήσιες Ημέρες Πλου",run_btn:"⚡ Εκτέλεση Ανάλυσης",drag_score:"Δείκτης Αντίστασης",fuel_savings:"Εξοικονόμηση",annual_savings:"Ετήσια Εξοικονόμηση",co2_cut:"Μείωση CO2/έτος",tab_map:"🗺️ Χάρτης",tab_analysis:"📊 Ανάλυση",tab_cii:"⚖️ CII",tab_table:"📋 Τηλεμετρία",map_title:"Χάρτης Διαδρομής",drag_profile:"Προφίλ Αντίστασης",drag_per_wp:"Αντίσταση ανά Σημείο",savings_per_wp:"Εξοικονόμηση ανά Σημείο",depth_profile:"Προφίλ Βάθους",drag_vs_depth:"Αντίσταση - Βάθος",cii_title:"Βαθμολογία IMO CII 2026",without_b:"Χωρίς Batimetrix",with_b:"Με Batimetrix",telemetry:"Τηλεμετρία Διαδρομής",th_waypoint:"Σημείο",th_depth:"Βάθος",th_savings:"Εξοικονόμηση",th_status:"Κατάσταση",fuel_eff:"Απόδοση Καυσίμου",analyzing:"ΕΚΤΕΛΕΣΗ PINN...",empty_txt:"Επιλέξτε πλοίο και διαδρομή",cii_improved:"✅ Η βαθμολογία CII βελτιώθηκε!",cii_same:"CII — δυνατή περαιτέρω βελτιστοποίηση",efficient:"ΑΠΟΔΟΤΙΚΟ",nominal:"ΚΑΝΟΝΙΚΟ",high_drag:"ΥΨΗΛΗ ΑΝΤΙΣΤΑΣΗ"},
  zh:{tagline:"主动式水动力阻力预测引擎",sat_swot:"海面高度",sat_gpm:"风暴预测",sat_modis:"海面温度",sat_gebco:"海底地形2026",mission_params:"任务参数",vessel_type:"船舶类型",route_lbl:"航线",speed_lbl:"航速（节）",draft_lbl:"吃水（米）",wave_lbl:"波高（米）",temp_lbl:"海温°C",voyage_lbl:"年航行天数",run_btn:"⚡ 运行分析",drag_score:"阻力指数",fuel_savings:"燃油节省",annual_savings:"年度节省",co2_cut:"CO2减排/年",tab_map:"🗺️ 航线图",tab_analysis:"📊 分析",tab_cii:"⚖️ CII评级",tab_table:"📋 遥测",map_title:"航线图",drag_profile:"阻力分析",drag_per_wp:"各航点阻力",savings_per_wp:"各航点节省",depth_profile:"水深分析",drag_vs_depth:"阻力-水深",cii_title:"IMO CII评级2026",without_b:"未使用Batimetrix",with_b:"使用Batimetrix",telemetry:"航线遥测",th_waypoint:"航点",th_depth:"水深",th_savings:"节省",th_status:"状态",fuel_eff:"燃油效率",analyzing:"AI分析运行中...",empty_txt:"选择船舶和航线，运行分析",cii_improved:"✅ CII评级已提升！",cii_same:"CII — 可进一步优化",efficient:"高效",nominal:"正常",high_drag:"高阻力"},
  ru:{tagline:"Проактивная гидродинамическая система",sat_swot:"Высота поверхности моря",sat_gpm:"Прогноз шторма",sat_modis:"Температура моря",sat_gebco:"Батиметрия 2026",mission_params:"Параметры миссии",vessel_type:"Тип судна",route_lbl:"Маршрут",speed_lbl:"Скорость (уз)",draft_lbl:"Осадка (м)",wave_lbl:"Высота волны (м)",temp_lbl:"Темп. моря °C",voyage_lbl:"Дней плавания в год",run_btn:"⚡ Запустить анализ",drag_score:"Индекс сопротивления",fuel_savings:"Экономия топлива",annual_savings:"Годовая экономия",co2_cut:"Снижение CO2/год",tab_map:"🗺️ Карта",tab_analysis:"📊 Анализ",tab_cii:"⚖️ CII",tab_table:"📋 Телеметрия",map_title:"Карта маршрута",drag_profile:"Профиль сопротивления",drag_per_wp:"Сопротивление по точкам",savings_per_wp:"Экономия по точкам",depth_profile:"Профиль глубины",drag_vs_depth:"Сопротивление - Глубина",cii_title:"Рейтинг IMO CII 2026",without_b:"Без Batimetrix",with_b:"С Batimetrix",telemetry:"Телеметрия маршрута",th_waypoint:"Точка",th_depth:"Глубина",th_savings:"Экономия",th_status:"Статус",fuel_eff:"Топливная эффективность",analyzing:"ВЫПОЛНЯЕТСЯ PINN...",empty_txt:"Выберите судно и маршрут",cii_improved:"✅ Рейтинг CII улучшен!",cii_same:"CII — возможна оптимизация",efficient:"ЭФФЕКТИВНО",nominal:"НОРМАЛЬНО",high_drag:"ВЫСОКОЕ СОПРОТИВЛЕНИЕ"},
  es:{tagline:"Inteligencia Hidrodinámica Proactiva",sat_swot:"Altura Superficie Marina",sat_gpm:"Predicción Tormentas",sat_modis:"Temperatura del Mar",sat_gebco:"Batimetría 2026",mission_params:"Parámetros de Misión",vessel_type:"Tipo de Buque",route_lbl:"Ruta",speed_lbl:"Velocidad (nudos)",draft_lbl:"Calado (m)",wave_lbl:"Altura Ola (m)",temp_lbl:"Temp. Mar °C",voyage_lbl:"Días Navegación/año",run_btn:"⚡ Ejecutar Análisis",drag_score:"Índice de Resistencia",fuel_savings:"Ahorro Combustible",annual_savings:"Ahorro Anual",co2_cut:"Reducción CO2/año",tab_map:"🗺️ Mapa de Ruta",tab_analysis:"📊 Análisis",tab_cii:"⚖️ CII",tab_table:"📋 Telemetría",map_title:"Mapa de Ruta",drag_profile:"Perfil de Resistencia",drag_per_wp:"Resistencia por Punto",savings_per_wp:"Ahorro por Punto",depth_profile:"Perfil de Profundidad",drag_vs_depth:"Resistencia - Profundidad",cii_title:"Calificación IMO CII 2026",without_b:"Sin Batimetrix",with_b:"Con Batimetrix",telemetry:"Telemetría de Ruta",th_waypoint:"Punto",th_depth:"Profundidad",th_savings:"Ahorro",th_status:"Estado",fuel_eff:"Eficiencia Combustible",analyzing:"EJECUTANDO PINN...",empty_txt:"Seleccione buque y ruta",cii_improved:"✅ ¡Calificación CII mejorada!",cii_same:"CII — optimización posible",efficient:"EFICIENTE",nominal:"NOMINAL",high_drag:"ALTA RESISTENCIA"},
  fr:{tagline:"Intelligence Hydrodynamique Proactive",sat_swot:"Hauteur Surface Marine",sat_gpm:"Prédiction Tempêtes",sat_modis:"Température de Mer",sat_gebco:"Bathymétrie 2026",mission_params:"Paramètres de Mission",vessel_type:"Type de Navire",route_lbl:"Route",speed_lbl:"Vitesse (noeuds)",draft_lbl:"Tirant d'eau (m)",wave_lbl:"Hauteur vagues (m)",temp_lbl:"Temp. mer °C",voyage_lbl:"Jours de navigation/an",run_btn:"⚡ Lancer l'analyse",drag_score:"Indice de résistance",fuel_savings:"Économie carburant",annual_savings:"Économie annuelle",co2_cut:"Réduction CO2/an",tab_map:"🗺️ Carte de route",tab_analysis:"📊 Analyse",tab_cii:"⚖️ CII",tab_table:"📋 Télémétrie",map_title:"Carte de Route",drag_profile:"Profil de résistance",drag_per_wp:"Résistance par point",savings_per_wp:"Économie par point",depth_profile:"Profil de profondeur",drag_vs_depth:"Résistance - Profondeur",cii_title:"Notation IMO CII 2026",without_b:"Sans Batimetrix",with_b:"Avec Batimetrix",telemetry:"Télémétrie de route",th_waypoint:"Point",th_depth:"Profondeur",th_savings:"Économie",th_status:"Statut",fuel_eff:"Efficacité carburant",analyzing:"EXÉCUTION PINN...",empty_txt:"Sélectionnez navire et route",cii_improved:"✅ Note CII améliorée avec Batimetrix!",cii_same:"CII — optimisation possible",efficient:"EFFICACE",nominal:"NORMAL",high_drag:"FORTE RÉSISTANCE"}
};

var lang="en", lastData=null;
var map=null, routeLayer=null, markerLayer=null;
var charts={};

// Clock
setInterval(function(){
  var d=new Date();
  document.getElementById("clock").textContent=
    String(d.getUTCHours()).padStart(2,"0")+":"+
    String(d.getUTCMinutes()).padStart(2,"0")+":"+
    String(d.getUTCSeconds()).padStart(2,"0")+" UTC";
},1000);

// Init map
function initMap(){
  if(map) return;
  if(!window.leafletReady){window.pendingMapInit=true;return;}
  map=L.map("map",{zoomControl:true,attributionControl:false}).setView([42,33],5);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",{
    maxZoom:18,subdomains:"abcd"
  }).addTo(map);

}

var aisMarkers={};
var aisWS=null;
function startAIS(){
  if(aisWS) return;
  var API_KEY="d451be71cb01176584568bcd91c632349587eaff";
  aisWS=new WebSocket("wss://stream.aisstream.io/v0/stream");
  aisWS.onopen=function(){
    aisWS.send(JSON.stringify({
      APIKey:API_KEY,
      BoundingBoxes:[[[-90,-180],[90,180]]],
      FilterMessageTypes:["PositionReport"]
    }));
  };
  aisWS.onmessage=function(ev){
    try{
      var msg=JSON.parse(ev.data);
      if(msg.MessageType!=="PositionReport") return;
      var r=msg.Message.PositionReport;
      var mmsi=r.UserID;
      var lat=r.Latitude, lon=r.Longitude;
      var cog=r.Cog||0;
      if(aisMarkers[mmsi]){
        aisMarkers[mmsi].setLatLng([lat,lon]);
      }else{
        if(Object.keys(aisMarkers).length>500) return;
        var icon=L.divIcon({
          className:"ais-ship",
          html:"<div style='width:8px;height:8px;background:#00E5B0;border-radius:50%;box-shadow:0 0 6px #00E5B0;transform:rotate("+cog+"deg)'></div>",
          iconSize:[8,8]
        });
        var m=L.marker([lat,lon],{icon:icon}).addTo(map);
        var name=(msg.MetaData&&msg.MetaData.ShipName)?msg.MetaData.ShipName.trim():"Vessel";
        m.bindTooltip("<b>"+name+"</b><br>MMSI: "+mmsi+"<br>Speed: "+(r.Sog||0)+" kn",{className:"leaflet-tooltip-dark"});
        aisMarkers[mmsi]=m;
      }
    }catch(e){}
  };
  aisWS.onclose=function(){aisWS=null;setTimeout(startAIS,5000);};
}

function previewRoute(){
  initMap();
  var rkey=document.getElementById("route").value;
  fetch("/route_data/"+rkey).then(r=>r.json()).then(function(wps){
    if(routeLayer){map.removeLayer(routeLayer);}
    if(markerLayer){map.removeLayer(markerLayer);}
    var latlngs=wps.map(w=>[w.lat,w.lon]);
    routeLayer=L.polyline(latlngs,{color:"#00E5B0",weight:2,dashArray:"6,4",opacity:.6}).addTo(map);
    markerLayer=L.layerGroup();
    wps.forEach(function(w,i){
      var m=L.circleMarker([w.lat,w.lon],{radius:6,color:"#00E5B0",fillColor:"#071525",fillOpacity:1,weight:2});
      m.bindTooltip("<b>"+w.name+"</b><br>Depth: "+w.depth+"m<br>SSH: "+w.ssh.toFixed(3)+"m",{className:"leaflet-tooltip-dark"});
      markerLayer.addLayer(m);
    });
    markerLayer.addTo(map);
    map.fitBounds(routeLayer.getBounds(),{padding:[30,30]});
  });
}

function updateMap(data){
  if(typeof L==="undefined"||typeof L.polyline!=="function"){setTimeout(function(){updateMap(data);},300);return;}
  if(map){try{map.remove();}catch(e){} map=null; routeLayer=null; markerLayer=null;}
  initMap();
  if(routeLayer){map.removeLayer(routeLayer);}
  if(markerLayer){map.removeLayer(markerLayer);}
  var latlngs=data.waypoints.map(w=>[w.lat,w.lon]);
  routeLayer=L.polyline(latlngs,{color:"#00E5B0",weight:3,opacity:.8}).addTo(map);
  markerLayer=L.layerGroup();
  data.waypoints.forEach(function(w){
    var c=w.drag<0.20?"#00E5B0":w.drag<0.35?"#FFC107":"#FF4757";
    var m=L.circleMarker([w.lat,w.lon],{radius:8,color:c,fillColor:c,fillOpacity:.3,weight:2});
    var depthCat = w.depth < 200 ? "🟡 Shallow" : w.depth < 1000 ? "🟢 Continental" : "🔵 Deep Ocean";
    var status = w.drag < 0.20 ? "✅ EFFICIENT" : w.drag < 0.35 ? "⚠️ NOMINAL" : "🔴 HIGH DRAG";
    var swh = typeof w.swh !== "undefined" ? w.swh.toFixed(1)+"m" : "—";
    m.bindPopup(
      "<div style='font-family:JetBrains Mono;font-size:11px;min-width:200px;line-height:1.8'>"+
      "<b style='color:#00E5B0;font-size:13px'>"+w.name+"</b><br>"+
      "<hr style='border:none;border-top:1px solid #0F2A42;margin:4px 0'>"+
      "<span style='color:#4A6FA5'>DEPTH</span>  <b>"+w.depth+"m</b>  "+depthCat+"<br>"+
      "<span style='color:#4A6FA5'>SSH   </span>  <b>"+w.ssh.toFixed(3)+"m</b><br>"+
      "<span style='color:#4A6FA5'>SWH   </span>  <b>"+swh+"</b><br>"+
      "<hr style='border:none;border-top:1px solid #0F2A42;margin:4px 0'>"+
      "<span style='color:#4A6FA5'>DRAG  </span>  <b style='color:"+c+"'>"+w.drag.toFixed(4)+"</b><br>"+
      "<span style='color:#4A6FA5'>SAVING</span>  <b style='color:#00E5B0'>"+w.savings.toFixed(1)+"%</b><br>"+
      "<span style='color:#4A6FA5'>STATUS</span>  "+status+
      "</div>",
      {maxWidth: 240}
    );
    markerLayer.addLayer(m);
  });
  markerLayer.addTo(map);
  setTimeout(function(){
    map.invalidateSize();
    map.fitBounds(routeLayer.getBounds(),{padding:[30,30]});
  },300);
}

function setLang(l){
  lang=l;
  var t=LANG[l];
  document.querySelectorAll("[data-i18n]").forEach(function(el){
    var k=el.getAttribute("data-i18n");
    if(t[k]) el.textContent=t[k];
  });
  if(lastData) renderResults(lastData);
}

function showTab(id, el){
  ["map_tab","analysis_tab","cii_tab","table_tab","fleet_tab","ssh_tab","globe_tab","compare_tab","ets_tab","speed_tab"].forEach(function(t){
    document.getElementById(t).style.display="none";
  });
  document.querySelectorAll(".tab").forEach(function(t){t.classList.remove("active")});
  document.getElementById(id).style.display="block";
  el.classList.add("active");
  if(id==="map_tab"){if(!map){initMap();} setTimeout(function(){if(map)map.invalidateSize();},200);}
  if(id==="globe_tab"){setTimeout(function(){initGlobe();initSatMap();},300);}
}

function destroyChart(id){
  if(charts[id]){charts[id].destroy();delete charts[id];}
}

function makeChart(id,type,labels,datasets,opts){
  destroyChart(id);
  var ctx=document.getElementById(id).getContext("2d");
  charts[id]=new Chart(ctx,{
    type:type,
    data:{labels:labels,datasets:datasets},
    options:Object.assign({
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:"#4A6FA5",font:{size:10}}},tooltip:{backgroundColor:"#071525",borderColor:"#0F2A42",borderWidth:1}},
      scales:{x:{ticks:{color:"#4A6FA5",font:{size:9}},grid:{color:"#0F2A42"}},y:{ticks:{color:"#4A6FA5",font:{size:9}},grid:{color:"#0F2A42"}}}
    },opts||{})
  });
}

function renderResults(data){
  var t=LANG[lang];
  var wps=data.waypoints;
  var labels=wps.map(w=>w.name);
  var drags=wps.map(w=>w.drag);
  var savs=wps.map(w=>w.savings);
  var depths=wps.map(w=>w.depth);

  // Drag chart
  makeChart("chart_drag","bar",labels,[{
    label:"Drag Score",data:drags,
    backgroundColor:drags.map(d=>d<0.20?"rgba(0,229,176,.7)":d<0.35?"rgba(255,193,7,.7)":"rgba(255,71,87,.7)"),
    borderRadius:4
  }]);

  // Savings chart
  makeChart("chart_sav","bar",labels,[{
    label:"Savings %",data:savs,
    backgroundColor:"rgba(0,229,176,.6)",borderRadius:4
  }]);

  // Depth chart
  makeChart("chart_depth","line",labels,[{
    label:"Depth (m)",data:depths,
    borderColor:"#3498DB",backgroundColor:"rgba(52,152,219,.1)",
    tension:.4,fill:true,pointBackgroundColor:"#3498DB",pointRadius:4
  }]);

  // Scatter
  makeChart("chart_scatter","scatter",null,[{
    label:"Drag vs Depth",
    data:wps.map(w=>({x:w.depth,y:w.drag})),
    backgroundColor:"rgba(0,229,176,.7)",pointRadius:7
  }]);

  // CII chart
  makeChart("chart_cii","bar",["Baseline","With Batimetrix"],[{
    label:"CII Value",
    data:[data.cii_b_val,data.cii_a_val],
    backgroundColor:["rgba(255,71,87,.7)","rgba(0,229,176,.7)"],
    borderRadius:6
  }]);

  // CO2 chart
  var co2_base=data.co2_azalma/(data.tasarruf/100)*1;
  makeChart("chart_co2","doughnut",["CO2 Reduced","Remaining"],[{
    data:[data.co2_azalma,Math.max(0,co2_base-data.co2_azalma)],
    backgroundColor:["rgba(0,229,176,.8)","rgba(15,42,66,.8)"],
    borderWidth:0
  }],{scales:{}});

  // Table
  var tbody=document.getElementById("table_body");
  tbody.innerHTML="";
  wps.forEach(function(w){
    var cls=w.drag<0.20?"wg":w.drag<0.35?"wy":"wr";
    var badge=w.drag<0.20?"badge-green":""+w.drag<0.35?"badge-yellow":"badge-red";
    var lbl=w.drag<0.20?t.efficient:w.drag<0.35?t.nominal:t.high_drag;
    var color=w.drag<0.20?"#00E5B0":w.drag<0.35?"#FFC107":"#FF4757";
    tbody.innerHTML+="<tr>"+
      "<td><span class='wp-dot "+cls+"'></span>"+w.name+"</td>"+
      "<td>"+w.depth+"m</td>"+
      "<td>"+w.ssh.toFixed(3)+"</td>"+
      "<td>"+w.swh.toFixed(1)+"</td>"+
      "<td style='color:"+color+";font-weight:700'>"+w.drag.toFixed(4)+"</td>"+
      "<td style='color:#00E5B0'>"+w.savings.toFixed(1)+"%</td>"+
      "<td><span class='badge "+(w.drag<0.20?"badge-green":w.drag<0.35?"badge-yellow":"badge-red")+"'>"+lbl+"</span></td>"+
      "</tr>";
  });

  // CII
  document.getElementById("cii_before").textContent=data.cii_before;
  document.getElementById("cii_after").textContent=data.cii_after;
  document.getElementById("cii_before_val").textContent=data.cii_b_val.toFixed(2)+" g/DWT·nm";
  document.getElementById("cii_after_val").textContent=data.cii_a_val.toFixed(2)+" g/DWT·nm";
  document.getElementById("cii_before_card").className="cii-card c"+data.cii_before;
  document.getElementById("cii_after_card").className="cii-card c"+data.cii_after;
  var improved=data.cii_before!==data.cii_after;
  var msg=document.getElementById("cii_msg");
  msg.textContent=improved?t.cii_improved:t.cii_same;
  msg.style.background=improved?"rgba(0,229,176,.1)":"rgba(74,111,165,.1)";
  msg.style.color=improved?"#00E5B0":"#4A6FA5";

  // Progress bars
  var dp=Math.min(data.drag*100,100);
  document.getElementById("fill_drag").style.width=dp+"%";
  document.getElementById("prog_drag").textContent=data.drag.toFixed(4);
  document.getElementById("fill_eff").style.width=(100-dp)+"%";
  document.getElementById("prog_eff").textContent=(100-dp).toFixed(1)+"%";

  // Map
}

function runAnalysis(){
  var vessel=document.getElementById("vessel").value;
  var route=document.getElementById("route").value;
  var speed=parseFloat(document.getElementById("speed").value);
  var draft=parseFloat(document.getElementById("draft").value);
  var swh=parseFloat(document.getElementById("swh").value);
  var sst=parseFloat(document.getElementById("sst").value);
  var days=parseInt(document.getElementById("days").value);

  document.getElementById("loading").style.display="block";
  document.getElementById("results").style.display="none";
  document.getElementById("empty_state").style.display="none";

  fetch("/analyze",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({vessel,route,speed,draft,swh,sst,days})
  })
  .then(r=>r.json())
  .then(function(data){
    lastData=data;
    document.getElementById("loading").style.display="none";
    document.getElementById("results").style.display="block";

    // KPIs
    document.getElementById("kpi_drag").textContent=data.drag.toFixed(4);
    document.getElementById("kpi_fuel").textContent="%"+data.savings.toFixed(1);
    document.getElementById("kpi_cash").textContent="$"+(data.cost_savings/1000).toFixed(0)+"K";
    document.getElementById("kpi_co2").textContent=data.co2_reduction.toFixed(0)+"t";

    // Sidebar stats
    document.getElementById("stat_drag").textContent=data.drag.toFixed(3);
    document.getElementById("stat_sav").textContent="%"+data.savings.toFixed(1);
    document.getElementById("stat_cii").textContent=data.cii_before+"→"+data.cii_after;

    renderResults(data);
    var pdfBtn = document.getElementById("btn_pdf");
    if(pdfBtn) pdfBtn.style.display = "block";
    var pdfBtn = document.getElementById("btn_pdf");
    if(pdfBtn) pdfBtn.style.display = "block";
    var vsel = document.getElementById("vessel").value;
    var rsel = document.getElementById("route").options[document.getElementById("route").selectedIndex].text;
    addToFleet(data, vsel, rsel);
    setTimeout(function(){
      if(map){try{map.remove();}catch(e){} map=null; routeLayer=null; markerLayer=null;}
      initMap();
      updateMap(lastData);
    },400);
  });
}

// ===== WORLD CLOCKS (38 UTC offsets) =====
var WCT_ZONES=[
[-12,"BAKER IS."],[-11,"PAGO PAGO"],[-10,"HONOLULU"],[-9.5,"MARQUESAS"],[-9,"ANCHORAGE"],
[-8,"LOS ANGELES"],[-7,"DENVER"],[-6,"MEXICO CITY"],[-5,"NEW YORK"],[-4,"SANTIAGO"],
[-3.5,"ST. JOHN'S"],[-3,"SAO PAULO"],[-2,"S. GEORGIA"],[-1,"AZORES"],
[0,"LONDON"],[1,"PARIS"],[2,"CAIRO"],[3,"ISTANBUL"],[3.5,"TEHRAN"],[4,"DUBAI"],
[4.5,"KABUL"],[5,"KARACHI"],[5.5,"MUMBAI"],[5.75,"KATHMANDU"],[6,"DHAKA"],
[6.5,"YANGON"],[7,"BANGKOK"],[8,"SHANGHAI"],[8.75,"EUCLA"],[9,"TOKYO"],
[9.5,"ADELAIDE"],[10,"SYDNEY"],[10.5,"LORD HOWE"],[11,"HONIARA"],[12,"AUCKLAND"],
[12.75,"CHATHAM"],[13,"NUKUALOFA"],[14,"KIRITIMATI"]
];
function wctFmt(off){
  var h=Math.floor(Math.abs(off)),m=Math.round((Math.abs(off)-h)*60);
  return (off<0?"-":"+")+h+(m?":"+(m<10?"0":"")+m:"");
}
function wctRender(){
  var now=new Date();
  var utcMs=now.getTime()+now.getTimezoneOffset()*60000;
  var html="";
  WCT_ZONES.forEach(function(z){
    var t=new Date(utcMs+z[0]*3600000);
    var hh=("0"+t.getHours()).slice(-2),mm=("0"+t.getMinutes()).slice(-2);
    var hl=z[1]==="ISTANBUL"?" wct-hl":"";
    html+='<span class="wct-item'+hl+'">'+z[1]+' <b>'+hh+":"+mm+'</b><span class="wct-off">UTC'+wctFmt(z[0])+'</span></span>';
  });
  var el=document.getElementById("wct_track");
  if(el) el.innerHTML=html+html;
}
setInterval(wctRender,30000);
wctRender();

// ===== FLEET DASHBOARD =====
var fleetData = [
  {name:"MV Bosphorus",  type:"Black Sea Cargo",   route:"Istanbul → Trabzon",           drag:0.142, sav:11.2, cash:187,  cii_b:"D", cii_a:"C"},
  {name:"MT Black Sea",  type:"Aframax Tanker",     route:"Novorossiysk → Istanbul",      drag:0.198, sav:9.8,  cash:1240, cii_b:"C", cii_a:"B"},
  {name:"MV Silk Road",  type:"Panamax Container",  route:"Shanghai → Rotterdam",         drag:0.231, sav:8.4,  cash:894,  cii_b:"D", cii_a:"C"},
  {name:"MT Gulf Star",  type:"VLCC Tanker",        route:"Ras Tanura → Ningbo",          drag:0.167, sav:12.1, cash:1447, cii_b:"E", cii_a:"C"},
  {name:"MV Iron Giant", type:"Capesize Bulk",      route:"Port Hedland → Qingdao",       drag:0.189, sav:10.3, cash:712,  cii_b:"D", cii_a:"B"},
];

function renderFleet(){
  var tbody = document.getElementById("fleet_tbody");
  if(!tbody) return;
  tbody.innerHTML = "";
  var totalCash = 0, improved = 0;
  fleetData.forEach(function(v){
    totalCash += v.cash;
    if(v.cii_b !== v.cii_a) improved++;
    var dragCol = v.drag < 0.20 ? "var(--teal)" : v.drag < 0.35 ? "var(--amber)" : "#E74C3C";
    var status = v.drag < 0.20 ? "✅ EFFICIENT" : v.drag < 0.35 ? "⚠️ NOMINAL" : "🔴 HIGH DRAG";
    var ciiCol = v.cii_a === "A" ? "#27AE60" : v.cii_a === "B" ? "var(--teal)" : v.cii_a === "C" ? "var(--amber)" : "#E74C3C";
    tbody.innerHTML += "<tr>" +
      "<td><b>"+v.name+"</b></td>" +
      "<td style='color:var(--mute)'>"+v.type+"</td>" +
      "<td style='color:var(--mute);font-size:10px'>"+v.route+"</td>" +
      "<td style='color:"+dragCol+";font-weight:700'>"+v.drag.toFixed(3)+"</td>" +
      "<td style='color:var(--teal)'>%"+v.sav.toFixed(1)+"</td>" +
      "<td style='color:var(--amber);font-weight:700'>$"+v.cash+"K</td>" +
      "<td style='color:"+ciiCol+";font-weight:900;font-size:14px'>"+v.cii_b+"→"+v.cii_a+"</td>" +
      "<td>"+status+"</td>" +
      "</tr>";
  });
  // KPI guncelle
  var totalEl = document.getElementById("fleet_savings");
  if(totalEl) totalEl.textContent = "$"+(totalCash/1000).toFixed(1)+"M";
  var co2El = document.getElementById("fleet_co2");
  if(co2El){
    var co2 = fleetData.reduce(function(s,v){return s + v.cash*1000/650*3.151*0.1;},0);
    co2El.textContent = Math.round(co2).toLocaleString()+"t";
  }
  var ciiEl = document.getElementById("fleet_cii");
  if(ciiEl) ciiEl.textContent = improved+" improved";
}

// Analiz sonucu filo verisine ekle
function addToFleet(data, vessel, route){
  var existing = fleetData.findIndex(function(v){ return v.name === "⚡ "+vessel; });
  var entry = {
    name: "⚡ "+vessel,
    type: vessel,
    route: route,
    drag: data.drag,
    sav: data.savings,
    cash: Math.round(data.cost_savings/1000),
    cii_b: data.cii_before,
    cii_a: data.cii_after
  };
  if(existing >= 0) fleetData[existing] = entry;
  else fleetData.unshift(entry);
  document.getElementById("fleet_vessels").textContent = fleetData.length;
  renderFleet();
}

// ===== PDF REPORT =====
function downloadPDF(){
  // Tarih guncelle
  var now = new Date();
  var dateStr = now.toLocaleDateString("en-GB",{
    year:"numeric", month:"long", day:"numeric",
    hour:"2-digit", minute:"2-digit"
  });
  var el = document.getElementById("print_date");
  if(el) el.textContent = dateStr;

  // Aktif tab'lari goster, digerlerini gizle
  var tabs = ["map_tab","analysis_tab","cii_tab","table_tab","fleet_tab"];
  var hidden = [];
  tabs.forEach(function(id){
    var el = document.getElementById(id);
    if(el && el.style.display === "none"){
      hidden.push(id);
      el.style.display = "block";
    }
  });

  // Print
  window.print();

  // Geri al
  hidden.forEach(function(id){
    var el = document.getElementById(id);
    if(el) el.style.display = "none";
  });
}

// ===== PDF REPORT =====



// ===== SSH INTELLIGENCE =====
var SSH_ZONES=[
  {name:"Hormuz Strait",    region:"Persian Gulf",  ssh:0.08, base:0.05, wind:8.2},
  {name:"Malacca Strait",   region:"SE Asia",       ssh:0.10, base:0.08, wind:6.1},
  {name:"Bab el-Mandeb",    region:"Red Sea",       ssh:0.09, base:0.07, wind:11.2},
  {name:"Dover Strait",     region:"North Sea",     ssh:0.07, base:0.06},
  {name:"Taiwan Strait",    region:"East Asia",     ssh:0.12, base:0.09, wind:9.4},
  {name:"Suez Canal",       region:"Mediterranean", ssh:0.05, base:0.05},
  {name:"North Atlantic",   region:"Atlantic",      ssh:0.35, base:0.28, wind:14.5, chl:1.24},
  {name:"Mid Pacific",      region:"Pacific",       ssh:0.35, base:0.30, wind:12.8, chl:0.18},
  {name:"Cape Good Hope",   region:"S.Atlantic",    ssh:0.30, base:0.24, wind:16.3, chl:0.95},
  {name:"Arabian Sea",      region:"Indian Ocean",  ssh:0.16, base:0.14, wind:7.8, chl:0.61},
  {name:"South China Sea",  region:"SE Asia",       ssh:0.15, base:0.13},
  {name:"Bay of Bengal",    region:"Indian Ocean",  ssh:0.16, base:0.14},
  {name:"Kara Sea",         region:"Arctic",        ssh:0.10, base:0.07, ice:1.8},
  {name:"Gulf of Mexico",   region:"Americas",      ssh:0.14, base:0.12},
  {name:"Mediterranean",    region:"Mediterranean", ssh:0.12, base:0.10},
  {name:"Black Sea",        region:"Black Sea",     ssh:0.09, base:0.08, wind:5.2, chl:2.81}
];

function renderSSH(){
  var tbody=document.getElementById("ssh_tbody");
  if(!tbody) return;
  tbody.innerHTML="";
  var crit=0,warn=0,norm=0,maxDev=0;
  SSH_ZONES.forEach(function(z){
    var dev=parseFloat((z.ssh-z.base).toFixed(3));
    var absD=Math.abs(dev);
    if(absD>maxDev) maxDev=absD;
    var col,icon,label;
    if(absD>=0.06){col="#E74C3C";icon="&#128308;";label="CRITICAL";crit++;}
    else if(absD>=0.02){col="#F39C12";icon="&#128993;";label="WARNING";warn++;}
    else{col="#27AE60";icon="&#128994;";label="NORMAL";norm++;}
    var dragPct=Math.round(absD*180);
    var dragCol=dragPct>15?"#E74C3C":dragPct>5?"#F39C12":"#27AE60";
    tbody.innerHTML+="<tr>"+
      "<td><b>"+z.name+"</b></td>"+
      "<td style='color:var(--mute)'>"+z.region+"</td>"+
      "<td style='font-family:JetBrains Mono'>"+z.ssh.toFixed(3)+"</td>"+
      "<td style='font-family:JetBrains Mono;color:var(--mute)'>"+z.base.toFixed(3)+"</td>"+
      "<td style='font-family:JetBrains Mono;color:"+col+";font-weight:700'>"+(dev>=0?"+":"")+dev.toFixed(3)+"m</td>"+
      "<td style='color:"+dragCol+"'>"+(dragPct>0?"+"+dragPct+"%":"—")+"</td>"+
      "<td>"+icon+" "+label+"</td>"+"<td style='color:#9B59B6'>"+(z.wind?z.wind.toFixed(1)+" m/s":"—")+"</td>"+"<td style='color:#27AE60'>"+(z.chl?z.chl+" mg/m³":"—")+"</td>"+"<td style='color:#E67E22'>"+(z.sal?z.sal+" PSU":"—")+"</td>"+"<td style='color:#3498DB'>"+(z.ice?z.ice+" m":"—")+"</td></tr>";
  });
  var c=document.getElementById("ssh_crit");if(c)c.textContent=crit;
  var w=document.getElementById("ssh_warn");if(w)w.textContent=warn;
  var n=document.getElementById("ssh_norm");if(n)n.textContent=norm;
  var cnt=document.getElementById("ssh_count");if(cnt)cnt.textContent=crit+warn;
  var mx=document.getElementById("ssh_max");if(mx)mx.textContent="+"+maxDev.toFixed(3)+"m";
}




// ===== SATELLITE MAP =====
var satMapInitialized = false;
var satMap = null;

function initSatMap() {
    if (satMapInitialized) return;
    if (!document.getElementById('sat_map')) return;
    satMapInitialized = true;

    satMap = L.map('sat_map', {zoomControl: true}).setView([20, 0], 2);

    // ESRI World Imagery - gercek uydu goruntüsü
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 18
    }).addTo(satMap);

    // SSH Anomali bölgeleri
    var SSH_ZONES = [
        {name:"Hormuz Strait",    lat:26.57, lon:56.25,  dev:0.030, wind:8.2,  label:"Persian Gulf",   region:"Critical Chokepoint"},
        {name:"Malacca Strait",   lat:2.50,  lon:101.00, dev:0.020, wind:6.1,  label:"SE Asia",         region:"Major Trade Route"},
        {name:"Bab el-Mandeb",    lat:12.60, lon:43.30,  dev:0.020, wind:11.2, label:"Red Sea",         region:"Strategic Strait"},
        {name:"Taiwan Strait",    lat:24.00, lon:119.00, dev:0.030, wind:9.4,  label:"East Asia",       region:"High Traffic"},
        {name:"North Atlantic",   lat:48.00, lon:-30.00, dev:0.070, wind:14.5, label:"Atlantic",        region:"Major Ocean"},
        {name:"Mid Pacific",      lat:42.00, lon:-175.0, dev:0.050, wind:12.8, label:"Pacific",         region:"Major Ocean"},
        {name:"Cape Good Hope",   lat:-35.0, lon:20.00,  dev:0.060, wind:16.3, label:"S.Atlantic",      region:"Critical Route"},
        {name:"Arabian Sea",      lat:18.00, lon:62.00,  dev:0.020, wind:7.8,  label:"Indian Ocean",    region:"Trade Route"},
        {name:"South China Sea",  lat:10.00, lon:113.00, dev:0.020, wind:8.5,  label:"SE Asia",         region:"Disputed Waters"},
        {name:"Kara Sea",         lat:75.00, lon:65.00,  dev:0.030, wind:5.1,  label:"Arctic",          region:"Arctic Route"},
        {name:"Gulf of Mexico",   lat:26.00, lon:-88.0,  dev:0.020, wind:6.8,  label:"Americas",        region:"Energy Hub"},
        {name:"Mediterranean",    lat:36.00, lon:15.00,  dev:0.020, wind:7.4,  label:"Mediterranean",   region:"Major Trade"},
        {name:"Black Sea",        lat:42.00, lon:33.00,  dev:0.010, wind:5.2,  label:"Black Sea",       region:"Regional Sea"},
        {name:"Dover Strait",     lat:51.00, lon:1.50,   dev:0.010, wind:9.3,  label:"North Sea",       region:"Busiest Strait"},
        {name:"Bay of Bengal",    lat:15.00, lon:90.00,  dev:0.020, wind:7.2,  label:"Indian Ocean",    region:"Trade Route"},
        {name:"Suez Canal",       lat:29.97, lon:32.55,  dev:0.000, wind:4.3,  label:"Mediterranean",   region:"Critical Canal"}
    ];

    SSH_ZONES.forEach(function(z) {
        var col = z.dev >= 0.06 ? '#E74C3C' : z.dev >= 0.02 ? '#F39C12' : '#27AE60';
        var label = z.dev >= 0.06 ? 'CRITICAL' : z.dev >= 0.02 ? 'WARNING' : 'NORMAL';
        var radius = z.dev >= 0.06 ? 350000 : z.dev >= 0.02 ? 250000 : 180000;

        // Daire
        L.circle([z.lat, z.lon], {
            color: col,
            fillColor: col,
            fillOpacity: 0.15,
            weight: 2,
            radius: radius
        }).addTo(satMap).bindPopup(
            "<div style='font-family:JetBrains Mono;font-size:12px;min-width:200px;line-height:1.8'>" +
            "<b style='color:" + col + ";font-size:14px'>" + z.name + "</b><br>" +
            "<span style='color:#666'>" + z.region + "</span><br>" +
            "<hr style='margin:4px 0;border-color:#ddd'>" +
            "<b>SSH Deviation:</b> " + (z.dev>=0?"+":"") + z.dev.toFixed(3) + "m<br>" +
            "<b>Wind Speed:</b> " + z.wind + " m/s<br>" +
            "<b>Status:</b> <span style='color:" + col + ";font-weight:700'>" + label + "</span><br>" +
            "<b>Source:</b> NASA SWOT + Sentinel-6" +
            "</div>"
        );

        // Merkez nokta
        L.circleMarker([z.lat, z.lon], {
            color: col,
            fillColor: col,
            fillOpacity: 0.9,
            weight: 2,
            radius: 6
        }).addTo(satMap).bindTooltip(z.name, {permanent: false});
    });

    setTimeout(function(){satMap.invalidateSize();}, 300);
}


// ===== SATELLITE MAP =====
var satMapInitialized = false;
var satMap = null;

function initSatMap() {
    if (satMapInitialized) return;
    if (!document.getElementById('sat_map')) return;
    satMapInitialized = true;

    satMap = L.map('sat_map', {zoomControl: true}).setView([20, 0], 2);

    // ESRI World Imagery - gercek uydu goruntüsü
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 18
    }).addTo(satMap);

    // SSH Anomali bölgeleri
    var SSH_ZONES = [
        {name:"Hormuz Strait",    lat:26.57, lon:56.25,  dev:0.030, wind:8.2,  label:"Persian Gulf",   region:"Critical Chokepoint"},
        {name:"Malacca Strait",   lat:2.50,  lon:101.00, dev:0.020, wind:6.1,  label:"SE Asia",         region:"Major Trade Route"},
        {name:"Bab el-Mandeb",    lat:12.60, lon:43.30,  dev:0.020, wind:11.2, label:"Red Sea",         region:"Strategic Strait"},
        {name:"Taiwan Strait",    lat:24.00, lon:119.00, dev:0.030, wind:9.4,  label:"East Asia",       region:"High Traffic"},
        {name:"North Atlantic",   lat:48.00, lon:-30.00, dev:0.070, wind:14.5, label:"Atlantic",        region:"Major Ocean"},
        {name:"Mid Pacific",      lat:42.00, lon:-175.0, dev:0.050, wind:12.8, label:"Pacific",         region:"Major Ocean"},
        {name:"Cape Good Hope",   lat:-35.0, lon:20.00,  dev:0.060, wind:16.3, label:"S.Atlantic",      region:"Critical Route"},
        {name:"Arabian Sea",      lat:18.00, lon:62.00,  dev:0.020, wind:7.8,  label:"Indian Ocean",    region:"Trade Route"},
        {name:"South China Sea",  lat:10.00, lon:113.00, dev:0.020, wind:8.5,  label:"SE Asia",         region:"Disputed Waters"},
        {name:"Kara Sea",         lat:75.00, lon:65.00,  dev:0.030, wind:5.1,  label:"Arctic",          region:"Arctic Route"},
        {name:"Gulf of Mexico",   lat:26.00, lon:-88.0,  dev:0.020, wind:6.8,  label:"Americas",        region:"Energy Hub"},
        {name:"Mediterranean",    lat:36.00, lon:15.00,  dev:0.020, wind:7.4,  label:"Mediterranean",   region:"Major Trade"},
        {name:"Black Sea",        lat:42.00, lon:33.00,  dev:0.010, wind:5.2,  label:"Black Sea",       region:"Regional Sea"},
        {name:"Dover Strait",     lat:51.00, lon:1.50,   dev:0.010, wind:9.3,  label:"North Sea",       region:"Busiest Strait"},
        {name:"Bay of Bengal",    lat:15.00, lon:90.00,  dev:0.020, wind:7.2,  label:"Indian Ocean",    region:"Trade Route"},
        {name:"Suez Canal",       lat:29.97, lon:32.55,  dev:0.000, wind:4.3,  label:"Mediterranean",   region:"Critical Canal"}
    ];

    SSH_ZONES.forEach(function(z) {
        var col = z.dev >= 0.06 ? '#E74C3C' : z.dev >= 0.02 ? '#F39C12' : '#27AE60';
        var label = z.dev >= 0.06 ? 'CRITICAL' : z.dev >= 0.02 ? 'WARNING' : 'NORMAL';
        var radius = z.dev >= 0.06 ? 350000 : z.dev >= 0.02 ? 250000 : 180000;

        // Daire
        L.circle([z.lat, z.lon], {
            color: col,
            fillColor: col,
            fillOpacity: 0.15,
            weight: 2,
            radius: radius
        }).addTo(satMap).bindPopup(
            "<div style='font-family:JetBrains Mono;font-size:12px;min-width:200px;line-height:1.8'>" +
            "<b style='color:" + col + ";font-size:14px'>" + z.name + "</b><br>" +
            "<span style='color:#666'>" + z.region + "</span><br>" +
            "<hr style='margin:4px 0;border-color:#ddd'>" +
            "<b>SSH Deviation:</b> " + (z.dev>=0?"+":"") + z.dev.toFixed(3) + "m<br>" +
            "<b>Wind Speed:</b> " + z.wind + " m/s<br>" +
            "<b>Status:</b> <span style='color:" + col + ";font-weight:700'>" + label + "</span><br>" +
            "<b>Source:</b> NASA SWOT + Sentinel-6" +
            "</div>"
        );

        // Merkez nokta
        L.circleMarker([z.lat, z.lon], {
            color: col,
            fillColor: col,
            fillOpacity: 0.9,
            weight: 2,
            radius: 6
        }).addTo(satMap).bindTooltip(z.name, {permanent: false});
    });

    setTimeout(function(){satMap.invalidateSize();}, 300);
}



// ===== ROUTE COMPARISON =====
function runComparison() {
    var vesselA = document.getElementById('compare_vessel_a').value;
    var routeA  = document.getElementById('compare_route_a').value;
    var speedA  = parseFloat(document.getElementById('compare_speed_a').value);
    var draftA  = parseFloat(document.getElementById('compare_draft_a').value);

    var vesselB = document.getElementById('compare_vessel_b').value;
    var routeB  = document.getElementById('compare_route_b').value;
    var speedB  = parseFloat(document.getElementById('compare_speed_b').value);
    var draftB  = parseFloat(document.getElementById('compare_draft_b').value);

    // Her iki rotayı da analiz et
    Promise.all([
        fetch('/analyze', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({
                vessel: vesselA, route: routeA,
                speed: speedA, draft: draftA,
                swh: 1.2, sst: 22, sefer_gun: 280
            })
        }).then(r=>r.json()),
        fetch('/analyze', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({
                vessel: vesselB, route: routeB,
                speed: speedB, draft: draftB,
                swh: 1.2, sst: 22, sefer_gun: 280
            })
        }).then(r=>r.json())
    ]).then(function(results) {
        var a = results[0];
        var b = results[1];

        document.getElementById('compare_results').style.display = 'block';

        // Rota isimleri
        var routeNames = {
            'istanbul_trabzon': 'Istanbul → Trabzon',
            'istanbul_novorossiysk': 'Istanbul → Novorossiysk',
            'odessa_istanbul': 'Odessa → Istanbul',
            'batumi_constanta': 'Batumi → Constanta',
            'karadeniz_sakin': 'Black Sea — Calm',
            'shanghai_rotterdam': 'Shanghai → Rotterdam',
            'rastanura_ningbo': 'Ras Tanura → Ningbo',
            'singapore_rotterdam_cape': 'Singapore → Rotterdam',
            'murmansk_shanghai': 'Murmansk → Shanghai',
            'hormuz_transit': 'Hormuz Transit'
        };

        document.getElementById('compare_a_name').textContent = vesselA + ' / ' + (routeNames[routeA] || routeA);
        document.getElementById('compare_b_name').textContent = vesselB + ' / ' + (routeNames[routeB] || routeB);

        // Metrikler A
        document.getElementById('compare_a_drag').textContent = a.drag.toFixed(4);
        document.getElementById('compare_a_savings').textContent = '%' + a.savings.toFixed(1);
        document.getElementById('compare_a_cash').textContent = '$' + (a.cost_savings/1000).toFixed(0) + 'K';
        document.getElementById('compare_a_cii').textContent = a.cii_before + '→' + a.cii_after;

        // Metrikler B
        document.getElementById('compare_b_drag').textContent = b.drag.toFixed(4);
        document.getElementById('compare_b_savings').textContent = '%' + b.savings.toFixed(1);
        document.getElementById('compare_b_cash').textContent = '$' + (b.cost_savings/1000).toFixed(0) + 'K';
        document.getElementById('compare_b_cii').textContent = b.cii_before + '→' + b.cii_after;

        // Kazanan
        var winner = document.getElementById('compare_winner');
        if (a.drag < b.drag) {
            winner.style.background = 'linear-gradient(135deg,#1ABC9C22,#1ABC9C11)';
            winner.style.border = '2px solid #1ABC9C';
            winner.innerHTML = '&#9989; ROUTE A is more efficient — ' + ((b.drag - a.drag)*100).toFixed(1) + '% less drag';
            winner.style.color = '#1ABC9C';
            document.getElementById('compare_metrics_a').style.borderColor = '#1ABC9C';
            document.getElementById('compare_metrics_b').style.borderColor = '#4A6FA5';
        } else if (b.drag < a.drag) {
            winner.style.background = 'linear-gradient(135deg,#E74C3C22,#E74C3C11)';
            winner.style.border = '2px solid #E74C3C';
            winner.innerHTML = '&#9989; ROUTE B is more efficient — ' + ((a.drag - b.drag)*100).toFixed(1) + '% less drag';
            winner.style.color = '#E74C3C';
            document.getElementById('compare_metrics_b').style.borderColor = '#E74C3C';
            document.getElementById('compare_metrics_a').style.borderColor = '#4A6FA5';
        } else {
            winner.innerHTML = '&#9866; ROUTES ARE EQUAL in efficiency';
            winner.style.color = '#F39C12';
        }

        // Waypoint tabloları
        var tbA = document.getElementById('compare_table_a');
        var tbB = document.getElementById('compare_table_b');
        tbA.innerHTML = '';
        tbB.innerHTML = '';

        a.waypoints.forEach(function(w) {
            var col = w.drag < 0.20 ? 'var(--teal)' : w.drag < 0.35 ? '#F39C12' : '#E74C3C';
            var st = w.drag < 0.20 ? '✅' : w.drag < 0.35 ? '⚠️' : '🔴';
            tbA.innerHTML += '<tr><td>' + w.name + '</td><td style="color:' + col + ';font-weight:700">' + w.drag.toFixed(4) + '</td><td>' + st + '</td></tr>';
        });

        b.waypoints.forEach(function(w) {
            var col = w.drag < 0.20 ? 'var(--teal)' : w.drag < 0.35 ? '#F39C12' : '#E74C3C';
            var st = w.drag < 0.20 ? '✅' : w.drag < 0.35 ? '⚠️' : '🔴';
            tbB.innerHTML += '<tr><td>' + w.name + '</td><td style="color:' + col + ';font-weight:700">' + w.drag.toFixed(4) + '</td><td>' + st + '</td></tr>';
        });

    }).catch(function(err) {
        console.error('Compare error:', err);
    });
}


// ===== FUEL COST CALCULATOR =====
function calcFuel() {
    var price    = parseFloat(document.getElementById('fuel_price').value) || 650;
    var days     = parseFloat(document.getElementById('fuel_days').value) || 20;
    var consump  = parseFloat(document.getElementById('fuel_consumption').value) || 80;
    var savPct   = parseFloat(document.getElementById('fuel_savings_pct').value) || 10;
    var voyages  = parseFloat(document.getElementById('fuel_voyages').value) || 12;

    // Hesaplamalar
    var totalFuel    = consump * days;
    var savedFuel    = totalFuel * (savPct / 100);
    var fuelWithout  = totalFuel * price;
    var fuelWith     = (totalFuel - savedFuel) * price;
    var savingVoyage = fuelWithout - fuelWith;
    var savingYear   = savingVoyage * voyages;
    var co2Saved     = savedFuel * voyages * 3.151;

    // Formatla
    function fmt(n) {
        if (n >= 1000000) return '$' + (n/1000000).toFixed(2) + 'M';
        if (n >= 1000) return '$' + (n/1000).toFixed(1) + 'K';
        return '$' + n.toFixed(0);
    }

    document.getElementById('fuel_results').style.display = 'block';
    document.getElementById('fuel_cost_without').textContent = fmt(fuelWithout);
    document.getElementById('fuel_cost_with').textContent    = fmt(fuelWith);
    document.getElementById('fuel_saving_voyage').textContent = fmt(savingVoyage);
    document.getElementById('fuel_saving_year').textContent  = fmt(savingYear);

    document.getElementById('fuel_breakdown').innerHTML =
        '<b style="color:var(--teal)">Voyage Analysis:</b><br>' +
        'Total fuel consumption: <b>' + totalFuel.toFixed(0) + ' tons</b><br>' +
        'Fuel saved with Batimetrix: <b style="color:#27AE60">' + savedFuel.toFixed(1) + ' tons/voyage</b><br>' +
        'Bunker price: <b>$' + price + '/ton</b><br>' +
        'Voyage duration: <b>' + days + ' days</b><br>' +
        'Annual voyages: <b>' + voyages + '</b><br>' +
        'Drag reduction: <b style="color:var(--teal)">' + savPct + '%</b>';

    document.getElementById('fuel_co2').innerHTML =
        '&#127807; CO2 Reduction: <b>' + co2Saved.toFixed(0) + ' tons/year</b> — ' +
        'equivalent to removing <b>' + Math.round(co2Saved/4.6) + ' cars</b> from the road annually';
}



// ===== SPEED OPTIMIZATION WIZARD =====
var SPEED_DEFAULTS = {
    "VLCC Tanker":       {fuel: 75,  max: 15, hire: 55000},
    "Capesize Bulk":     {fuel: 40,  max: 14.5, hire: 28000},
    "Panamax Container": {fuel: 80,  max: 20, hire: 35000},
    "Aframax Tanker":    {fuel: 45,  max: 14, hire: 22000},
    "LNG Carrier":       {fuel: 65,  max: 19.5, hire: 80000},
    "Panamax Bulk":      {fuel: 32,  max: 14, hire: 18000},
    "Handy Bulk":        {fuel: 25,  max: 14, hire: 12000},
    "Black Sea Cargo":   {fuel: 12,  max: 12, hire: 8000}
};

function updateSpeedDefaults() {
    var vessel = document.getElementById('speed_vessel').value;
    var def = SPEED_DEFAULTS[vessel];
    if (def) {
        document.getElementById('speed_fuel_max').value = def.fuel;
        document.getElementById('speed_max').value = def.max;
        document.getElementById('speed_hire').value = def.hire;
    }
}

function calcSpeedOpt() {
    var distance  = parseFloat(document.getElementById('speed_distance').value) || 1200;
    var maxSpeed  = parseFloat(document.getElementById('speed_max').value) || 14;
    var fuelMax   = parseFloat(document.getElementById('speed_fuel_max').value) || 12;
    var bunker    = parseFloat(document.getElementById('speed_bunker').value) || 650;
    var hire      = parseFloat(document.getElementById('speed_hire').value) || 15000;
    var sshDrag   = parseFloat(document.getElementById('speed_ssh_drag').value) / 100 || 0.08;
    var euaPrice  = parseFloat(document.getElementById('speed_eua').value) || 85;

    document.getElementById('speed_results').style.display = 'block';

    var speeds = [];
    var minCost = Infinity;
    var optSpeed = maxSpeed;
    var tbody = document.getElementById('speed_table');
    tbody.innerHTML = '';

    // Hız aralığı: max hızdan %60'ına kadar
    var minSpeed = Math.max(maxSpeed * 0.6, 6);
    var step = 0.5;

    for (var s = minSpeed; s <= maxSpeed + 0.01; s += step) {
        s = Math.round(s * 2) / 2;

        // Admiralty formula: yakıt ∝ hız^3
        var fuelRatio = Math.pow(s / maxSpeed, 3);
        var fuelDay = fuelMax * fuelRatio;

        // SSH drag etkisi ekle
        fuelDay = fuelDay * (1 + sshDrag * (s / maxSpeed));

        // Voyage süresi
        var voyageDays = distance / (s * 24);

        // Maliyetler
        var fuelCost = fuelDay * voyageDays * bunker;
        var hireCost = hire * voyageDays;

        // EU ETS
        var co2 = fuelDay * voyageDays * 3.151;
        var etsCost = co2 * euaPrice * 1.09 * 0.6; // %60 EU exposure

        var totalCost = fuelCost + hireCost + etsCost;

        if (totalCost < minCost) {
            minCost = totalCost;
            optSpeed = s;
        }

        speeds.push({
            speed: s,
            voyageDays: voyageDays,
            fuelCost: fuelCost,
            hireCost: hireCost,
            etsCost: etsCost,
            totalCost: totalCost,
            co2: co2
        });
    }

    // Tablo
    var maxCostEntry = speeds[speeds.length - 1];
    speeds.forEach(function(entry) {
        var isOpt = Math.abs(entry.speed - optSpeed) < 0.01;
        var isMax = Math.abs(entry.speed - maxSpeed) < 0.01;
        var rowStyle = isOpt ? "background:rgba(243,156,18,0.1);" : "";

        function fmtK(n) { return n >= 1000 ? "$" + (n/1000).toFixed(0) + "K" : "$" + n.toFixed(0); }

        tbody.innerHTML += "<tr style='" + rowStyle + "'>" +
            "<td><b style='color:" + (isOpt ? "#F39C12" : "white") + "'>" + entry.speed.toFixed(1) + " kn" + (isOpt ? " ⚡" : "") + (isMax ? " (max)" : "") + "</b></td>" +
            "<td>" + entry.voyageDays.toFixed(1) + " days</td>" +
            "<td>" + fmtK(entry.fuelCost) + "</td>" +
            "<td>" + fmtK(entry.hireCost) + "</td>" +
            "<td>" + fmtK(entry.etsCost) + "</td>" +
            "<td style='font-weight:700;color:" + (isOpt ? "#F39C12" : "white") + "'>" + fmtK(entry.totalCost) + "</td>" +
            "<td>" + (isOpt ? "&#9889; OPTIMAL" : isMax ? "&#128308; EXPENSIVE" : "&#9898;") + "</td>" +
            "</tr>";
    });

    // Optimal banner
    document.getElementById('speed_optimal').textContent = optSpeed.toFixed(1) + " kn";
    document.getElementById('speed_optimal_reason').textContent =
        "Minimum total cost: fuel + hire + EU ETS + NASA SSH drag adjustment";

    // Savings vs max speed
    var maxEntry = speeds[speeds.length - 1];
    var saving = maxEntry.totalCost - minCost;
    var timeAdd = (distance / (optSpeed * 24)) - (distance / (maxSpeed * 24));
    var co2Save = maxEntry.co2 - speeds.find(function(e){return Math.abs(e.speed - optSpeed) < 0.01;}).co2;

    function fmtK(n) { return n >= 1000 ? "$" + (n/1000).toFixed(0) + "K" : "$" + n.toFixed(0); }

    document.getElementById('speed_saving').textContent = fmtK(saving);
    document.getElementById('speed_time_add').textContent = (timeAdd * 24).toFixed(0) + " hours";
    document.getElementById('speed_co2_save').textContent = co2Save.toFixed(0) + " t";
}


// ===== SPEED OPTIMIZATION WIZARD =====
var SPEED_DEFAULTS = {
    "VLCC Tanker":       {fuel: 75,  max: 15, hire: 55000},
    "Capesize Bulk":     {fuel: 40,  max: 14.5, hire: 28000},
    "Panamax Container": {fuel: 80,  max: 20, hire: 35000},
    "Aframax Tanker":    {fuel: 45,  max: 14, hire: 22000},
    "LNG Carrier":       {fuel: 65,  max: 19.5, hire: 80000},
    "Panamax Bulk":      {fuel: 32,  max: 14, hire: 18000},
    "Handy Bulk":        {fuel: 25,  max: 14, hire: 12000},
    "Black Sea Cargo":   {fuel: 12,  max: 12, hire: 8000}
};

function updateSpeedDefaults() {
    var vessel = document.getElementById('speed_vessel').value;
    var def = SPEED_DEFAULTS[vessel];
    if (def) {
        document.getElementById('speed_fuel_max').value = def.fuel;
        document.getElementById('speed_max').value = def.max;
        document.getElementById('speed_hire').value = def.hire;
    }
}

function calcSpeedOpt() {
    var distance  = parseFloat(document.getElementById('speed_distance').value) || 1200;
    var maxSpeed  = parseFloat(document.getElementById('speed_max').value) || 14;
    var fuelMax   = parseFloat(document.getElementById('speed_fuel_max').value) || 12;
    var bunker    = parseFloat(document.getElementById('speed_bunker').value) || 650;
    var hire      = parseFloat(document.getElementById('speed_hire').value) || 15000;
    var sshDrag   = parseFloat(document.getElementById('speed_ssh_drag').value) / 100 || 0.08;
    var euaPrice  = parseFloat(document.getElementById('speed_eua').value) || 85;

    document.getElementById('speed_results').style.display = 'block';

    var speeds = [];
    var minCost = Infinity;
    var optSpeed = maxSpeed;
    var tbody = document.getElementById('speed_table');
    tbody.innerHTML = '';

    // Hız aralığı: max hızdan %60'ına kadar
    var minSpeed = Math.max(maxSpeed * 0.6, 6);
    var step = 0.5;

    for (var s = minSpeed; s <= maxSpeed + 0.01; s += step) {
        s = Math.round(s * 2) / 2;

        // Admiralty formula: yakıt ∝ hız^3
        var fuelRatio = Math.pow(s / maxSpeed, 3);
        var fuelDay = fuelMax * fuelRatio;

        // SSH drag etkisi ekle
        fuelDay = fuelDay * (1 + sshDrag * (s / maxSpeed));

        // Voyage süresi
        var voyageDays = distance / (s * 24);

        // Maliyetler
        var fuelCost = fuelDay * voyageDays * bunker;
        var hireCost = hire * voyageDays;

        // EU ETS
        var co2 = fuelDay * voyageDays * 3.151;
        var etsCost = co2 * euaPrice * 1.09 * 0.6; // %60 EU exposure

        var totalCost = fuelCost + hireCost + etsCost;

        if (totalCost < minCost) {
            minCost = totalCost;
            optSpeed = s;
        }

        speeds.push({
            speed: s,
            voyageDays: voyageDays,
            fuelCost: fuelCost,
            hireCost: hireCost,
            etsCost: etsCost,
            totalCost: totalCost,
            co2: co2
        });
    }

    // Tablo
    var maxCostEntry = speeds[speeds.length - 1];
    speeds.forEach(function(entry) {
        var isOpt = Math.abs(entry.speed - optSpeed) < 0.01;
        var isMax = Math.abs(entry.speed - maxSpeed) < 0.01;
        var rowStyle = isOpt ? "background:rgba(243,156,18,0.1);" : "";

        function fmtK(n) { return n >= 1000 ? "$" + (n/1000).toFixed(0) + "K" : "$" + n.toFixed(0); }

        tbody.innerHTML += "<tr style='" + rowStyle + "'>" +
            "<td><b style='color:" + (isOpt ? "#F39C12" : "white") + "'>" + entry.speed.toFixed(1) + " kn" + (isOpt ? " ⚡" : "") + (isMax ? " (max)" : "") + "</b></td>" +
            "<td>" + entry.voyageDays.toFixed(1) + " days</td>" +
            "<td>" + fmtK(entry.fuelCost) + "</td>" +
            "<td>" + fmtK(entry.hireCost) + "</td>" +
            "<td>" + fmtK(entry.etsCost) + "</td>" +
            "<td style='font-weight:700;color:" + (isOpt ? "#F39C12" : "white") + "'>" + fmtK(entry.totalCost) + "</td>" +
            "<td>" + (isOpt ? "&#9889; OPTIMAL" : isMax ? "&#128308; EXPENSIVE" : "&#9898;") + "</td>" +
            "</tr>";
    });

    // Optimal banner
    document.getElementById('speed_optimal').textContent = optSpeed.toFixed(1) + " kn";
    document.getElementById('speed_optimal_reason').textContent =
        "Minimum total cost: fuel + hire + EU ETS + NASA SSH drag adjustment";

    // Savings vs max speed
    var maxEntry = speeds[speeds.length - 1];
    var saving = maxEntry.totalCost - minCost;
    var timeAdd = (distance / (optSpeed * 24)) - (distance / (maxSpeed * 24));
    var co2Save = maxEntry.co2 - speeds.find(function(e){return Math.abs(e.speed - optSpeed) < 0.01;}).co2;

    function fmtK(n) { return n >= 1000 ? "$" + (n/1000).toFixed(0) + "K" : "$" + n.toFixed(0); }

    document.getElementById('speed_saving').textContent = fmtK(saving);
    document.getElementById('speed_time_add').textContent = (timeAdd * 24).toFixed(0) + " hours";
    document.getElementById('speed_co2_save').textContent = co2Save.toFixed(0) + " t";
}

// ===== EU ETS CALCULATOR =====
var ETS_VESSEL_DEFAULTS = {
    "VLCC Tanker":       {fuel: 75,  days: 280},
    "Capesize Bulk":     {fuel: 40,  days: 280},
    "Panamax Container": {fuel: 80,  days: 300},
    "Aframax Tanker":    {fuel: 45,  days: 280},
    "LNG Carrier":       {fuel: 65,  days: 300},
    "Panamax Bulk":      {fuel: 32,  days: 280},
    "Black Sea Cargo":   {fuel: 12,  days: 280}
};

function updateETSDefaults() {
    var vessel = document.getElementById('ets_vessel').value;
    var def = ETS_VESSEL_DEFAULTS[vessel];
    if (def) {
        document.getElementById('ets_fuel').value = def.fuel;
        document.getElementById('ets_days').value = def.days;
    }
}

function calcETS() {
    var fuel      = parseFloat(document.getElementById('ets_fuel').value) || 12;
    var days      = parseFloat(document.getElementById('ets_days').value) || 280;
    var euaPrice  = parseFloat(document.getElementById('ets_price').value) || 85;
    var fx        = parseFloat(document.getElementById('ets_fx').value) || 1.09;
    var savings   = parseFloat(document.getElementById('ets_savings').value) || 10;
    var exposure  = parseFloat(document.getElementById('ets_exposure').value) / 100 || 0.6;
    var co2Factor = parseFloat(document.getElementById('ets_fuel_type').value) || 3.151;

    // Hesaplamalar
    var totalFuel    = fuel * days;
    var co2Total     = totalFuel * co2Factor;
    var co2Exposed   = co2Total * exposure;
    var etsCostEUR   = co2Exposed * euaPrice;
    var etsCostUSD   = etsCostEUR * fx;

    // Batimetrix ile
    var fuelSaved    = totalFuel * (savings / 100);
    var co2Saved     = fuelSaved * co2Factor * exposure;
    var etsSavedEUR  = co2Saved * euaPrice;
    var etsSavedUSD  = etsSavedEUR * fx;

    // Format
    function fmtEUR(n) {
        if (n >= 1000000) return '€' + (n/1000000).toFixed(2) + 'M';
        if (n >= 1000) return '€' + (n/1000).toFixed(0) + 'K';
        return '€' + n.toFixed(0);
    }
    function fmtUSD(n) {
        if (n >= 1000000) return '$' + (n/1000000).toFixed(2) + 'M';
        if (n >= 1000) return '$' + (n/1000).toFixed(0) + 'K';
        return '$' + n.toFixed(0);
    }

    document.getElementById('ets_results').style.display = 'block';
    document.getElementById('ets_total_cost').textContent = fmtEUR(etsCostEUR);
    document.getElementById('ets_saved').textContent      = fmtEUR(etsSavedEUR);
    document.getElementById('ets_co2').textContent        = Math.round(co2Total).toLocaleString() + ' t';
    document.getElementById('ets_co2_saved').textContent  = Math.round(co2Saved).toLocaleString() + ' t';

    document.getElementById('ets_breakdown').innerHTML =
        '<b style="color:var(--teal)">Annual Carbon Analysis:</b><br>' +
        'Total fuel consumption: <b>' + totalFuel.toLocaleString() + ' tonnes/year</b><br>' +
        'Total CO2 emissions: <b>' + Math.round(co2Total).toLocaleString() + ' tCO2/year</b><br>' +
        'EU ETS exposure (' + Math.round(exposure*100) + '%): <b>' + Math.round(co2Exposed).toLocaleString() + ' tCO2 covered</b><br>' +
        'EUA price: <b>€' + euaPrice + '/tonne</b><br>' +
        '<hr style="border:none;border-top:1px solid var(--line);margin:6px 0">' +
        '<b style="color:#E74C3C">Without Batimetrix:</b><br>' +
        'ETS liability: <b style="color:#E74C3C">' + fmtEUR(etsCostEUR) + ' (' + fmtUSD(etsCostUSD) + ')</b><br>' +
        '<b style="color:#27AE60">With Batimetrix (' + savings + '% drag reduction):</b><br>' +
        'Fuel saved: <b>' + Math.round(fuelSaved).toLocaleString() + ' tonnes/year</b><br>' +
        'CO2 reduced: <b>' + Math.round(co2Saved).toLocaleString() + ' tCO2/year</b><br>' +
        'ETS saving: <b style="color:#27AE60">' + fmtEUR(etsSavedEUR) + ' (' + fmtUSD(etsSavedUSD) + ')</b><br>' +
        '<hr style="border:none;border-top:1px solid var(--line);margin:6px 0">' +
        '<b style="color:#F39C12">Total Annual Benefit (Fuel + ETS):</b> ' +
        '<b style="color:#F39C12;font-size:14px">' + fmtUSD(etsSavedUSD * 1.8) + '</b>';
}

// ===== GLOBE ROUTE OPTIMIZER =====
var GLOBE_ROUTES = {
    "istanbul_trabzon": {
        name: "Istanbul → Trabzon",
        color: "#1ABC9C",
        waypoints: [
            {name:"Istanbul Bogazi", lat:41.10, lon:29.05, depth:35},
            {name:"Karadeniz Girisi", lat:41.30, lon:29.50, depth:120},
            {name:"Bati Karadeniz", lat:41.80, lon:30.50, depth:650},
            {name:"Zonguldak", lat:41.60, lon:31.80, depth:850},
            {name:"Sinop", lat:42.00, lon:35.10, depth:950},
            {name:"Samsun", lat:41.70, lon:36.20, depth:800},
            {name:"Trabzon", lat:41.00, lon:39.73, depth:200}
        ]
    },
    "istanbul_novorossiysk": {
        name: "Istanbul → Novorossiysk",
        color: "#3498DB",
        waypoints: [
            {name:"Istanbul Bogazi", lat:41.10, lon:29.05, depth:35},
            {name:"Bati KB", lat:41.80, lon:30.50, depth:650},
            {name:"Orta KB", lat:42.10, lon:33.00, depth:1100},
            {name:"Novorossiysk", lat:44.72, lon:37.77, depth:120}
        ]
    },
    "odessa_istanbul": {
        name: "Odessa → Istanbul",
        color: "#9B59B6",
        waypoints: [
            {name:"Odessa", lat:46.48, lon:30.73, depth:80},
            {name:"Bati KB", lat:44.00, lon:31.00, depth:800},
            {name:"Orta KB", lat:42.50, lon:32.00, depth:1100},
            {name:"Istanbul", lat:41.10, lon:29.05, depth:35}
        ]
    },
    "shanghai_rotterdam": {
        name: "Shanghai → Rotterdam",
        color: "#E74C3C",
        waypoints: [
            {name:"Shanghai", lat:31.23, lon:121.47, depth:15},
            {name:"Taiwan Strait", lat:24.00, lon:119.00, depth:60},
            {name:"South China Sea", lat:15.00, lon:113.00, depth:1200},
            {name:"Malacca Strait", lat:2.50, lon:101.00, depth:25},
            {name:"Indian Ocean", lat:5.00, lon:75.00, depth:3800},
            {name:"Suez Canal", lat:29.97, lon:32.55, depth:20},
            {name:"Mediterranean", lat:36.00, lon:15.00, depth:1500},
            {name:"Rotterdam", lat:51.90, lon:4.48, depth:15}
        ]
    },
    "rastanura_ningbo": {
        name: "Ras Tanura → Ningbo (VLCC)",
        color: "#F39C12",
        waypoints: [
            {name:"Ras Tanura", lat:26.70, lon:50.03, depth:30},
            {name:"Hormuz Strait", lat:26.57, lon:56.25, depth:60},
            {name:"Arabian Sea", lat:15.00, lon:65.00, depth:3500},
            {name:"Indian Ocean", lat:5.00, lon:80.00, depth:4000},
            {name:"Malacca Strait", lat:2.50, lon:101.00, depth:25},
            {name:"South China Sea", lat:15.00, lon:115.00, depth:1500},
            {name:"Ningbo", lat:29.87, lon:121.55, depth:15}
        ]
    },
    "singapore_rotterdam_cape": {
        name: "Singapore → Rotterdam (Cape)",
        color: "#27AE60",
        waypoints: [
            {name:"Singapore", lat:1.29, lon:103.85, depth:20},
            {name:"Indian Ocean", lat:-10.00, lon:80.00, depth:4000},
            {name:"Cape Good Hope", lat:-34.36, lon:18.48, depth:100},
            {name:"S.Atlantic", lat:-20.00, lon:5.00, depth:3500},
            {name:"Equator", lat:0.00, lon:-10.00, depth:3000},
            {name:"N.Atlantic", lat:30.00, lon:-20.00, depth:4000},
            {name:"Bay of Biscay", lat:45.00, lon:-5.00, depth:2000},
            {name:"Rotterdam", lat:51.90, lon:4.48, depth:15}
        ]
    },
    "murmansk_shanghai": {
        name: "Murmansk → Shanghai (Arctic NSR)",
        color: "#00BCD4",
        waypoints: [
            {name:"Murmansk", lat:68.97, lon:33.05, depth:100},
            {name:"Kara Sea", lat:75.00, lon:65.00, depth:150},
            {name:"Laptev Sea", lat:75.00, lon:120.00, depth:200},
            {name:"East Siberian Sea", lat:72.00, lon:160.00, depth:60},
            {name:"Bering Strait", lat:65.77, lon:-168.00, depth:50},
            {name:"N.Pacific", lat:50.00, lon:165.00, depth:4000},
            {name:"Japan Sea", lat:40.00, lon:135.00, depth:2000},
            {name:"Shanghai", lat:31.23, lon:121.47, depth:15}
        ]
    },
    "hormuz_transit": {
        name: "Hormuz Strait Transit",
        color: "#FF5722",
        waypoints: [
            {name:"Persian Gulf", lat:26.00, lon:52.00, depth:40},
            {name:"Hormuz Entry", lat:26.57, lon:55.00, depth:50},
            {name:"Hormuz Strait", lat:26.57, lon:56.25, depth:60},
            {name:"Oman Sea", lat:25.00, lon:58.00, depth:2000},
            {name:"Arabian Sea", lat:23.00, lon:60.00, depth:3000}
        ]
    },
    "batumi_constanta": {
        name: "Batumi → Constanta",
        color: "#FF9800",
        waypoints: [
            {name:"Batumi", lat:41.65, lon:41.64, depth:150},
            {name:"Dogu KB", lat:42.00, lon:38.00, depth:900},
            {name:"Orta KB", lat:42.20, lon:33.00, depth:1100},
            {name:"Constanta", lat:44.17, lon:28.65, depth:60}
        ]
    },
    "porthedland_qingdao": {
        name: "Port Hedland → Qingdao",
        color: "#8BC34A",
        waypoints: [
            {name:"Port Hedland", lat:-20.31, lon:118.57, depth:20},
            {name:"W.Australia", lat:-15.00, lon:115.00, depth:200},
            {name:"Java Sea", lat:-6.00, lon:110.00, depth:60},
            {name:"South China Sea", lat:10.00, lon:113.00, depth:1200},
            {name:"Taiwan Strait", lat:24.00, lon:119.00, depth:60},
            {name:"Qingdao", lat:36.07, lon:120.38, depth:20}
        ]
    }
};

var activeGlobeRoute = null;

function drawGlobeRoute() {
    var select = document.getElementById('globe_route_select');
    var routeKey = select.value;
    if (!routeKey) { clearGlobeRoute(); return; }
    
    var route = GLOBE_ROUTES[routeKey];
    if (!route) return;
    
    activeGlobeRoute = route;
    
    var info = document.getElementById('globe_route_info');
    if (info) {
        info.textContent = '&#9654; ' + route.name + ' — ' + route.waypoints.length + ' waypoints';
        info.innerHTML = '&#9654; ' + route.name + ' — ' + route.waypoints.length + ' waypoints';
    }
}

function clearGlobeRoute() {
    activeGlobeRoute = null;
    var select = document.getElementById('globe_route_select');
    if (select) select.value = '';
    var info = document.getElementById('globe_route_info');
    if (info) info.textContent = '';
}

// ===== THALASSA 3D GLOBE (GeoJSON) =====
var LAND_POLYGONS = [[[-59.57,-80.04],[-60.16,-81.0],[-64.49,-80.92],[-65.74,-80.55],[-64.04,-80.29],[-61.14,-79.98],[-59.57,-80.04]],[[-159.21,-79.5],[-162.44,-79.28],[-163.07,-78.87],[-163.71,-78.6],[-161.25,-78.38],[-159.48,-79.05]],[[-45.15,-78.05],[-43.49,-79.09],[-43.33,-80.03],[-46.51,-80.59],[-50.48,-81.03],[-54.16,-80.63],[-51.85,-79.95],[-50.36,-79.18],[-49.31,-78.46],[-48.66,-78.05],[-46.66,-77.83]],[[-121.21,-73.5],[-118.72,-73.48],[-120.23,-74.09],[-122.62,-73.66],[-122.41,-73.32]],[[-125.56,-73.48],[-124.62,-73.83],[-127.28,-73.46],[-126.56,-73.25]],[[-98.98,-71.93],[-96.79,-71.95],[-96.98,-72.44],[-99.43,-72.44],[-101.8,-72.31],[-102.33,-71.89],[-100.43,-71.85]],[[-68.45,-70.96],[-68.51,-71.8],[-69.96,-72.31],[-72.39,-72.48],[-73.07,-72.23],[-74.95,-72.07],[-73.92,-71.27],[-73.23,-71.15],[-71.78,-70.68],[-71.74,-69.51],[-70.25,-68.88],[-69.49,-69.62],[-68.73,-70.51]],[[-58.61,-64.15],[-59.79,-64.21],[-61.3,-64.54],[-62.51,-65.09],[-62.59,-65.86],[-62.81,-66.43],[-64.29,-66.84],[-65.51,-67.58],[-65.31,-68.37],[-63.96,-68.91],[-62.79,-69.62],[-62.28,-70.38],[-61.51,-71.09],[-61.08,-72.38],[-60.69,-73.17],[-61.38,-74.11],[-63.3,-74.58],[-64.35,-75.26],[-67.19,-75.79],[-69.8,-76.22],[-72.21,-76.67],[-75.56,-76.71],[-76.93,-77.1],[-74.28,-77.56],[-74.77,-78.22],[-77.93,-78.38],[-78.02,-79.18],[-76.63,-79.89],[-73.24,-80.42],[-70.01,-81.0],[-65.7,-81.47],[-61.55,-82.04],[-58.71,-82.85],[-57.01,-82.87],[-53.62,-82.26],[-49.76,-81.73],[-44.83,-81.85],[-42.16,-81.65],[-38.24,-81.34],[-34.39,-80.91],[-30.1,-80.59],[-29.25,-79.99],[-29.69,-79.26],[-33.68,-79.46],[-35.91,-79.08],[-35.33,-78.12],[-32.21,-77.65],[-29.78,-77.07],[-27.51,-76.5],[-25.47,-76.28],[-22.46,-76.11],[-20.01,-75.67],[-17.52,-75.13],[-15.7,-74.5],[-16.47,-73.87],[-15.45,-73.15],[-13.31,-72.72],[-11.51,-72.01],[-10.3,-71.27],[-8.61,-71.66],[-7.38,-71.32],[-5.79,-71.03],[-4.34,-71.46],[-1.8,-71.17],[-0.23,-71.64],[1.89,-71.13],[4.14,-70.85],[6.27,-70.46],[7.74,-69.89],[9.53,-70.01],[10.82,-70.83],[12.4,-70.25],[14.73,-70.03],[15.95,-70.03],[18.2,-69.87],[20.38,-70.01],[21.92,-70.4],[23.67,-70.52],[25.98,-70.48],[28.09,-70.32],[30.03,-69.93],[31.99,-69.66],[33.3,-68.84],[34.91,-68.66],[36.16,-69.25],[37.91,-69.52],[39.67,-69.54],[40.92,-68.93],[42.94,-68.46],[44.9,-68.05],[46.5,-67.6],[48.34,-67.37],[49.93,-67.11],[50.95,-66.52],[52.61,-66.05],[54.53,-65.82],[56.36,-65.97],[57.26,-66.68],[58.74,-67.29],[60.61,-67.68],[62.39,-68.01],[64.05,-67.41],[65.97,-67.74],[67.89,-67.93],[69.71,-68.97],[69.56,-69.68],[67.81,-70.31],[69.07,-70.68],[68.42,-71.44],[68.71,-72.17],[71.02,-72.09],[71.91,-71.32],[73.08,-70.72],[73.86,-69.87],[75.63,-69.74],[77.64,-69.46],[78.43,-68.7],[80.09,-68.07],[81.48,-67.54],[82.78,-67.21],[84.68,-67.21],[86.75,-67.15],[87.99,-66.21],[88.83,-66.95],[90.63,-67.23],[92.61,-67.19],[94.18,-67.11],[95.78,-67.39],[97.76,-67.25],[99.72,-67.25],[100.89,-66.58],[102.83,-65.56],[104.24,-65.97],[106.18,-66.93],[108.08,-66.95],[110.24,-66.7],[111.74,-66.13],[113.6,-65.88],[114.9,-66.39],[116.7,-66.66],[118.58,-67.17],[120.87,-67.19],[122.32,-66.56],[124.12,-66.62],[126.1,-66.56],[127.88,-66.66],[129.7,-66.58],[131.8,-66.39],[133.86,-66.29],[135.03,-65.72],[135.7,-65.58],[136.21,-66.45],[137.46,-66.95],[139.91,-66.88],[142.12,-66.82],[144.37,-66.84],[146.2,-67.23],[146.65,-67.9],[148.84,-68.39],[151.48,-68.72],[153.64,-68.89],[155.17,-68.84],[156.81,-69.38],[159.18,-69.6],[160.81,-70.23],[162.69,-70.74],[164.92,-70.78],[167.31,-70.83],[169.46,-71.21],[171.21,-71.7],[170.56,-72.44],[169.76,-73.24],[167.98,-73.81],[166.09,-74.38],[164.96,-75.15],[163.82,-75.87],[163.47,-76.69],[164.06,-77.46],[164.74,-78.18],[167.0,-78.75],[163.67,-79.12],[160.92,-79.73],[160.32,-80.57],[161.12,-81.28],[162.49,-82.06],[165.1,-82.71],[168.9,-83.34],[172.28,-84.04],[173.22,-84.41],[178.28,-84.47],[180,-90],[-180,-84.71],[-179.06,-84.14],[-177.14,-84.42],[-176.52,-84.23],[-176.08,-84.1],[-175.83,-84.12],[-173.12,-84.12],[-169.95,-83.88],[-168.53,-84.24],[-164.18,-84.83],[-158.07,-85.37],[-150.94,-85.3],[-145.89,-85.32],[-142.89,-84.57],[-150.06,-84.3],[-153.59,-83.69],[-153.04,-82.83],[-152.86,-82.04],[-155.29,-81.42],[-154.41,-81.16],[-150.65,-81.34],[-147.22,-80.67],[-146.77,-79.93],[-149.53,-79.36],[-153.39,-79.16],[-155.98,-78.69],[-158.05,-78.03],[-157.88,-76.99],[-155.33,-77.2],[-152.92,-77.5],[-150.0,-77.18],[-147.61,-76.58],[-146.14,-76.11],[-146.2,-75.38],[-144.32,-75.54],[-141.64,-75.09],[-138.86,-74.97],[-136.43,-74.52],[-134.43,-74.36],[-132.26,-74.3],[-129.55,-74.46],[-126.89,-74.42],[-124.01,-74.48],[-121.07,-74.52],[-118.68,-74.19],[-116.22,-74.24],[-113.94,-73.71],[-112.95,-74.38],[-111.26,-74.42],[-108.71,-74.91],[-106.15,-75.13],[-103.37,-74.99],[-100.65,-75.3],[-100.76,-74.54],[-102.55,-74.11],[-103.33,-73.36],[-102.92,-72.75],[-100.31,-72.75],[-98.12,-73.21],[-96.34,-73.62],[-93.67,-73.28],[-91.42,-73.4],[-89.23,-72.56],[-87.27,-73.19],[-85.19,-73.48],[-82.67,-73.64],[-80.69,-73.48],[-79.3,-73.52],[-76.91,-73.64],[-74.89,-73.87],[-72.83,-73.4],[-70.21,-73.15],[-67.96,-72.79],[-67.13,-72.05],[-67.56,-71.25],[-68.23,-70.46],[-68.54,-69.72],[-67.98,-68.95],[-67.43,-68.15],[-67.74,-67.33],[-66.7,-66.58],[-65.37,-65.9],[-64.18,-65.17],[-63.0,-64.64],[-61.41,-64.27],[-59.89,-63.96],[-58.59,-63.39],[-57.22,-63.53],[-58.61,-64.15]],[[-67.75,-53.85],[-65.05,-54.7],[-66.45,-55.25],[-67.29,-55.3],[-69.23,-55.5],[-71.01,-55.05],[-73.29,-53.96],[-73.84,-53.05],[-71.11,-54.07],[-70.27,-52.93],[-68.63,-52.64],[-68.25,-53.1]],[[-58.55,-51.1],[-58.05,-51.9],[-59.85,-51.85],[-61.2,-51.85],[-59.15,-51.5]],[[70.28,-49.71],[68.72,-49.24],[68.94,-48.62],[70.53,-49.06],[70.28,-49.71]],[[145.4,-40.79],[146.91,-41.0],[148.29,-40.88],[148.02,-42.41],[147.56,-42.94],[146.66,-43.58],[145.43,-42.69],[144.72,-41.16],[145.4,-40.79]],[[173.02,-40.92],[173.96,-40.93],[174.25,-41.77],[173.22,-42.97],[173.08,-43.85],[171.45,-44.24],[170.62,-45.91],[169.33,-46.64],[167.76,-46.29],[166.51,-45.85],[168.3,-44.12],[169.67,-43.56],[171.13,-42.51],[171.95,-41.51],[172.8,-40.49]],[[174.61,-36.16],[175.36,-36.53],[175.96,-37.56],[177.44,-37.96],[178.52,-37.7],[177.97,-39.17],[176.94,-39.45],[176.89,-40.07],[176.01,-41.29],[175.07,-41.43],[175.23,-40.46],[173.82,-39.51],[174.57,-38.8],[174.7,-37.38],[174.32,-36.53],[173.05,-35.24],[173.01,-34.45],[174.33,-35.27]],[[167.12,-22.16],[166.19,-22.13],[164.83,-21.15],[164.03,-20.11],[165.02,-20.46],[165.78,-21.08],[167.12,-22.16]],[[178.37,-17.34],[178.55,-18.15],[177.38,-18.16],[177.67,-17.38],[178.37,-17.34]],[[179.36,-16.8],[178.6,-16.64],[179.41,-16.38],[180,-16.56]],[[-179.92,-16.5],[-180,-16.07],[-179.92,-16.5]],[[167.84,-16.47],[167.18,-16.16],[167.84,-16.47]],[[167.11,-14.93],[167.0,-15.61],[166.65,-15.39],[167.11,-14.93]],[[50.06,-13.56],[50.48,-15.23],[50.2,-16.0],[49.67,-15.71],[49.77,-16.88],[49.44,-17.95],[48.55,-20.5],[47.55,-23.78],[46.28,-25.18],[44.83,-25.35],[43.76,-24.46],[43.35,-22.78],[43.43,-21.34],[43.9,-20.83],[44.46,-19.44],[44.04,-18.33],[44.31,-16.85],[44.94,-16.18],[45.87,-15.79],[46.88,-15.21],[48.01,-14.09],[48.29,-13.78],[48.86,-12.49],[49.54,-12.47],[50.06,-13.56]],[[143.56,-13.76],[144.56,-14.17],[145.37,-14.98],[145.49,-16.29],[145.89,-16.91],[146.06,-18.28],[147.47,-19.48],[148.85,-20.39],[149.29,-21.26],[150.08,-22.12],[150.73,-22.4],[151.61,-24.08],[152.86,-25.27],[153.16,-26.64],[153.57,-28.11],[153.34,-29.46],[153.09,-30.92],[152.45,-32.55],[151.34,-33.82],[150.71,-35.17],[150.08,-36.42],[150.0,-37.43],[148.3,-37.81],[146.92,-38.61],[145.49,-38.59],[145.03,-37.9],[143.61,-38.81],[142.18,-38.38],[140.64,-38.02],[139.81,-36.64],[139.08,-35.73],[138.45,-35.13],[137.72,-35.08],[137.35,-34.71],[137.89,-33.64],[137.0,-33.75],[135.99,-34.89],[135.24,-33.95],[134.09,-32.85],[132.99,-32.01],[131.33,-31.5],[128.24,-31.95],[126.15,-32.22],[124.22,-32.96],[123.66,-33.89],[122.18,-34.0],[120.58,-33.93],[119.3,-34.51],[118.51,-34.75],[117.3,-35.03],[115.56,-34.39],[115.05,-33.62],[115.71,-33.26],[115.8,-32.21],[115.16,-30.6],[115.04,-29.46],[114.62,-28.52],[114.05,-27.33],[113.34,-26.12],[113.44,-25.62],[114.23,-26.3],[113.72,-25.0],[113.39,-24.38],[113.71,-23.56],[113.74,-22.48],[114.23,-22.52],[115.46,-21.5],[116.71,-20.7],[117.44,-20.75],[118.84,-20.26],[119.25,-19.95],[120.86,-19.68],[121.66,-18.71],[122.29,-17.8],[123.01,-16.41],[123.86,-17.07],[123.82,-16.11],[124.38,-15.57],[125.17,-14.68],[125.69,-14.23],[126.14,-14.1],[127.07,-13.82],[128.36,-14.87],[129.62,-14.97],[129.89,-13.62],[130.18,-13.11],[131.22,-12.18],[132.58,-12.11],[131.82,-11.27],[133.02,-11.38],[134.39,-12.04],[135.3,-12.25],[136.26,-12.05],[136.95,-12.35],[136.31,-13.29],[136.08,-13.72],[135.43,-14.72],[136.3,-15.55],[137.58,-16.22],[138.59,-16.81],[139.26,-17.37],[140.88,-17.37],[141.27,-16.39],[141.7,-15.04],[141.64,-14.27],[141.65,-12.94],[141.69,-12.41],[142.12,-11.33],[142.52,-10.67],[142.87,-11.78],[143.16,-12.33],[143.6,-13.4]],[[162.12,-10.48],[161.7,-10.82],[161.92,-10.45]],[[120.72,-10.24],[118.97,-9.56],[120.43,-9.67],[120.72,-10.24]],[[160.85,-9.87],[159.85,-9.79],[159.7,-9.24],[160.69,-9.61]],[[161.68,-9.6],[160.79,-8.92],[160.92,-8.32],[161.68,-9.6]],[[124.44,-10.14],[123.46,-10.24],[123.98,-9.29],[125.09,-8.66],[126.64,-8.4],[127.34,-8.4],[125.93,-9.11],[124.44,-10.14]],[[117.9,-8.1],[118.88,-8.28],[117.97,-8.91],[116.74,-9.03],[117.63,-8.45]],[[122.9,-8.09],[121.25,-8.93],[119.92,-8.44],[121.34,-8.54],[122.9,-8.09]],[[159.88,-8.34],[159.13,-8.11],[158.21,-7.42],[158.82,-7.56],[159.88,-8.34]],[[157.54,-7.35],[156.9,-7.18],[156.54,-6.6],[157.54,-7.35]],[[108.62,-6.78],[110.76,-6.47],[112.98,-7.59],[115.71,-8.37],[113.46,-8.35],[111.52,-8.3],[109.43,-7.74],[108.28,-7.77],[106.28,-6.92],[106.05,-5.9],[108.07,-6.35],[108.62,-6.78]],[[134.72,-6.21],[134.11,-6.14],[134.5,-5.45],[134.72,-6.21]],[[155.88,-6.82],[155.17,-6.54],[154.51,-5.14],[154.76,-5.34],[155.55,-6.2],[155.88,-6.82]],[[151.98,-5.48],[151.3,-5.84],[150.24,-6.32],[148.89,-6.03],[148.4,-5.44],[149.85,-5.51],[150.14,-5.0],[150.81,-5.46],[151.65,-4.76],[152.14,-4.15],[152.32,-4.87]],[[127.25,-3.46],[126.18,-3.61],[127.0,-3.13]],[[130.47,-3.09],[129.99,-3.45],[128.59,-3.43],[128.14,-2.84],[130.47,-3.09]],[[153.14,-4.5],[152.64,-4.18],[151.95,-3.46],[150.66,-2.74],[151.48,-2.78],[152.24,-3.24],[153.02,-3.98]],[[134.14,-1.15],[135.46,-3.37],[137.44,-1.7],[139.18,-2.05],[141.0,-2.6],[144.58,-3.86],[145.83,-4.88],[147.65,-6.08],[146.97,-6.72],[148.08,-8.04],[149.31,-9.07],[150.04,-9.68],[150.8,-10.29],[150.03,-10.65],[148.92,-10.28],[147.14,-9.49],[146.05,-8.07],[143.9,-7.92],[143.41,-8.98],[142.07,-9.16],[140.14,-8.3],[138.88,-8.38],[138.04,-7.6],[138.41,-6.23],[135.99,-4.55],[133.66,-3.54],[132.98,-4.11],[132.75,-3.31],[133.07,-2.46],[133.7,-2.21],[131.84,-1.62],[130.52,-0.94],[132.38,-0.37],[134.14,-1.15]],[[125.24,1.42],[123.69,0.24],[121.06,0.38],[120.04,-0.52],[121.48,-0.96],[123.26,-1.08],[122.39,-1.52],[122.45,-3.19],[123.17,-4.68],[122.63,-5.63],[122.72,-4.46],[121.49,-4.57],[120.9,-3.6],[120.31,-2.93],[120.43,-5.53],[119.37,-5.38],[119.5,-3.49],[118.77,-2.8],[119.32,-1.35],[120.04,0.57],[121.67,1.01],[124.08,0.92],[125.24,1.42]],[[128.69,1.13],[128.12,0.36],[128.38,-0.78],[127.7,-0.27],[127.6,1.81],[128.0,1.63],[128.69,1.13]],[[105.82,-5.85],[103.87,-5.04],[102.16,-3.61],[100.9,-2.05],[99.26,0.18],[98.6,1.82],[97.18,3.31],[95.38,4.97],[95.94,5.44],[98.37,4.27],[99.69,3.17],[101.66,2.08],[103.08,0.56],[103.44,-0.71],[104.37,-1.08],[104.89,-2.34],[106.11,-3.06],[105.82,-5.85]],[[117.88,1.83],[117.81,0.78],[117.52,-0.8],[116.53,-2.48],[116.0,-3.66],[114.47,-3.5],[113.26,-3.12],[111.7,-2.99],[110.22,-2.93],[109.57,-1.31],[108.95,0.42],[109.66,2.01],[111.17,1.85],[111.8,2.89],[113.71,3.89],[114.6,4.9],[116.22,6.14],[117.13,6.93],[117.69,5.99],[119.18,5.41],[118.44,4.97],[117.88,4.14],[118.05,2.29]],[[126.38,8.41],[126.54,7.19],[125.83,7.29],[125.68,6.05],[124.22,6.16],[124.24,7.36],[123.3,7.42],[122.09,6.9],[122.31,8.03],[123.49,8.69],[124.6,8.51],[125.47,8.99],[126.22,9.29],[126.38,8.41]],[[81.22,6.2],[79.87,6.76],[80.15,9.82],[81.3,8.56],[81.64,6.48]],[[-60.94,10.11],[-61.95,10.09],[-61.68,10.76],[-60.9,10.86]],[[123.98,10.28],[123.31,9.32],[122.38,9.71],[122.84,10.26],[123.5,10.94],[124.08,11.23]],[[118.5,9.32],[117.66,9.07],[118.99,10.38],[119.69,10.55],[118.5,9.32]],[[121.88,11.89],[123.12,11.58],[122.64,10.74],[121.97,10.91],[121.88,11.89]],[[125.5,12.16],[125.01,11.31],[125.28,10.36],[124.76,10.84],[124.3,11.5],[124.88,11.79],[125.23,12.54]],[[121.53,13.07],[120.83,12.7],[121.18,13.43]],[[121.32,18.5],[122.25,18.48],[122.17,17.81],[122.25,16.26],[121.51,15.12],[122.26,14.22],[123.95,13.78],[124.18,13.0],[123.3,13.03],[122.67,13.19],[121.13,13.64],[120.68,14.27],[120.69,14.76],[120.07,14.97],[119.88,16.36],[120.39,17.6],[121.32,18.5]],[[-65.59,18.23],[-66.6,17.98],[-67.24,18.37],[-66.28,18.51],[-65.59,18.23]],[[-76.9,17.87],[-77.77,17.86],[-78.22,18.45],[-77.57,18.49],[-76.37,18.16],[-76.9,17.87]],[[-72.58,19.87],[-71.59,19.88],[-70.21,19.62],[-69.77,19.29],[-69.25,19.02],[-68.32,18.61],[-69.16,18.42],[-69.95,18.43],[-70.52,18.18],[-71.0,18.28],[-71.66,17.76],[-72.37,18.21],[-73.45,18.22],[-74.46,18.34],[-73.45,18.53],[-72.33,18.67],[-72.78,19.48],[-73.19,19.92]],[[110.34,18.68],[108.66,18.51],[109.12,19.82],[110.79,20.08],[110.57,19.26]],[[-155.54,19.08],[-155.94,19.06],[-156.07,19.7],[-155.85,19.98],[-155.86,20.27],[-155.4,20.08],[-155.06,19.86],[-154.83,19.45],[-155.54,19.08]],[[-156.08,20.64],[-156.59,20.78],[-156.71,20.93],[-156.26,20.92],[-156.08,20.64]],[[-156.76,21.18],[-157.33,21.1],[-156.76,21.18]],[[-157.65,21.32],[-157.78,21.28],[-158.25,21.54],[-158.03,21.72],[-157.65,21.32]],[[-159.35,21.98],[-159.8,22.07],[-159.6,22.24],[-159.35,21.98]],[[-79.68,22.77],[-78.35,22.51],[-77.15,21.66],[-76.19,21.22],[-75.67,20.74],[-74.18,20.28],[-74.96,19.92],[-76.32,19.95],[-77.09,20.41],[-78.14,20.74],[-78.72,21.6],[-80.22,21.83],[-81.82,22.19],[-81.8,22.64],[-83.49,22.17],[-84.05,21.91],[-84.97,21.9],[-84.23,22.57],[-83.27,22.98],[-82.27,23.19],[-80.62,23.11]],[[-77.53,23.76],[-78.03,24.29],[-78.19,25.21],[-77.54,24.34]],[[121.18,22.79],[120.22,22.81],[120.69,24.54],[121.95,25.0],[121.18,22.79]],[[-77.82,26.58],[-78.98,26.79],[-77.85,26.84]],[[-77,26.59],[-77.36,26.01],[-77.79,26.93],[-77,26.59]],[[134.64,34.15],[134.2,33.2],[133.28,33.29],[132.36,32.99],[132.92,34.06],[133.9,34.36]],[[34.58,35.67],[33.97,35.06],[32.98,34.57],[32.26,35.1],[32.8,35.15],[33.67,35.37]],[[23.7,35.71],[25.03,35.42],[25.75,35.18],[26.16,35.0],[24.74,35.08],[23.7,35.71]],[[15.52,38.23],[15.31,37.13],[14.34,37.0],[12.43,37.61],[13.74,38.03],[15.52,38.23]],[[9.21,41.21],[9.67,39.18],[8.81,38.91],[8.39,40.38],[8.71,40.9]],[[140.98,37.14],[140.77,35.84],[138.98,34.67],[135.79,33.46],[135.08,34.6],[132.16,33.9],[132.0,33.15],[130.69,31.03],[130.45,32.32],[129.41,33.3],[130.88,34.23],[132.62,35.43],[135.68,35.53],[137.39,36.83],[139.43,38.22],[139.88,40.56],[141.37,41.38],[141.88,39.18],[140.98,37.14]],[[9.56,42.15],[8.78,41.58],[8.75,42.63],[9.56,42.15]],[[143.91,44.17],[145.32,44.38],[144.06,42.99],[141.61,42.68],[139.96,41.57],[140.31,43.33],[141.67,44.77],[143.14,44.51]],[[-63.66,46.55],[-62.01,46.44],[-62.87,45.97],[-64.39,46.73],[-63.66,46.55]],[[-61.81,49.11],[-63.59,49.4],[-64.17,49.96],[-61.84,49.29]],[[-123.51,48.51],[-125.66,48.83],[-126.85,49.53],[-128.06,49.99],[-128.36,50.77],[-126.7,50.4],[-125.42,49.95],[-123.92,49.06]],[[-56.13,50.69],[-56.14,50.15],[-55.82,49.59],[-54.47,49.56],[-53.79,48.52],[-52.96,48.16],[-53.07,46.66],[-54.18,46.81],[-54.24,47.75],[-56.0,46.92],[-56.25,47.63],[-59.27,47.6],[-58.8,48.25],[-58.39,49.13],[-56.74,51.29],[-55.41,51.59],[-56.13,50.69]],[[-132.71,54.04],[-132.71,54.04],[-131.75,54.12],[-131.18,52.18],[-132.18,52.64],[-133.05,53.41],[-133.18,54.17]],[[143.65,50.75],[143.17,49.31],[143.53,46.84],[142.75,46.74],[141.91,46.81],[141.9,48.86],[142.18,50.95],[141.68,53.3],[142.21,54.23],[142.91,53.7],[143.24,51.76]],[[-6.79,52.26],[-9.98,51.82],[-9.69,53.88],[-7.57,55.13],[-5.66,54.55],[-6.03,53.15]],[[12.69,55.61],[11.04,55.36],[12.37,56.11]],[[-153.01,57.12],[-154.52,56.99],[-153.76,57.82],[-152.56,57.9],[-153.01,57.12]],[[-3.01,58.63],[-3.06,57.69],[-2.22,56.87],[-2.09,55.91],[-0.43,54.46],[0.47,52.93],[1.56,52.1],[1.45,51.29],[-0.79,50.77],[-2.96,50.7],[-4.54,50.34],[-5.78,50.16],[-3.41,51.43],[-5.27,51.99],[-4.77,52.84],[-3.09,53.4],[-3.63,54.62],[-5.08,55.06],[-5.05,55.78],[-5.64,56.28],[-5.79,57.82],[-4.21,58.55]],[[-165.58,59.91],[-166.85,59.94],[-166.47,60.38],[-165.58,59.91]],[[-79.27,62.16],[-80.1,61.72],[-80.32,62.09],[-79.52,62.36]],[[-81.9,62.71],[-83.77,62.18],[-83.25,62.91],[-81.9,62.71]],[[-171.73,63.78],[-170.49,63.69],[-168.69,63.3],[-169.53,62.98],[-170.67,63.38],[-171.79,63.41]],[[-85.16,65.66],[-84.46,65.37],[-82.79,64.77],[-81.55,63.98],[-80.1,63.73],[-82.55,63.65],[-84.1,63.57],[-85.87,63.64],[-86.35,64.04],[-85.88,65.74]],[[-14.51,66.46],[-13.61,65.13],[-17.79,63.68],[-19.97,63.64],[-21.78,64.4],[-22.18,65.08],[-24.33,65.61],[-22.13,66.41],[-19.06,66.28],[-16.17,66.53]],[[-75.87,67.15],[-77.24,67.59],[-75.9,68.29],[-75.1,67.58],[-75.87,67.15]],[[-175.01,66.58],[-174.57,67.06],[-169.9,65.98],[-172.53,65.44],[-172.96,64.25],[-174.65,64.63],[-176.21,65.36],[-178.36,65.39],[-178.69,66.11],[-179.43,65.4],[-180,68.96],[-174.93,67.21]],[[-95.65,69.11],[-97.62,69.06],[-99.8,69.4],[-98.22,70.14],[-96.56,69.68],[-95.65,69.11]],[[180,70.83],[178.73,71.1],[180,70.83]],[[-178.69,70.89],[-180,71.52],[-179.02,71.56],[-177.66,71.13]],[[-90.55,69.5],[-89.22,69.26],[-88.32,67.87],[-86.31,67.92],[-85.52,69.88],[-82.62,69.66],[-81.22,68.67],[-81.26,67.6],[-83.34,66.41],[-85.77,66.56],[-87.03,65.21],[-88.48,64.1],[-90.7,63.61],[-91.93,62.84],[-94.24,60.9],[-94.68,58.95],[-92.76,57.85],[-90.9,57.28],[-88.04,56.47],[-86.07,55.72],[-83.36,55.24],[-82.44,54.28],[-81.4,52.16],[-79.14,51.53],[-79.12,54.14],[-78.23,55.14],[-76.54,56.53],[-77.3,58.05],[-77.34,59.85],[-78.11,62.32],[-75.7,62.28],[-73.84,62.44],[-71.68,61.53],[-69.59,61.06],[-69.29,58.96],[-67.65,58.21],[-65.25,59.87],[-63.8,59.44],[-61.4,56.97],[-60.47,55.78],[-57.98,54.95],[-56.94,53.78],[-55.76,53.27],[-56.41,51.77],[-58.77,51.06],[-61.72,50.08],[-65.36,50.3],[-67.24,49.51],[-69.95,47.74],[-70.26,46.99],[-66.55,49.13],[-64.17,48.74],[-64.8,46.99],[-63.17,45.74],[-60.52,47.01],[-59.8,45.92],[-63.25,44.67],[-65.36,43.55],[-66.16,44.47],[-66.03,45.26],[-66.96,44.81],[-69.06,43.98],[-70.69,43.03],[-70.83,42.34],[-70.08,41.78],[-69.88,41.92],[-70.64,41.48],[-71.86,41.32],[-72.88,41.22],[-72.24,41.12],[-73.34,40.63],[-73.95,40.75],[-73.96,40.43],[-74.91,38.94],[-75.2,39.25],[-75.32,38.96],[-75.06,38.4],[-75.94,37.22],[-75.72,37.94],[-76.35,39.15],[-76.33,38.08],[-76.3,37.92],[-75.97,36.9],[-75.73,35.55],[-77.4,34.51],[-78.55,33.86],[-79.2,33.16],[-80.86,32.03],[-81.49,30.73],[-80.98,29.18],[-80.53,28.04],[-80.09,26.21],[-80.38,25.21],[-81.17,25.2],[-81.71,25.87],[-82.71,27.5],[-82.65,28.55],[-83.71,29.94],[-85.11,29.64],[-85.77,30.15],[-87.53,30.27],[-89.18,30.32],[-89.41,29.89],[-89.22,29.29],[-89.78,29.31],[-90.88,29.15],[-92.5,29.55],[-93.85,29.71],[-95.6,28.74],[-97.14,27.83],[-97.38,26.69],[-97.14,25.87],[-97.14,25.87],[-97.7,24.27],[-97.87,22.44],[-97.39,21.41],[-96.53,19.89],[-95.9,18.83],[-94.43,18.14],[-92.79,18.52],[-91.41,18.88],[-90.53,19.87],[-90.28,21.0],[-88.54,21.49],[-87.05,21.54],[-86.85,20.85],[-87.62,19.65],[-87.59,19.04],[-88.09,18.52],[-88.3,18.35],[-88.12,18.08],[-88.2,17.49],[-88.24,17.04],[-88.55,16.27],[-88.93,15.89],[-88.52,15.86],[-88.12,15.69],[-87.62,15.88],[-87.37,15.85],[-86.44,15.78],[-86.0,16.01],[-85.44,15.89],[-84.98,16.0],[-84.37,15.84],[-83.77,15.42],[-83.15,15.0],[-83.28,14.68],[-83.41,13.97],[-83.55,13.13],[-83.47,12.42],[-83.72,11.89],[-83.86,11.37],[-83.66,10.94],[-83.02,9.99],[-82.19,9.21],[-81.81,8.95],[-81.44,8.79],[-80.52,9.11],[-79.57,9.61],[-79.06,9.45],[-78.06,9.25],[-77.35,8.67],[-76.09,9.34],[-75.66,9.77],[-74.91,11.08],[-74.2,11.31],[-72.63,11.73],[-71.75,12.44],[-71.14,12.11],[-71.36,11.54],[-71.62,10.97],[-72.07,9.87],[-71.26,9.14],[-71.35,10.21],[-70.16,11.38],[-69.94,12.16],[-68.88,11.44],[-68.19,10.55],[-66.23,10.65],[-64.89,10.08],[-64.32,10.64],[-61.88,10.72],[-62.39,9.95],[-60.83,9.38],[-60.15,8.6],[-59.1,8.0],[-58.45,6.83],[-57.54,6.32],[-55.95,5.77],[-55.03,6.03],[-53.62,5.65],[-51.82,4.57],[-51.32,4.2],[-50.51,1.9],[-49.95,1.05],[-50.39,-0.08],[-48.58,-1.24],[-46.57,-0.94],[-44.42,-2.14],[-43.42,-2.38],[-39.98,-2.87],[-37.22,-4.82],[-35.6,-5.15],[-34.9,-6.74],[-35.13,-9.0],[-37.05,-11.04],[-38.42,-13.04],[-38.95,-13.79],[-39.16,-17.21],[-39.58,-18.26],[-40.77,-20.9],[-41.75,-22.37],[-43.07,-22.97],[-45.35,-23.8],[-47.65,-24.89],[-48.64,-26.62],[-48.66,-28.19],[-49.59,-29.22],[-51.58,-31.78],[-52.71,-33.2],[-53.81,-34.4],[-55.67,-34.75],[-57.14,-34.43],[-58.43,-33.91],[-57.23,-35.29],[-56.74,-36.41],[-57.75,-38.18],[-61.24,-38.93],[-62.13,-39.42],[-62.15,-40.68],[-63.77,-41.17],[-65.12,-41.06],[-64.3,-42.36],[-63.46,-42.56],[-65.18,-43.5],[-65.57,-45.04],[-67.29,-45.55],[-66.6,-47.03],[-65.99,-48.13],[-67.82,-49.87],[-69.14,-50.73],[-68.15,-52.35],[-69.46,-52.29],[-70.85,-52.9],[-71.43,-53.86],[-73.7,-52.84],[-75.26,-51.63],[-75.48,-50.38],[-75.18,-47.71],[-75.64,-46.65],[-74.35,-44.1],[-72.72,-42.38],[-73.7,-43.37],[-74.02,-41.79],[-73.22,-39.26],[-73.59,-37.16],[-72.55,-35.51],[-71.44,-32.42],[-71.37,-30.1],[-70.91,-27.64],[-70.4,-23.63],[-70.16,-19.76],[-71.38,-17.77],[-73.44,-16.36],[-76.01,-14.65],[-76.26,-13.53],[-78.09,-10.38],[-79.45,-7.93],[-80.54,-6.54],[-80.93,-5.69],[-81.1,-4.04],[-79.77,-2.66],[-80.37,-2.69],[-80.76,-1.97],[-80.58,-0.91],[-80.02,0.36],[-79.54,0.98],[-78.99,1.69],[-78.66,2.27],[-77.93,2.7],[-77.13,3.85],[-77.31,4.67],[-77.32,5.85],[-77.88,7.22],[-78.43,8.05],[-78.44,8.39],[-79.12,9.0],[-79.76,8.58],[-80.38,8.3],[-80.0,7.55],[-80.42,7.27],[-81.06,7.82],[-81.52,7.71],[-82.13,8.18],[-82.82,8.29],[-82.97,8.23],[-83.71,8.66],[-83.63,9.05],[-84.3,9.49],[-84.71,9.91],[-84.91,9.8],[-85.34,9.83],[-85.8,10.13],[-85.66,10.75],[-85.71,11.09],[-86.53,11.81],[-87.17,12.46],[-87.56,13.06],[-87.32,12.98],[-87.79,13.38],[-88.48,13.16],[-89.26,13.46],[-90.1,13.74],[-91.23,13.93],[-92.23,14.54],[-93.88,15.94],[-95.25,16.13],[-96.56,15.65],[-98.01,16.11],[-99.7,16.71],[-101.67,17.65],[-102.48,17.98],[-103.92,18.75],[-105.49,19.95],[-105.4,20.53],[-105.27,21.08],[-105.6,21.87],[-106.03,22.77],[-107.92,24.55],[-109.26,25.58],[-109.29,26.44],[-110.39,27.16],[-111.18,27.94],[-112.23,28.95],[-112.81,30.02],[-113.15,31.17],[-114.21,31.52],[-114.94,31.39],[-114.67,30.16],[-113.59,29.06],[-113.27,28.75],[-112.96,28.43],[-112.46,27.53],[-111.62,26.66],[-110.99,25.29],[-110.66,24.3],[-109.77,23.81],[-109.43,23.19],[-110.03,22.82],[-110.95,24.0],[-112.18,24.74],[-112.3,26.01],[-113.46,26.77],[-113.85,26.9],[-115.06,27.72],[-114.57,27.74],[-114.16,28.57],[-115.52,29.56],[-116.26,30.84],[-117.13,32.54],[-117.94,33.62],[-118.52,34.03],[-119.44,34.35],[-120.62,34.61],[-121.71,36.16],[-122.51,37.78],[-123.73,38.95],[-124.4,40.31],[-124.21,42.0],[-124.14,43.71],[-124.08,46.86],[-124.69,48.18],[-123.12,48.04],[-122.34,47.36],[-122.84,49.0],[-124.91,49.98],[-127.44,50.83],[-127.85,52.33],[-129.31,53.56],[-130.54,54.8],[-131.97,55.5],[-133.54,57.18],[-135.04,58.19],[-137.8,58.5],[-140.83,59.73],[-143.96,60.0],[-147.11,60.88],[-148.02,59.98],[-149.73,59.71],[-151.72,59.16],[-151.41,60.73],[-150.62,61.28],[-152.58,60.06],[-153.29,58.86],[-155.31,57.73],[-156.56,56.98],[-158.43,55.99],[-160.29,55.64],[-162.24,55.02],[-164.79,54.4],[-163.85,55.04],[-161.8,55.9],[-160.07,56.42],[-158.46,57.22],[-157.55,58.33],[-158.19,58.62],[-159.06,58.42],[-159.98,58.57],[-161.35,58.67],[-162.05,59.27],[-162.52,59.99],[-164.66,60.27],[-165.35,61.07],[-165.73,62.08],[-164.56,63.15],[-163.07,63.06],[-161.53,63.46],[-160.96,64.22],[-160.78,64.79],[-162.45,64.56],[-163.55,64.56],[-166.43,64.69],[-168.11,65.67],[-164.47,66.58],[-163.79,66.08],[-162.49,66.74],[-164.43,67.62],[-166.76,68.36],[-164.43,68.92],[-162.93,69.86],[-160.93,70.45],[-158.12,70.82],[-155.07,71.15],[-153.9,70.89],[-152.27,70.6],[-149.72,70.53],[-145.69,70.12],[-143.59,70.15],[-140.99,69.71],[-137.55,68.99],[-135.63,69.32],[-132.93,69.51],[-129.79,70.19],[-128.36,70.01],[-127.45,70.38],[-124.42,70.16],[-123.06,69.56],[-121.47,69.8],[-117.6,69.01],[-115.25,68.91],[-115.3,67.9],[-110.8,67.81],[-108.88,67.38],[-108.81,68.31],[-106.95,68.7],[-105.34,68.56],[-103.22,68.1],[-99.9,67.81],[-98.56,68.4],[-96.12,68.24],[-95.49,68.09],[-94.23,69.07],[-96.47,70.09],[-95.21,71.92],[-92.88,71.32],[-92.41,69.7]],[[-114.17,73.12],[-112.44,72.96],[-109.92,72.96],[-108.19,71.65],[-108.4,73.09],[-106.52,73.08],[-104.77,71.7],[-102.79,70.5],[-101.09,69.58],[-102.09,69.12],[-104.24,68.91],[-107.12,69.12],[-111.97,68.6],[-113.85,69.01],[-116.11,69.17],[-116.67,70.07],[-113.72,70.19],[-114.35,70.6],[-117.9,70.54],[-116.11,71.31],[-119.4,71.56],[-117.87,72.71],[-114.17,73.12]],[[-104.5,73.42],[-106.94,73.46],[-105.26,73.64]],[[-76.34,73.1],[-77.31,72.86],[-79.49,72.74],[-80.88,73.33],[-80.35,73.76],[-76.34,73.1]],[[-86.56,73.16],[-84.85,73.34],[-80.6,72.72],[-78.77,72.35],[-75.61,72.24],[-74.1,71.33],[-71.2,70.92],[-67.91,70.12],[-68.81,68.72],[-64.86,67.85],[-61.85,66.86],[-63.92,65.0],[-66.72,66.39],[-68.14,65.69],[-65.73,64.65],[-64.67,63.39],[-66.28,62.95],[-67.37,62.88],[-66.17,61.93],[-71.02,62.91],[-71.89,63.68],[-74.83,64.68],[-77.71,64.23],[-77.9,65.31],[-73.96,65.45],[-73.94,66.31],[-72.93,67.73],[-74.84,68.55],[-76.23,69.15],[-78.17,69.83],[-79.49,69.87],[-84.94,69.97],[-88.68,70.41],[-88.47,71.22],[-90.21,72.24],[-88.41,73.54],[-86.56,73.16]],[[-100.36,73.84],[-97.38,73.76],[-98.05,72.99],[-96.72,71.66],[-99.32,71.36],[-102.5,72.51],[-100.44,72.71],[-100.36,73.84]],[[143.6,73.21],[140.04,73.32],[140.81,73.77],[143.48,73.48]],[[-93.2,72.77],[-95.41,72.06],[-96.02,73.44],[-94.5,74.13],[-90.51,73.86],[-93.2,72.77]],[[-120.46,71.4],[-123.62,71.34],[-125.59,72.19],[-123.94,73.68],[-121.54,74.45],[-117.56,74.19],[-115.51,73.48],[-119.22,72.52],[-120.46,71.4]],[[150.73,75.08],[147.98,74.78],[146.36,75.5],[150.73,75.08]],[[-93.61,74.98],[-95.61,74.67],[-96.29,75.38],[-93.98,75.3]],[[145.09,75.56],[140.61,74.85],[136.97,75.26],[138.83,76.14],[145.09,75.56]],[[-98.5,76.72],[-97.7,75.74],[-99.81,74.9],[-100.86,75.64],[-102.57,76.34],[-99.98,76.65],[-98.5,76.72]],[[-108.21,76.2],[-106.93,76.01],[-105.7,75.48],[-109.7,74.85],[-113.74,74.39],[-111.79,75.16],[-117.71,75.22],[-115.4,76.48],[-110.81,75.55],[-110.5,76.43],[-108.55,76.68]],[[57.54,70.72],[53.68,70.76],[51.6,71.47],[52.48,72.23],[54.43,73.63],[55.9,74.63],[57.87,75.61],[64.5,76.44],[68.16,76.94],[68.18,76.23],[61.58,75.26],[56.99,73.33],[55.62,71.54]],[[-94.68,77.1],[-91.61,76.78],[-90.97,76.07],[-89.19,75.61],[-86.38,75.48],[-82.75,75.78],[-80.06,75.34],[-80.46,74.66],[-83.23,74.56],[-88.15,74.39],[-92.42,74.84],[-92.89,75.88],[-95.96,76.44],[-96.75,77.16]],[[-116.2,77.65],[-117.11,76.53],[-119.9,76.05],[-122.85,76.12],[-121.16,76.86],[-117.57,77.5]],[[106.97,76.97],[108.15,76.72],[113.33,76.22],[113.89,75.33],[110.15,74.48],[110.64,74.04],[113.02,73.98],[113.97,73.59],[118.78,73.59],[123.2,72.97],[125.38,73.56],[128.59,73.04],[128.46,71.98],[131.29,70.79],[133.86,71.39],[137.5,71.35],[139.87,71.49],[140.47,72.85],[150.35,71.61],[157.01,71.03],[159.83,70.45],[160.94,69.44],[164.05,69.67],[167.84,69.58],[170.82,69.01],[170.45,70.1],[175.72,69.88],[180,68.96],[179.99,64.97],[177.41,64.61],[178.91,63.25],[179.49,62.57],[177.36,62.52],[173.68,61.65],[170.7,60.34],[168.9,60.57],[165.84,60.16],[163.54,59.87],[162.02,58.24],[163.19,57.62],[162.13,56.12],[162.12,54.86],[160.02,53.2],[158.23,51.94],[156.42,51.7],[155.43,55.38],[156.76,57.36],[158.36,58.06],[161.87,60.34],[164.47,62.55],[162.66,61.64],[159.3,61.77],[154.22,59.76],[152.81,58.88],[151.34,59.5],[148.54,59.16],[142.2,59.04],[135.13,54.73],[137.19,53.98],[138.8,54.25],[141.35,53.09],[140.6,51.24],[140.06,48.45],[138.22,46.31],[135.52,43.99],[133.54,42.81],[132.28,43.28],[130.78,42.22],[129.97,41.94],[129.71,40.88],[129.01,40.49],[127.97,40.03],[127.5,39.32],[127.78,39.05],[129.21,37.43],[129.47,35.63],[128.19,34.89],[126.49,34.39],[126.56,35.68],[126.86,36.89],[125.69,37.94],[125.28,37.67],[124.98,37.95],[124.99,38.55],[125.13,38.85],[125.32,39.55],[124.27,39.93],[122.13,39.17],[121.59,39.36],[122.17,40.42],[120.77,40.59],[119.02,39.25],[117.53,38.74],[118.88,37.9],[119.7,37.16],[121.71,37.48],[122.52,36.93],[120.64,36.11],[119.15,34.91],[120.62,33.38],[121.91,31.69],[121.26,30.68],[122.09,29.83],[121.68,28.23],[120.4,27.05],[118.66,24.55],[115.89,22.78],[114.15,22.22],[113.24,22.05],[110.79,21.4],[109.89,20.28],[109.86,21.4],[108.05,21.55],[105.88,19.75],[106.43,18.0],[108.27,16.08],[109.34,13.43],[108.37,11.01],[106.41,9.53],[104.8,9.24],[104.33,10.49],[103.09,11.15],[101.69,12.65],[100.98,13.41],[100.02,12.31],[99.15,9.96],[99.87,9.21],[100.46,7.43],[101.62,6.74],[102.37,6.13],[103.38,4.86],[103.33,3.73],[103.5,2.79],[104.25,1.63],[103.52,1.23],[101.39,2.76],[100.7,3.94],[100.2,5.31],[100.09,6.46],[99.52,7.34],[98.5,8.38],[98.15,8.35],[98.55,9.93],[98.76,11.44],[98.51,13.12],[97.78,14.84],[97.16,16.93],[95.37,15.71],[94.19,16.04],[94.32,18.21],[93.66,19.73],[92.37,20.67],[92.03,21.7],[91.42,22.77],[90.59,22.39],[89.85,22.04],[89.42,21.97],[88.89,21.69],[86.98,21.5],[86.5,20.15],[83.94,18.3],[82.19,17.02],[81.69,16.31],[80.32,15.9],[80.23,13.84],[79.86,12.06],[79.34,10.31],[79.19,9.22],[77.94,8.25],[76.59,8.9],[75.75,11.31],[74.86,12.74],[74.44,14.62],[73.12,17.93],[72.82,20.42],[71.18,20.76],[69.16,22.09],[69.35,22.84],[67.44,23.94],[66.37,25.43],[62.91,25.22],[59.62,25.38],[57.4,25.74],[56.49,27.14],[54.72,26.48],[52.48,27.58],[50.85,28.81],[49.58,29.99],[48.57,29.93],[48.18,29.53],[48.42,28.55],[49.3,27.46],[50.15,26.69],[50.11,25.94],[50.53,25.33],[50.81,24.75],[51.01,26.01],[51.59,25.8],[51.39,24.63],[51.76,24.29],[52.58,24.18],[54.01,24.12],[55.44,25.44],[56.36,26.4],[56.39,25.9],[56.4,24.92],[57.4,23.88],[58.73,23.57],[59.45,22.66],[59.81,22.31],[59.28,21.43],[58.49,20.43],[57.83,20.24],[57.79,19.07],[57.23,18.95],[56.51,18.09],[55.66,17.88],[55.27,17.23],[54.24,17.05],[53.11,16.65],[52.19,15.94],[51.17,15.18],[48.68,14.0],[47.94,14.01],[46.72,13.4],[45.63,13.29],[45.14,12.95],[44.49,12.72],[43.48,12.64],[43.25,13.77],[42.89,14.8],[42.81,15.26],[42.82,15.91],[42.65,16.77],[42.27,17.47],[41.22,18.67],[40.25,20.17],[39.14,21.29],[39.07,22.58],[38.02,24.08],[37.15,24.86],[36.93,25.6],[36.25,26.57],[35.13,28.06],[34.79,28.61],[34.96,29.36],[34.64,29.1],[34.15,27.82],[33.59,27.97],[32.42,29.85],[32.73,28.71],[34.1,26.14],[34.8,25.03],[35.49,23.75],[36.69,22.2],[37.19,21.02],[37.11,19.81],[37.86,18.37],[38.99,16.84],[39.81,15.44],[41.73,13.92],[42.59,13.0],[43.32,12.39],[42.72,11.74],[43.47,11.28],[44.12,10.45],[45.56,10.7],[47.53,11.13],[48.38,11.38],[49.27,11.43],[50.26,11.68],[51.11,12.02],[51.04,11.17],[50.83,10.28],[50.07,8.08],[48.59,5.34],[46.56,2.86],[44.07,1.05],[42.04,-0.92],[41.59,-1.68],[40.64,-2.5],[40.12,-3.28],[39.6,-4.35],[38.74,-5.91],[39.44,-6.84],[39.19,-7.7],[39.19,-8.49],[39.95,-10.1],[40.48,-10.77],[40.56,-12.64],[40.78,-14.69],[40.09,-16.1],[38.54,-17.1],[36.28,-18.66],[35.2,-19.55],[34.7,-20.5],[35.37,-21.84],[35.56,-22.09],[35.37,-23.54],[35.46,-24.12],[34.22,-24.82],[32.57,-25.73],[32.92,-26.22],[32.58,-27.47],[32.2,-28.75],[31.33,-29.4],[30.62,-30.42],[28.93,-32.17],[27.46,-33.23],[25.91,-33.67],[25.17,-33.8],[23.59,-33.79],[22.57,-33.86],[20.69,-34.42],[19.62,-34.82],[18.86,-34.44],[18.38,-34.14],[18.25,-33.28],[18.25,-32.43],[17.57,-30.73],[17.06,-29.88],[15.6,-27.82],[14.99,-26.12],[14.41,-23.85],[14.26,-22.11],[13.35,-20.87],[12.61,-19.05],[11.73,-17.3],[11.78,-15.79],[12.18,-14.45],[12.74,-13.14],[13.63,-12.04],[13.69,-10.73],[13.12,-9.77],[12.93,-8.96],[12.93,-7.6],[12.23,-6.29],[12.18,-5.79],[11.09,-3.98],[9.41,-2.14],[8.83,-0.78],[9.29,0.27],[9.31,1.16],[9.8,3.07],[8.95,3.9],[8.49,4.5],[7.46,4.41],[6.7,4.24],[5.36,4.89],[4.33,6.27],[2.69,6.26],[1.06,5.93],[-1.06,5.0],[-2.86,4.99],[-4.01,5.18],[-5.83,4.99],[-7.52,4.34],[-7.97,4.36],[-9.91,5.59],[-11.44,6.79],[-12.43,7.26],[-13.12,8.16],[-13.69,9.49],[-14.33,10.02],[-14.69,10.66],[-15.13,11.04],[-16.09,11.52],[-16.31,11.96],[-16.68,12.38],[-16.71,13.6],[-17.62,14.73],[-16.7,15.62],[-16.55,16.67],[-16.15,18.11],[-16.38,19.59],[-16.54,20.57],[-17.02,21.42],[-16.59,22.16],[-16.33,23.02],[-15.43,24.36],[-14.82,25.1],[-14.44,26.25],[-13.14,27.64],[-11.69,28.15],[-10.4,29.1],[-9.81,31.18],[-9.3,32.56],[-7.65,33.7],[-6.24,35.15],[-5.19,35.76],[-3.64,35.4],[-2.17,35.17],[-0.13,35.89],[1.47,36.61],[4.82,36.87],[6.26,37.11],[7.74,36.89],[9.51,37.35],[10.18,36.72],[11.1,36.9],[10.59,35.95],[10.81,34.83],[10.34,33.79],[11.11,33.29],[12.66,32.79],[13.92,32.71],[15.71,31.38],[18.02,30.76],[19.57,30.53],[19.82,31.75],[20.85,32.71],[22.9,32.64],[23.61,32.19],[24.92,31.9],[26.5,31.59],[28.45,31.03],[29.68,31.19],[30.98,31.56],[31.96,30.93],[32.99,31.02],[34.27,31.22],[34.49,31.61],[34.96,32.83],[35.13,33.09],[35.98,34.61],[35.91,35.41],[35.78,36.28],[35.55,36.57],[34.03,36.22],[31.7,36.64],[30.39,36.26],[28.73,36.68],[27.05,37.65],[26.8,38.99],[27.28,40.42],[29.24,41.22],[32.35,41.74],[35.17,42.04],[38.35,40.95],[40.37,41.01],[41.7,41.96],[40.88,43.01],[39.96,43.44],[37.54,44.66],[37.4,45.4],[37.67,46.64],[39.12,47.26],[37.43,47.02],[35.82,46.65],[35.02,45.65],[36.53,45.47],[35.24,44.94],[33.33,44.56],[32.45,45.33],[33.59,45.85],[31.74,46.33],[30.75,46.58],[29.6,45.29],[29.14,44.82],[28.56,43.71],[27.67,42.58],[28.12,41.62],[28.81,41.05],[27.19,40.69],[26.04,40.62],[25.45,40.85],[23.71,40.69],[23.9,39.96],[22.81,40.48],[22.85,39.66],[22.97,38.97],[24.03,38.22],[23.12,37.92],[22.78,37.31],[22.49,36.41],[21.3,37.65],[20.73,38.77],[20.15,39.63],[19.96,39.92],[19.32,40.73],[19.54,41.72],[19.16,41.96],[18.45,42.48],[16.93,43.21],[15.17,44.24],[14.92,44.74],[14.26,45.23],[13.66,45.14],[13.72,45.5],[13.14,45.74],[12.38,44.89],[12.59,44.09],[14.03,42.76],[15.93,41.96],[15.89,41.54],[17.52,40.88],[18.48,40.17],[17.74,40.28],[16.45,39.8],[17.05,38.9],[16.1,37.99],[15.69,38.21],[16.11,38.96],[15.41,40.05],[14.7,40.6],[13.63,41.19],[12.11,41.7],[10.51,42.93],[9.7,44.04],[8.43,44.23],[7.44,43.69],[4.56,43.4],[2.99,42.47],[2.09,41.23],[0.72,40.68],[-0.28,39.31],[-0.47,38.29],[-1.44,37.44],[-3.42,36.66],[-5.0,36.32],[-5.87,36.03],[-6.52,36.94],[-7.86,36.84],[-8.9,36.87],[-8.84,38.27],[-9.53,38.74],[-9.05,39.76],[-8.77,40.76],[-8.99,41.54],[-8.98,42.59],[-7.98,43.75],[-5.41,43.57],[-3.52,43.46],[-1.38,44.02],[-2.23,47.06],[-4.49,47.96],[-3.3,48.9],[-1.93,49.78],[1.34,50.13],[2.51,51.15],[3.83,51.62],[6.07,53.51],[7.1,53.69],[8.12,53.53],[8.57,54.4],[8.12,55.52],[8.26,56.81],[9.42,57.17],[10.58,57.73],[10.25,56.89],[10.91,56.46],[10.37,56.19],[9.92,54.98],[10.95,54.36],[11.96,54.2],[13.65,54.08],[14.8,54.05],[17.62,54.85],[18.7,54.44],[19.89,54.87],[21.06,56.03],[21.58,57.41],[23.32,57.01],[24.31,57.79],[24.06,58.26],[23.34,59.19],[25.86,59.61],[27.98,59.48],[28.07,60.5],[24.5,60.06],[22.29,60.39],[21.54,61.71],[21.54,63.19],[24.73,64.9],[25.29,65.53],[22.18,65.72],[21.37,64.41],[17.85,62.75],[17.83,60.64],[17.87,58.95],[16.45,57.04],[14.67,56.2],[12.94,55.36],[11.79,57.44],[10.36,59.47],[7.05,58.08],[5.31,59.66],[5.91,62.61],[10.53,64.49],[14.76,67.81],[19.18,69.82],[23.02,70.2],[26.37,70.99],[31.29,70.45],[31.1,69.56],[33.78,69.3],[40.29,67.93],[41.13,66.79],[38.38,66.0],[33.18,66.63],[34.94,64.41],[37.01,63.85],[36.52,64.78],[39.59,64.52],[39.76,65.5],[43.02,66.42],[44.53,66.76],[44.19,67.95],[46.25,68.25],[45.56,67.57],[46.35,66.67],[48.14,67.52],[53.72,68.86],[53.49,68.2],[55.44,68.44],[58.8,68.88],[61.08,68.94],[60.55,69.85],[64.89,69.23],[69.18,68.62],[68.14,69.36],[67.26,69.93],[66.69,71.03],[69.2,72.84],[72.59,72.78],[71.85,71.41],[72.79,70.39],[73.67,68.41],[71.28,66.32],[72.82,66.53],[74.19,67.28],[74.47,68.33],[73.84,69.07],[74.4,70.63],[74.89,72.12],[75.16,72.86],[75.29,71.34],[75.9,71.87],[79.65,72.32],[80.61,72.58],[82.25,73.85],[86.82,73.94],[87.17,75.12],[90.26,75.64],[93.23,76.05],[96.68,75.92],[100.76,76.43],[101.99,77.29],[106.07,77.37],[106.97,76.97]],[[-93.84,77.52],[-96.17,77.56],[-94.42,77.82],[-93.84,77.52]],[[-110.19,77.7],[-113.53,77.73],[-111.26,78.15],[-110.19,77.7]],[[24.72,77.85],[20.73,77.68],[20.81,78.25],[23.28,78.08]],[[-109.66,78.6],[-112.54,78.41],[-111.5,78.85],[-109.66,78.6]],[[-95.83,78.06],[-98.12,78.08],[-98.63,78.87],[-96.75,78.77],[-95.83,78.06]],[[-100.06,78.32],[-101.3,78.02],[-105.18,78.38],[-105.42,78.92],[-103.53,79.17],[-100.06,78.32]],[[105.08,78.31],[101.26,79.23],[102.84,79.28],[105.08,78.31]],[[18.25,79.7],[19.03,78.56],[17.59,77.64],[15.91,76.77],[14.67,77.74],[11.22,78.87],[13.17,80.01],[15.14,79.67],[16.99,80.05]],[[25.45,80.41],[25.92,79.52],[20.08,79.57],[18.46,79.86],[20.46,80.6],[22.92,80.66]],[[51.14,80.55],[48.89,80.34],[47.59,80.01],[47.07,80.56],[46.8,80.77],[48.52,80.51],[50.04,80.92],[51.14,80.55]],[[99.94,78.88],[94.97,79.04],[92.55,80.14],[93.78,81.02],[97.88,80.75],[99.94,78.88]],[[-87.02,79.66],[-87.19,79.04],[-90.8,78.22],[-93.95,78.75],[-93.15,79.38],[-96.08,79.71],[-96.02,80.6],[-94.3,80.98],[-92.41,81.26],[-89.45,80.51],[-87.02,79.66]],[[-68.5,83.11],[-63.68,82.9],[-61.89,82.36],[-66.75,81.73],[-65.48,81.51],[-69.47,80.62],[-73.24,79.63],[-76.91,79.32],[-76.22,79.02],[-76.34,78.18],[-78.36,77.51],[-79.62,76.98],[-77.89,76.78],[-83.17,76.45],[-87.6,76.42],[-89.62,76.95],[-88.26,77.9],[-84.98,77.54],[-87.96,78.37],[-85.38,79.0],[-86.51,79.74],[-84.2,80.21],[-81.85,80.46],[-87.6,80.52],[-90.2,81.26],[-91.59,81.89],[-88.93,82.12],[-85.5,82.65],[-83.18,82.32],[-81.1,83.02],[-76.25,83.17],[-72.83,83.23],[-68.5,83.11]],[[-27.1,83.52],[-22.69,82.34],[-31.9,82.2],[-27.86,82.13],[-22.9,82.09],[-23.17,81.15],[-15.77,81.91],[-12.21,81.29],[-16.85,80.35],[-17.73,80.13],[-19.7,78.75],[-18.47,76.99],[-21.68,76.63],[-19.6,75.25],[-19.37,74.3],[-20.43,73.82],[-22.17,73.31],[-22.31,72.63],[-24.28,72.6],[-23.44,72.08],[-21.75,70.66],[-24.31,70.86],[-25.2,70.75],[-23.73,70.18],[-25.03,69.26],[-30.67,68.13],[-32.81,67.74],[-36.35,65.98],[-38.38,65.69],[-40.67,64.84],[-41.19,63.48],[-42.42,61.9],[-43.38,60.1],[-46.26,60.85],[-49.23,61.41],[-51.63,63.63],[-52.28,65.18],[-53.3,66.84],[-52.98,68.36],[-51.08,69.15],[-52.01,69.57],[-53.46,69.28],[-54.75,70.29],[-53.43,70.84],[-53.11,71.2],[-55,71.41],[-54.72,72.59],[-56.12,73.65],[-58.6,75.1],[-61.27,76.1],[-66.06,76.13],[-69.66,76.38],[-68.78,77.32],[-71.04,77.64],[-73.16,78.43],[-65.71,79.39],[-68.02,80.12],[-63.69,81.21],[-62.65,81.77],[-57.21,82.19],[-53.04,81.89],[-48.0,82.06],[-44.52,81.66],[-46.76,82.63],[-39.9,83.18],[-35.09,83.65]]];
var globeInitialized = false;

function initGlobe() {
    if (globeInitialized) return;
    var canvas = document.getElementById('globe_canvas');
    if (!canvas) return;
    var container = document.getElementById('globe_container');
    if (!container) return;
    canvas.width = container.offsetWidth || 800;
    canvas.height = container.offsetHeight || 500;
    globeInitialized = true;

    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    var cx = W/2, cy = H/2;
    var R = Math.min(W,H) * 0.40;
    var rot = 0;
    var tooltip = document.getElementById('globe_tooltip');
    var mouseX = -999, mouseY = -999;
    var isDragging = false, prevX = 0;

    var ZONES = [
        {name:"Hormuz",     lat:26.57, lon:56.25,  dev:0.030, wind:8.2,  label:"Persian Gulf"},
        {name:"Malacca",    lat:2.50,  lon:101.00, dev:0.020, wind:6.1,  label:"SE Asia"},
        {name:"Bab-el-M",   lat:12.60, lon:43.30,  dev:0.020, wind:11.2, label:"Red Sea"},
        {name:"Taiwan",     lat:24.00, lon:119.00, dev:0.030, wind:9.4,  label:"E.Asia"},
        {name:"N.Atlantic", lat:48.00, lon:-30.00, dev:0.070, wind:14.5, label:"Atlantic"},
        {name:"S.Pacific",  lat:42.00, lon:-175.0, dev:0.050, wind:12.8, label:"Pacific"},
        {name:"C.G.Hope",   lat:-35.0, lon:20.00,  dev:0.060, wind:16.3, label:"S.Atlantic"},
        {name:"Arabia",     lat:18.00, lon:62.00,  dev:0.020, wind:7.8,  label:"Indian"},
        {name:"S.China",    lat:10.00, lon:113.00, dev:0.020, wind:8.5,  label:"SE Asia"},
        {name:"Kara Sea",   lat:75.00, lon:65.00,  dev:0.030, wind:5.1,  label:"Arctic"},
        {name:"G.Mexico",   lat:26.00, lon:-88.0,  dev:0.020, wind:6.8,  label:"Americas"},
        {name:"Mediterr",   lat:36.00, lon:15.00,  dev:0.020, wind:7.4,  label:"Med"},
        {name:"Black Sea",  lat:42.00, lon:33.00,  dev:0.010, wind:5.2,  label:"Black Sea"},
        {name:"Dover",      lat:51.00, lon:1.50,   dev:0.010, wind:9.3,  label:"N.Sea"},
        {name:"Bay Beng",   lat:15.00, lon:90.00,  dev:0.020, wind:7.2,  label:"Indian"},
        {name:"Suez",       lat:29.97, lon:32.55,  dev:0.000, wind:4.3,  label:"Med"}
    ];

    function getColor(dev){
        if(dev>=0.06) return "#E74C3C";
        if(dev>=0.02) return "#F39C12";
        return "#27AE60";
    }
    function getLabel(dev){
        if(dev>=0.06) return "CRITICAL";
        if(dev>=0.02) return "WARNING";
        return "NORMAL";
    }

    function project(lat,lon){
        var phi=(90-lat)*Math.PI/180;
        var theta=(lon*Math.PI/180)+rot;
        var x=-R*Math.sin(phi)*Math.cos(theta);
        var y=R*Math.cos(phi);
        var z=R*Math.sin(phi)*Math.sin(theta);
        return {x:cx+x, y:cy-y, z:z, vis:z>0};
    }

    function drawLand(){
        ctx.fillStyle="rgba(20,60,90,0.7)";
        ctx.strokeStyle="rgba(0,229,176,0.3)";
        ctx.lineWidth=0.5;

        LAND_POLYGONS.forEach(function(ring){
            ctx.beginPath();
            var started=false, prevVis=false;
            for(var i=0;i<ring.length;i++){
                var p=project(ring[i][1],ring[i][0]);
                if(p.vis){
                    if(!started||!prevVis){ctx.moveTo(p.x,p.y);started=true;}
                    else ctx.lineTo(p.x,p.y);
                }
                prevVis=p.vis;
            }
            if(started){ctx.fill();ctx.stroke();}
        });
    }

    canvas.addEventListener("mousedown",function(e){isDragging=true;prevX=e.clientX;});
    window.addEventListener("mouseup",function(){isDragging=false;});
    canvas.addEventListener("mousemove",function(e){
        if(isDragging){rot+=(e.clientX-prevX)*0.008;prevX=e.clientX;}
        var rect=canvas.getBoundingClientRect();
        mouseX=(e.clientX-rect.left)*(canvas.width/rect.width);
        mouseY=(e.clientY-rect.top)*(canvas.height/rect.height);
    });
    canvas.addEventListener("mouseleave",function(){
        mouseX=-999;mouseY=-999;
        if(tooltip)tooltip.style.display="none";
    });

    var t=0;
    function draw(){
        requestAnimationFrame(draw);
        t+=0.016;
        if(!isDragging) rot+=0.003;

        ctx.clearRect(0,0,W,H);

        // Arka plan
        var bg=ctx.createRadialGradient(cx,cy,0,cx,cy,R*1.3);
        bg.addColorStop(0,"#071525");bg.addColorStop(1,"#000510");
        ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);

        // Okyanus
        var og=ctx.createRadialGradient(cx-R*0.3,cy-R*0.3,0,cx,cy,R);
        og.addColorStop(0,"#0D2A45");og.addColorStop(0.7,"#061828");og.addColorStop(1,"#030C18");
        ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);
        ctx.fillStyle=og;ctx.fill();

        // Kıtalar (GeoJSON)
        drawLand();

        // Grid
        ctx.strokeStyle="rgba(27,79,114,0.15)";ctx.lineWidth=0.4;
        for(var la=-60;la<=60;la+=30){
            ctx.beginPath();var first=true;
            for(var lo=-180;lo<=180;lo+=5){
                var p=project(la,lo);
                if(p.vis){if(first){ctx.moveTo(p.x,p.y);first=false;}else ctx.lineTo(p.x,p.y);}
            }ctx.stroke();
        }
        for(var lo2=-180;lo2<180;lo2+=30){
            ctx.beginPath();var first2=true;
            for(var la2=-85;la2<=85;la2+=5){
                var p2=project(la2,lo2);
                if(p2.vis){if(first2){ctx.moveTo(p2.x,p2.y);first2=false;}else ctx.lineTo(p2.x,p2.y);}
            }ctx.stroke();
        }

        // Globe kenarı
        ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);
        ctx.strokeStyle="rgba(0,229,176,0.5)";ctx.lineWidth=1.5;ctx.stroke();

        // Atmosfer
        var atm=ctx.createRadialGradient(cx,cy,R*0.94,cx,cy,R*1.1);
        atm.addColorStop(0,"rgba(26,188,156,0.07)");atm.addColorStop(1,"rgba(26,188,156,0)");
        ctx.beginPath();ctx.arc(cx,cy,R*1.1,0,Math.PI*2);ctx.fillStyle=atm;ctx.fill();

        // Parlama
        var gl=ctx.createRadialGradient(cx-R*0.4,cy-R*0.35,0,cx-R*0.4,cy-R*0.35,R*0.5);
        gl.addColorStop(0,"rgba(52,152,219,0.08)");gl.addColorStop(1,"rgba(0,0,0,0)");
        ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.fillStyle=gl;ctx.fill();

        // SSH Zone noktaları
        var hoveredZone=null;
        var visZones=ZONES.map(function(z){return{z:z,p:project(z.lat,z.lon)};})
            .filter(function(i){return i.p.vis;})
            .sort(function(a,b){return a.p.z-b.p.z;});

        visZones.forEach(function(item){
            var z=item.z,p=item.p;
            var col=getColor(z.dev);
            var sz=(z.dev>=0.06?9:z.dev>=0.02?7:5);
            var depth=(p.z+R)/(2*R);sz*=(0.5+depth*0.5);
            var pulse=1+0.3*Math.sin(t*2+z.lon*0.05);

            // Halo
            var hcol=col==="@E74C3C"?"rgba(231,76,60,0.15)":
                      col==="#F39C12"?"rgba(243,156,18,0.15)":"rgba(39,174,96,0.15)";
            ctx.beginPath();ctx.arc(p.x,p.y,sz*pulse*2.5,0,Math.PI*2);
            ctx.fillStyle=hcol;ctx.fill();

            // Nokta
            ctx.beginPath();ctx.arc(p.x,p.y,sz,0,Math.PI*2);
            ctx.fillStyle=col;ctx.shadowBlur=15;ctx.shadowColor=col;
            ctx.fill();ctx.shadowBlur=0;

            // Hover
            var dx=mouseX-p.x,dy=mouseY-p.y;
            if(Math.sqrt(dx*dx+dy*dy)<sz*2.5) hoveredZone={z:z,p:p,col:col};
        });

        if(hoveredZone&&tooltip){
            var hz=hoveredZone;
            tooltip.style.display="block";
            tooltip.style.left=(hz.p.x+15)+"px";
            tooltip.style.top=(hz.p.y-10)+"px";
            tooltip.innerHTML="<b style='color:#00E5B0'>"+hz.z.name+"</b> <span style='color:#4A6FA5'>("+hz.z.label+")</span><br>"+
                "<hr style='border:none;border-top:1px solid #0F2A42;margin:4px 0'>"+
                "SSH Dev: <b style='color:"+hz.col+"'>"+(hz.z.dev>=0?"+":"")+hz.z.dev.toFixed(3)+"m</b><br>"+
                "Wind: <b style='color:#9B59B6'>"+hz.z.wind+" m/s</b><br>"+
                "Status: <span style='color:"+hz.col+";font-weight:700'>"+getLabel(hz.z.dev)+"</span>";
            canvas.style.cursor="pointer";
        } else if(tooltip){
            tooltip.style.display="none";
            canvas.style.cursor="grab";
        }

        // Aktif rota ciz
        if (activeGlobeRoute) {
            var wps = activeGlobeRoute.waypoints;
            var col = activeGlobeRoute.color;
            
            // Rota cizgisi
            ctx.beginPath();
            ctx.strokeStyle = col;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 3]);
            var routeStarted = false;
            for (var ri = 0; ri < wps.length; ri++) {
                var rp = project(wps[ri].lat, wps[ri].lon);
                if (rp.vis) {
                    if (!routeStarted) { ctx.moveTo(rp.x, rp.y); routeStarted = true; }
                    else ctx.lineTo(rp.x, rp.y);
                } else {
                    routeStarted = false;
                }
            }
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Waypoint noktaları
            wps.forEach(function(wp, idx) {
                var rp = project(wp.lat, wp.lon);
                if (!rp.vis) return;
                
                // Pulse animasyonu
                var pulse = 1 + 0.4 * Math.sin(t * 3 + idx * 0.5);
                
                // Dış halka
                ctx.beginPath();
                ctx.arc(rp.x, rp.y, 8 * pulse, 0, Math.PI * 2);
                ctx.fillStyle = col.replace(')', ',0.2)').replace('rgb', 'rgba') || 'rgba(26,188,156,0.2)';
                ctx.fill();
                
                // İç nokta
                ctx.beginPath();
                ctx.arc(rp.x, rp.y, 4, 0, Math.PI * 2);
                ctx.fillStyle = col;
                ctx.shadowBlur = 10;
                ctx.shadowColor = col;
                ctx.fill();
                ctx.shadowBlur = 0;
                
                // Başlangıç ve bitiş özel
                if (idx === 0 || idx === wps.length - 1) {
                    ctx.beginPath();
                    ctx.arc(rp.x, rp.y, 6, 0, Math.PI * 2);
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
                
                // Waypoint ismi (mouse yakınsa)
                var dx = mouseX - rp.x, dy = mouseY - rp.y;
                if (Math.sqrt(dx*dx + dy*dy) < 15) {
                    ctx.fillStyle = 'rgba(13,31,53,0.9)';
                    var tw = ctx.measureText(wp.name).width + 16;
                    ctx.fillRect(rp.x + 10, rp.y - 12, tw, 20);
                    ctx.fillStyle = col;
                    ctx.font = '10px monospace';
                    ctx.fillText(wp.name, rp.x + 18, rp.y + 2);
                }
            });
            
            // Rota başlığı
            ctx.fillStyle = col;
            ctx.font = 'bold 11px monospace';
            ctx.fillText(activeGlobeRoute.name, 12, 20);
        }

        // Alt yazı
        ctx.fillStyle="rgba(74,111,165,0.7)";
        ctx.font="10px monospace";
        ctx.fillText("THALASSA // SSH ANOMALY // NASA+9SAT",12,H-12);
    }
    draw();
}

// Init on load
window.onload=function(){ renderFleet(); renderSSH(); startAIS();
  initMap();
  previewRoute();
};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" onload="window._threeReady=true"></script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/route_data/<route_key>")
def route_data(route_key):
    route = ROUTES.get(route_key, ROUTES["istanbul_trabzon"])
    return jsonify(route["waypoints"])

@app.route("/analyze", methods=["POST"])
@limiter.limit("20 per minute")
def analyze():
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        return jsonify({"error": "invalid request body"}), 400
    if len(str(d.get("vessel", ""))) > 60 or len(str(d.get("route", ""))) > 60:
        return jsonify({"error": "invalid parameters"}), 400
    try:
        d = dict(d)
        d["speed"] = float(d.get("speed", 12))
        d["draft"] = float(d.get("draft", 8.5))
        d["swh"]   = float(d.get("swh", 1.2))
        d["sst"]   = float(d.get("sst", 22))
        d["days"]  = int(d.get("days", 280))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid parameters"}), 400
    if not (1 <= d["speed"] <= 30 and 1 <= d["draft"] <= 26
            and 0 <= d["swh"] <= 20 and -2 <= d["sst"] <= 40
            and 1 <= d["days"] <= 366):
        return jsonify({"error": "parameters out of range"}), 400
    vessel = d["vessel"]
    route_key = d["route"]
    speed = float(d["speed"])
    draft = float(d["draft"])
    swh = float(d["swh"])
    days = int(d["days"])

    route = ROUTES.get(route_key, ROUTES["istanbul_trabzon"])
    profile = VESSEL_PROFILE.get(vessel, {"dwt":8000,"fuel":12})

    result_wps = []
    drag_total = 0
    for wp in route["waypoints"]:
        drag = predict_drag(wp["lat"],wp["lon"],wp["depth"],wp["ssh"],swh,speed,draft)
        sav = max(8, min(15, (1 - drag) * 18))
        drag_total += drag
        result_wps.append({
            "name":wp["name"],"lat":wp["lat"],"lon":wp["lon"],
            "depth":wp["depth"],"ssh":wp["ssh"],"swh":swh,
            "drag":round(drag,4),"savings":round(sav,1)
        })

    avg_drag = drag_total / len(route["waypoints"])
    sav_rate = max(0.08, min(0.15, 0.20 - avg_drag * 0.15))
    fuel = profile["fuel"]; dwt = profile["dwt"]
    cost_savings = fuel * days * 650 * sav_rate
    co2_reduction = fuel * sav_rate * days * 3.151

    p = CII_REF.get(vessel,{"a":588.0,"c":0.3885,"d":[0.83,0.94,1.06,1.19]})
    cap = min(dwt, p.get("cap", dwt))
    req = p["a"] * (cap**(-p["c"])) * (1-11/100)
    cii_b = (fuel*days*3.151*1_000_000)/(dwt*5000)
    cii_a = (fuel*(1-sav_rate)*days*3.151*1_000_000)/(dwt*5000)

    def cii_grade(v, r, d=None):
        d = d or [0.86,0.94,1.06,1.18]
        o = v / r
        if o <= d[0]: return "A"
        if o <= d[1]: return "B"
        if o <= d[2]: return "C"
        if o <= d[3]: return "D"
        return "E"

    return jsonify({
        "drag":round(avg_drag,4),
        "savings":round(sav_rate*100,1),
        "cost_savings":round(cost_savings,0),
        "co2_reduction":round(co2_reduction,1),
        "cii_before":cii_grade(cii_b, req, p.get("d")),
        "cii_after":cii_grade(cii_a, req, p.get("d")),
        "cii_b_val":round(cii_b,3),
        "cii_a_val":round(cii_a,3),
        "waypoints":result_wps,
        "tasarruf":round(sav_rate*100,1),
        "co2_azalma":round(co2_reduction,1),
    })

if __name__ == "__main__":
    import os; app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
