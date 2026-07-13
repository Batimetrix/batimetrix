# BATIMETRIX V3 PRO - 7 dil + Leaflet harita + Chart.js analiz
from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import numpy as np
import math

app = Flask(__name__)

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
    "VLCC Tanker":       {"a":5247.0,"c":0.610},
    "Panamax Container": {"a":1984.0,"c":0.489},
    "Capesize Bulk":     {"a":4745.0,"c":0.622},
    "LNG Carrier":       {"a":9.827, "c":0.000},
    "Handy Bulk":        {"a":588.0, "c":0.3885},
    "Black Sea Cargo":   {"a":588.0, "c":0.3885},
}
VESSEL_PROFILE = {
    "VLCC Tanker":       {"dwt":300000,"fuel":120},
    "Panamax Container": {"dwt": 65000,"fuel": 80},
    "Capesize Bulk":     {"dwt":180000,"fuel": 40},
    "LNG Carrier":       {"dwt": 80000,"fuel": 65},
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
                <th>SSH (m)</th>
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
  </div>
</div>

<footer>
  <div class="footer-dots"><div class="footer-dot"></div><div class="footer-dot"></div><div class="footer-dot"></div></div>
  <div>BATIMETRIX V3 PRO // PINN 1,657,025 params // NASA SWOT+GPM+MODIS // GEBCO 2026 // IMO CII MEPC.354(78)</div>
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
  initMap();
  if(routeLayer){map.removeLayer(routeLayer);}
  if(markerLayer){map.removeLayer(markerLayer);}
  var latlngs=data.waypoints.map(w=>[w.lat,w.lon]);
  routeLayer=L.polyline(latlngs,{color:"#00E5B0",weight:3,opacity:.8}).addTo(map);
  markerLayer=L.layerGroup();
  data.waypoints.forEach(function(w){
    var c=w.drag<0.20?"#00E5B0":w.drag<0.35?"#FFC107":"#FF4757";
    var m=L.circleMarker([w.lat,w.lon],{radius:8,color:c,fillColor:c,fillOpacity:.3,weight:2});
    m.bindPopup(
      "<div style='font-family:JetBrains Mono;font-size:12px;min-width:160px'>"+
      "<b style='color:#00E5B0'>"+w.name+"</b><br>"+
      "Depth: "+w.depth+"m<br>"+
      "SSH: "+w.ssh.toFixed(3)+"m<br>"+
      "Drag: <b style='color:"+c+"'>"+w.drag.toFixed(4)+"</b><br>"+
      "Savings: <b style='color:#00E5B0'>"+w.savings.toFixed(1)+"%</b>"+
      "</div>"
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
  ["map_tab","analysis_tab","cii_tab","table_tab"].forEach(function(t){
    document.getElementById(t).style.display="none";
  });
  document.querySelectorAll(".tab").forEach(function(t){t.classList.remove("active")});
  document.getElementById(id).style.display="block";
  el.classList.add("active");
  if(id==="map_tab"){if(!map){initMap();} setTimeout(function(){if(map)map.invalidateSize();},200);}
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
  updateMap(data);
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
  });
}

// Init on load
window.onload=function(){
  initMap();
  previewRoute();
};
</script>
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
def analyze():
    d = request.json
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

    p = CII_REF.get(vessel,{"a":588.0,"c":0.3885})
    req = p["a"] * (dwt**(-p["c"])) * (1-11/100)
    cii_b = (fuel*days*3.151*1_000_000)/(dwt*5000)
    cii_a = (fuel*(1-sav_rate)*days*3.151*1_000_000)/(dwt*5000)

    def cii_grade(v,r):
        o=v/r
        if o<=0.86:return "A"
        if o<=0.94:return "B"
        if o<=1.06:return "C"
        if o<=1.18:return "D"
        return "E"

    return jsonify({
        "drag":round(avg_drag,4),
        "savings":round(sav_rate*100,1),
        "cost_savings":round(cost_savings,0),
        "co2_reduction":round(co2_reduction,1),
        "cii_before":cii_grade(cii_b,req),
        "cii_after":cii_grade(cii_a,req),
        "cii_b_val":round(cii_b,3),
        "cii_a_val":round(cii_a,3),
        "waypoints":result_wps,
        "tasarruf":round(sav_rate*100,1),
        "co2_azalma":round(co2_reduction,1),
    })

if __name__ == "__main__":
    import os; app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
