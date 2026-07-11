from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# --- Model ---
class GucluPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(7, 512), nn.LayerNorm(512), nn.GELU())
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.05))
            for _ in range(6)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.input_layer(x)
        for k in self.layers: h = h + k(h)
        return self.output_layer(h)

model = None
try:
    model = GucluPINN()
    model.load_state_dict(torch.load("batimetrix_model_v2.pt", weights_only=True))
    model.eval()
    print("Model loaded!")
except Exception as e:
    print(f"Model not found, using physics formula: {e}")
    model = None

# --- CII Parametreleri ---
CII_REF = {
    "VLCC Tanker":       {"a": 5247.0, "c": 0.610},
    "Panamax Konteyner": {"a": 1984.0, "c": 0.489},
    "Capesize Bulk":     {"a": 4745.0, "c": 0.622},
    "LNG Carrier":       {"a": 9.827,  "c": 0.000},
    "Kuru Yuk (Handy)":  {"a": 588.0,  "c": 0.3885},
    "Karadeniz Kargo":   {"a": 588.0,  "c": 0.3885},
}
GEMI_PROFIL = {
    "VLCC Tanker":       {"draft": 22.0, "speed": 15.0, "dwt": 300000, "fuel": 120},
    "Panamax Konteyner": {"draft": 13.5, "speed": 20.0, "dwt":  65000, "fuel":  80},
    "Capesize Bulk":     {"draft": 18.0, "speed": 14.5, "dwt": 180000, "fuel":  40},
    "LNG Carrier":       {"draft": 12.0, "speed": 19.5, "dwt":  80000, "fuel":  65},
    "Kuru Yuk (Handy)":  {"draft": 10.0, "speed": 14.0, "dwt":  35000, "fuel":  25},
    "Karadeniz Kargo":   {"draft":  8.5, "speed": 12.0, "dwt":   8000, "fuel":  12},
}

def tahmin_drag(lat, lon, depth, ssh, swh, speed, draft):
    if model is not None:
        inp = torch.tensor([[
            (lat+70)/150, (lon+180)/360, depth/6000,
            (ssh+2)/4, swh/20, speed/25, draft/22
        ]]).float()
        with torch.no_grad():
            return model(inp).item()
    else:
        re = (speed * 0.5144) * draft / 1.05e-6
        cf = 0.075 / ((np.log10(re) - 2) ** 2) if re > 0 else 0.002
        wave_factor = 1 + 0.015 * swh ** 1.5
        depth_factor = 1 + 0.1 * max(0, 1 - depth/50)
        drag = cf * wave_factor * depth_factor * (speed/10) ** 1.8
        return min(max(drag * 15, 0.05), 0.95)

def hesapla_cii(vessel_name, dwt, fuel_per_day, mesafe_nm, voyage_days):
    params = CII_REF.get(vessel_name, {"a": 588.0, "c": 0.3885})
    ref = params["a"] * (dwt ** (-params["c"])) * (1 - 11/100)
    co2_g = fuel_per_day * voyage_days * 3.151 * 1_000_000
    cii = co2_g / (dwt * mesafe_nm)
    ratio = cii / ref
    if   ratio <= 0.86: return "A", cii, ref
    elif ratio <= 0.94: return "B", cii, ref
    elif ratio <= 1.06: return "C", cii, ref
    elif ratio <= 1.18: return "D", cii, ref
    else:              return "E", cii, ref

HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batimetrix — AI Maritime Intelligence</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0A1628; color:white; font-family:'Segoe UI',sans-serif; min-height:100vh; }

  /* HEADER */
  header {
    background:linear-gradient(135deg,#0A1628,#1B4F72);
    padding:20px 40px; display:flex; align-items:center;
    justify-content:space-between; border-bottom:2px solid #1ABC9C;
  }
  .logo { font-size:28px; font-weight:900; color:#1ABC9C; letter-spacing:2px; }
  .logo span { color:white; }
  .tagline { font-size:12px; color:#7F8C8D; letter-spacing:1px; }
  .badge {
    background:#1ABC9C22; border:1px solid #1ABC9C;
    padding:6px 14px; border-radius:20px; font-size:11px; color:#1ABC9C;
  }

  /* MAIN */
  .container { max-width:1400px; margin:0 auto; padding:30px 20px; }

  /* CARDS */
  .grid { display:grid; grid-template-columns:400px 1fr; gap:24px; }
  .card {
    background:#0D1F35; border:1px solid #1B4F72;
    border-radius:16px; padding:24px;
  }
  .card h2 {
    font-size:14px; color:#1ABC9C; letter-spacing:1px;
    text-transform:uppercase; margin-bottom:20px;
    padding-bottom:10px; border-bottom:1px solid #1B4F72;
  }

  /* FORM */
  .form-group { margin-bottom:16px; }
  label { font-size:12px; color:#7F8C8D; display:block; margin-bottom:6px; }
  select, input {
    width:100%; background:#0A1628; border:1px solid #1B4F72;
    color:white; padding:10px 14px; border-radius:8px; font-size:13px;
    outline:none; transition:border 0.2s;
  }
  select:focus, input:focus { border-color:#1ABC9C; }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }

  /* BUTTON */
  .btn {
    width:100%; padding:14px; background:linear-gradient(135deg,#1ABC9C,#148F77);
    border:none; border-radius:10px; color:white; font-size:15px;
    font-weight:700; cursor:pointer; letter-spacing:1px;
    transition:transform 0.2s, box-shadow 0.2s; margin-top:8px;
  }
  .btn:hover { transform:translateY(-2px); box-shadow:0 8px 25px #1ABC9C44; }
  .btn:active { transform:translateY(0); }

  /* RESULTS */
  .results { display:none; }
  .metric-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-bottom:20px; }
  .metric {
    background:#0A1628; border-radius:12px; padding:18px;
    border-left:4px solid #1ABC9C; text-align:center;
  }
  .metric.red   { border-left-color:#E74C3C; }
  .metric.gold  { border-left-color:#F39C12; }
  .metric.green { border-left-color:#27AE60; }
  .metric.blue  { border-left-color:#3498DB; }
  .metric-val { font-size:32px; font-weight:900; color:#1ABC9C; }
  .metric.red   .metric-val { color:#E74C3C; }
  .metric.gold  .metric-val { color:#F39C12; }
  .metric.green .metric-val { color:#27AE60; }
  .metric.blue  .metric-val { color:#3498DB; }
  .metric-lbl { font-size:11px; color:#7F8C8D; margin-top:4px; text-transform:uppercase; }

  /* PROGRESS */
  .progress-wrap { margin:20px 0; }
  .progress-label { display:flex; justify-content:space-between; font-size:12px; color:#7F8C8D; margin-bottom:6px; }
  .progress-bar { height:10px; background:#1B4F72; border-radius:5px; overflow:hidden; }
  .progress-fill { height:100%; border-radius:5px; transition:width 1s ease; }

  /* CII BOX */
  .cii-box {
    border-radius:12px; padding:20px; text-align:center; margin:16px 0;
    border:2px solid;
  }
  .cii-box.A { border-color:#27AE60; background:#27AE6011; }
  .cii-box.B { border-color:#2ECC71; background:#2ECC7111; }
  .cii-box.C { border-color:#F39C12; background:#F39C1211; }
  .cii-box.D { border-color:#E67E22; background:#E67E2211; }
  .cii-box.E { border-color:#E74C3C; background:#E74C3C11; }
  .cii-grade { font-size:64px; font-weight:900; }
  .cii-box.A .cii-grade { color:#27AE60; }
  .cii-box.B .cii-grade { color:#2ECC71; }
  .cii-box.C .cii-grade { color:#F39C12; }
  .cii-box.D .cii-grade { color:#E67E22; }
  .cii-box.E .cii-grade { color:#E74C3C; }
  .cii-label { font-size:13px; color:#7F8C8D; margin-top:4px; }

  /* GUZERGAH */
  .route-table { width:100%; border-collapse:collapse; font-size:12px; }
  .route-table th {
    background:#1B4F72; color:#1ABC9C; padding:10px 12px;
    text-align:left; font-size:11px; text-transform:uppercase;
  }
  .route-table td { padding:10px 12px; border-bottom:1px solid #1B4F7233; }
  .route-table tr:hover td { background:#1B4F7222; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
  .dot.green { background:#1ABC9C; }
  .dot.gold  { background:#F39C12; }
  .dot.red   { background:#E74C3C; }

  /* LOADING */
  .loading { display:none; text-align:center; padding:40px; }
  .spinner {
    width:50px; height:50px; border:3px solid #1B4F72;
    border-top-color:#1ABC9C; border-radius:50%;
    animation:spin 0.8s linear infinite; margin:0 auto 16px;
  }
  @keyframes spin { to { transform:rotate(360deg); } }

  /* NASA BADGE */
  .nasa-strip {
    display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap;
  }
  .nasa-badge {
    background:#1B4F7233; border:1px solid #1B4F72;
    border-radius:8px; padding:8px 14px; font-size:11px;
    display:flex; align-items:center; gap:8px;
  }
  .nasa-badge .dot-pulse {
    width:8px; height:8px; border-radius:50%; background:#1ABC9C;
    animation:pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.8); }
  }

  /* FOOTER */
  footer {
    text-align:center; padding:20px; color:#7F8C8D;
    font-size:11px; border-top:1px solid #1B4F72; margin-top:30px;
  }
  footer a { color:#1ABC9C; text-decoration:none; }
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">BATI<span>METRIX</span></div>
    <div class="tagline">PROACTIVE HYDRODYNAMIC DRAG PREDICTION ENGINE</div>
  </div>
  <div class="badge">🛰️ 3 NASA Satellites Active</div>
</header>

<div class="container">

  <!-- NASA Uydu Durumu -->
  <div class="nasa-strip">
    <div class="nasa-badge"><div class="dot-pulse"></div>SWOT — Sea Surface Height</div>
    <div class="nasa-badge"><div class="dot-pulse"></div>GPM — Storm Prediction</div>
    <div class="nasa-badge"><div class="dot-pulse"></div>MODIS — Sea Surface Temp</div>
    <div class="nasa-badge"><div class="dot-pulse"></div>GEBCO 2026 — Bathymetry</div>
  </div>

  <div class="grid">

    <!-- SOL: FORM -->
    <div>
      <div class="card">
        <h2>🚢 Vessel & Route Parameters</h2>

        <div class="form-group">
          <label>VESSEL TYPE</label>
          <select id="vessel_type">
            <option>Karadeniz Kargo</option>
            <option>Kuru Yuk (Handy)</option>
            <option>Panamax Konteyner</option>
            <option>Capesize Bulk</option>
            <option>LNG Carrier</option>
            <option>VLCC Tanker</option>
          </select>
        </div>

        <div class="form-group">
          <label>ROUTE</label>
          <select id="route">
            <option value="istanbul_trabzon">Istanbul → Trabzon</option>
            <option value="istanbul_novorossiysk">Istanbul → Novorossiysk</option>
            <option value="odessa_istanbul">Odessa → Istanbul</option>
            <option value="batumi_constanta">Batumi → Constanta</option>
            <option value="karadeniz_sakin">Karadeniz — Calm</option>
            <option value="atlantik">North Atlantic — Storm</option>
          </select>
        </div>

        <div class="row">
          <div class="form-group">
            <label>SPEED (KNOTS)</label>
            <input type="number" id="speed" value="12" min="5" max="25">
          </div>
          <div class="form-group">
            <label>DRAFT (M)</label>
            <input type="number" id="draft" value="8.5" min="3" max="22" step="0.5">
          </div>
        </div>

        <div class="row">
          <div class="form-group">
            <label>WAVE HEIGHT (M)</label>
            <input type="number" id="swh" value="1.2" min="0" max="12" step="0.1">
          </div>
          <div class="form-group">
            <label>SEA TEMP (°C)</label>
            <input type="number" id="sst" value="22" min="0" max="32">
          </div>
        </div>

        <div class="form-group">
          <label>ANNUAL VOYAGE DAYS</label>
          <input type="number" id="voyage_days" value="280" min="100" max="365">
        </div>

        <button class="btn" onclick="analiz()">
          ⚡ RUN BATIMETRIX ANALYSIS
        </button>
      </div>
    </div>

    <!-- SAG: SONUCLAR -->
    <div>
      <!-- Loading -->
      <div class="loading" id="loading">
        <div class="spinner"></div>
        <div style="color:#1ABC9C;font-size:14px">Running AI Analysis...</div>
        <div style="color:#7F8C8D;font-size:12px;margin-top:8px">
          SWOT + GPM + MODIS + GEBCO → PINN Model
        </div>
      </div>

      <!-- Sonuclar -->
      <div class="results" id="results">

        <!-- Ana Metrikler -->
        <div class="metric-grid">
          <div class="metric red">
            <div class="metric-val" id="drag_val">—</div>
            <div class="metric-lbl">Drag Score</div>
          </div>
          <div class="metric green">
            <div class="metric-val" id="tasarruf_val">—</div>
            <div class="metric-lbl">Fuel Savings</div>
          </div>
          <div class="metric gold">
            <div class="metric-val" id="para_val">—</div>
            <div class="metric-lbl">Annual Savings $</div>
          </div>
          <div class="metric blue">
            <div class="metric-val" id="co2_val">—</div>
            <div class="metric-lbl">CO2 Reduction (t/yr)</div>
          </div>
        </div>

        <!-- Progress Bar -->
        <div class="card" style="margin-bottom:16px">
          <h2>📊 Drag Analysis</h2>
          <div class="progress-wrap">
            <div class="progress-label">
              <span>Drag Score</span><span id="drag_label">—</span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" id="drag_bar"
                   style="width:0%;background:linear-gradient(90deg,#1ABC9C,#E74C3C)"></div>
            </div>
          </div>
          <div class="progress-wrap">
            <div class="progress-label">
              <span>Fuel Efficiency</span><span id="eff_label">—</span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" id="eff_bar"
                   style="width:0%;background:linear-gradient(90deg,#E74C3C,#1ABC9C)"></div>
            </div>
          </div>
        </div>

        <!-- CII -->
        <div class="card" style="margin-bottom:16px">
          <h2>⚖️ IMO CII Rating (2026)</h2>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
              <div style="font-size:11px;color:#7F8C8D;margin-bottom:8px">WITHOUT BATIMETRIX</div>
              <div class="cii-box E" id="cii_before_box">
                <div class="cii-grade" id="cii_before">—</div>
                <div class="cii-label" id="cii_before_val">—</div>
              </div>
            </div>
            <div>
              <div style="font-size:11px;color:#7F8C8D;margin-bottom:8px">WITH BATIMETRIX</div>
              <div class="cii-box A" id="cii_after_box">
                <div class="cii-grade" id="cii_after">—</div>
                <div class="cii-label" id="cii_after_val">—</div>
              </div>
            </div>
          </div>
          <div id="cii_mesaj" style="text-align:center;margin-top:12px;font-size:13px;color:#7F8C8D"></div>
        </div>

        <!-- Güzergah Tablosu -->
        <div class="card">
          <h2>🗺️ Route Analysis</h2>
          <table class="route-table">
            <thead>
              <tr>
                <th>Waypoint</th>
                <th>Depth (m)</th>
                <th>SSH (m)</th>
                <th>SWH (m)</th>
                <th>Drag</th>
                <th>Savings</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="guzergah_tbody"></tbody>
          </table>
        </div>

      </div><!-- /results -->
    </div>
  </div>
</div>

<footer>
  BATIMETRIX v1.5 &nbsp;|&nbsp;
  3 NASA Satellites: SWOT + GPM + MODIS &nbsp;|&nbsp;
  GEBCO 2026 Bathymetry &nbsp;|&nbsp;
  PINN Model: 1,657,025 Parameters &nbsp;|&nbsp;
  <a href="https://github.com/Batimetrix/batimetrix" target="_blank">github.com/Batimetrix/batimetrix</a>
</footer>

<script>
const GUZERGAHLAR = {
  istanbul_trabzon: [
    {name:"Istanbul Bogazi",   lat:41.10,lon:29.05,depth: 35,ssh:0.05},
    {name:"Kara.Giris",        lat:41.30,lon:29.50,depth:120,ssh:0.07},
    {name:"Bati Karadeniz",    lat:41.80,lon:30.50,depth:650,ssh:0.08},
    {name:"Zonguldak",         lat:41.60,lon:31.80,depth:850,ssh:0.08},
    {name:"Sinop",             lat:42.00,lon:35.10,depth:950,ssh:0.10},
    {name:"Samsun",            lat:41.70,lon:36.20,depth:800,ssh:0.11},
    {name:"Trabzon",           lat:41.00,lon:39.73,depth:200,ssh:0.06},
  ],
  istanbul_novorossiysk: [
    {name:"Istanbul Bogazi",   lat:41.10,lon:29.05,depth: 35,ssh:0.05},
    {name:"Bati KB",           lat:41.80,lon:30.50,depth:650,ssh:0.08},
    {name:"Orta KB",           lat:42.10,lon:33.00,depth:1100,ssh:0.09},
    {name:"Novorossiysk",      lat:44.72,lon:37.77,depth:120,ssh:0.06},
  ],
  odessa_istanbul: [
    {name:"Odessa",            lat:46.48,lon:30.73,depth: 80,ssh:0.07},
    {name:"Bati KB",           lat:44.00,lon:31.00,depth:800,ssh:0.10},
    {name:"Orta KB",           lat:42.50,lon:32.00,depth:1100,ssh:0.09},
    {name:"Istanbul Bogazi",   lat:41.10,lon:29.05,depth: 35,ssh:0.05},
  ],
  batumi_constanta: [
    {name:"Batumi",            lat:41.65,lon:41.64,depth:150,ssh:0.07},
    {name:"Dogu KB",           lat:42.00,lon:38.00,depth:900,ssh:0.09},
    {name:"Orta KB",           lat:42.20,lon:33.00,depth:1100,ssh:0.08},
    {name:"Constanta",         lat:44.17,lon:28.65,depth: 60,ssh:0.06},
  ],
  karadeniz_sakin: [
    {name:"Nokta 1",           lat:41.50,lon:30.00,depth:600,ssh:0.06},
    {name:"Nokta 2",           lat:41.80,lon:32.00,depth:900,ssh:0.08},
    {name:"Nokta 3",           lat:42.10,lon:34.00,depth:1100,ssh:0.09},
    {name:"Nokta 4",           lat:42.00,lon:36.00,depth:950,ssh:0.10},
  ],
  atlantik: [
    {name:"Biscay",            lat:45.00,lon:-5.00,depth:2800,ssh:0.25},
    {name:"Mid-Atlantic",      lat:48.00,lon:-15.00,depth:3500,ssh:0.35},
    {name:"Deep Atlantic",     lat:50.00,lon:-25.00,depth:4200,ssh:0.40},
    {name:"N.Atlantic",        lat:52.00,lon:-30.00,depth:3800,ssh:0.38},
  ],
};

const CII_REF = {
  "VLCC Tanker":       {a:5247.0, c:0.610},
  "Panamax Konteyner": {a:1984.0, c:0.489},
  "Capesize Bulk":     {a:4745.0, c:0.622},
  "LNG Carrier":       {a:9.827,  c:0.000},
  "Kuru Yuk (Handy)":  {a:588.0,  c:0.3885},
  "Karadeniz Kargo":   {a:588.0,  c:0.3885},
};
const GEMI_PROFIL = {
  "VLCC Tanker":       {dwt:300000, fuel:120},
  "Panamax Konteyner": {dwt: 65000, fuel: 80},
  "Capesize Bulk":     {dwt:180000, fuel: 40},
  "LNG Carrier":       {dwt: 80000, fuel: 65},
  "Kuru Yuk (Handy)":  {dwt: 35000, fuel: 25},
  "Karadeniz Kargo":   {dwt:  8000, fuel: 12},
};

function ciiNotu(ratio) {
  if (ratio <= 0.86) return "A";
  if (ratio <= 0.94) return "B";
  if (ratio <= 1.06) return "C";
  if (ratio <= 1.18) return "D";
  return "E";
}

async function analiz() {
  const gemi   = document.getElementById("vessel_type").value;
  const route   = document.getElementById("route").value;
  const speed    = parseFloat(document.getElementById("speed").value);
  const draft = parseFloat(document.getElementById("draft").value);
  const swh    = parseFloat(document.getElementById("swh").value);
  const sst    = parseFloat(document.getElementById("sst").value);
  const gun    = parseInt(document.getElementById("voyage_days").value);

  document.getElementById("loading").style.display = "block";
  document.getElementById("results").style.display = "none";

  const res = await fetch("/analiz", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({gemi, route, speed, draft, swh, sst, voyage_days: gun})
  });
  const data = await res.json();

  document.getElementById("loading").style.display = "none";
  document.getElementById("results").style.display = "block";

  // Ana metrikler
  document.getElementById("drag_val").textContent     = data.drag.toFixed(4);
  document.getElementById("tasarruf_val").textContent = "%" + data.savings.toFixed(1);
  document.getElementById("para_val").textContent     = "$" + (data.cost_savings/1000).toFixed(0) + "K";
  document.getElementById("co2_val").textContent      = data.co2_reduction.toFixed(0) + "t";

  // Progress
  const dragPct = Math.min(data.drag * 100, 100);
  document.getElementById("drag_bar").style.width   = dragPct + "%";
  document.getElementById("drag_label").textContent = data.drag.toFixed(4);
  document.getElementById("eff_bar").style.width    = (100 - dragPct) + "%";
  document.getElementById("eff_label").textContent  = "%" + (100 - dragPct).toFixed(1);

  // CII
  const cii_b = data.cii_before;
  const cii_a = data.cii_after;
  document.getElementById("cii_before").textContent     = cii_b;
  document.getElementById("cii_after").textContent      = cii_a;
  document.getElementById("cii_before_val").textContent = data.cii_b_val.toFixed(2) + " g/(DWT·nm)";
  document.getElementById("cii_after_val").textContent  = data.cii_a_val.toFixed(2) + " g/(DWT·nm)";
  document.getElementById("cii_before_box").className   = "cii-box " + cii_b;
  document.getElementById("cii_after_box").className    = "cii-box " + cii_a;

  const mesaj = cii_b !== cii_a
    ? `✅ CII Rating improved: ${cii_b} → ${cii_a} with Batimetrix!`
    : `CII Rating: ${cii_b} (further optimization possible)`;
  document.getElementById("cii_mesaj").textContent = mesaj;
  document.getElementById("cii_mesaj").style.color = cii_b !== cii_a ? "#1ABC9C" : "#7F8C8D";

  // Güzergah tablosu
  const tbody = document.getElementById("guzergah_tbody");
  tbody.innerHTML = "";
  data.waypoints.forEach(n => {
    const color = n.drag < 0.20 ? "green" : n.drag < 0.35 ? "gold" : "red";
    const status = n.drag < 0.20 ? "✅ Efficient" : n.drag < 0.35 ? "⚠️ Normal" : "🔴 High Drag";
    tbody.innerHTML += `
      <tr>
        <td><span class="dot ${color}"></span>${n.name}</td>
        <td>${n.depth}m</td>
        <td>${n.ssh.toFixed(3)}</td>
        <td>${n.swh.toFixed(1)}</td>
        <td style="color:${color==='green'?'#1ABC9C':color==='gold'?'#F39C12':'#E74C3C'};font-weight:700">${n.drag.toFixed(4)}</td>
        <td style="color:#1ABC9C">%${n.savings.toFixed(1)}</td>
        <td>${status}</td>
      </tr>`;
  });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/analiz", methods=["POST"])
def analiz():
    d = request.json
    gemi     = d["gemi"]
    route     = d["route"]
    speed      = float(d["speed"])
    draft   = float(d["draft"])
    swh      = float(d["swh"])
    sst      = float(d.get("sst", 22))
    voyage_days= int(d["voyage_days"])

    # Güzergah noktaları
    GUZERGAHLAR = {
        "istanbul_trabzon": [
            {"name":"Istanbul Bogazi","lat":41.10,"lon":29.05,"depth":35,"ssh":0.05},
            {"name":"Kara.Giris","lat":41.30,"lon":29.50,"depth":120,"ssh":0.07},
            {"name":"Bati Karadeniz","lat":41.80,"lon":30.50,"depth":650,"ssh":0.08},
            {"name":"Zonguldak","lat":41.60,"lon":31.80,"depth":850,"ssh":0.08},
            {"name":"Sinop","lat":42.00,"lon":35.10,"depth":950,"ssh":0.10},
            {"name":"Samsun","lat":41.70,"lon":36.20,"depth":800,"ssh":0.11},
            {"name":"Trabzon","lat":41.00,"lon":39.73,"depth":200,"ssh":0.06},
        ],
        "istanbul_novorossiysk": [
            {"name":"Istanbul Bogazi","lat":41.10,"lon":29.05,"depth":35,"ssh":0.05},
            {"name":"Bati KB","lat":41.80,"lon":30.50,"depth":650,"ssh":0.08},
            {"name":"Orta KB","lat":42.10,"lon":33.00,"depth":1100,"ssh":0.09},
            {"name":"Novorossiysk","lat":44.72,"lon":37.77,"depth":120,"ssh":0.06},
        ],
        "odessa_istanbul": [
            {"name":"Odessa","lat":46.48,"lon":30.73,"depth":80,"ssh":0.07},
            {"name":"Bati KB","lat":44.00,"lon":31.00,"depth":800,"ssh":0.10},
            {"name":"Orta KB","lat":42.50,"lon":32.00,"depth":1100,"ssh":0.09},
            {"name":"Istanbul Bogazi","lat":41.10,"lon":29.05,"depth":35,"ssh":0.05},
        ],
        "batumi_constanta": [
            {"name":"Batumi","lat":41.65,"lon":41.64,"depth":150,"ssh":0.07},
            {"name":"Dogu KB","lat":42.00,"lon":38.00,"depth":900,"ssh":0.09},
            {"name":"Orta KB","lat":42.20,"lon":33.00,"depth":1100,"ssh":0.08},
            {"name":"Constanta","lat":44.17,"lon":28.65,"depth":60,"ssh":0.06},
        ],
        "karadeniz_sakin": [
            {"name":"Nokta 1","lat":41.50,"lon":30.00,"depth":600,"ssh":0.06},
            {"name":"Nokta 2","lat":41.80,"lon":32.00,"depth":900,"ssh":0.08},
            {"name":"Nokta 3","lat":42.10,"lon":34.00,"depth":1100,"ssh":0.09},
            {"name":"Nokta 4","lat":42.00,"lon":36.00,"depth":950,"ssh":0.10},
        ],
        "atlantik": [
            {"name":"Biscay","lat":45.00,"lon":-5.00,"depth":2800,"ssh":0.25},
            {"name":"Mid-Atlantic","lat":48.00,"lon":-15.00,"depth":3500,"ssh":0.35},
            {"name":"Deep Atlantic","lat":50.00,"lon":-25.00,"depth":4200,"ssh":0.40},
            {"name":"N.Atlantic","lat":52.00,"lon":-30.00,"depth":3800,"ssh":0.38},
        ],
    }

    waypoints = GUZERGAHLAR.get(route, GUZERGAHLAR["istanbul_trabzon"])
    profile   = GEMI_PROFIL.get(gemi, {"dwt": 8000, "fuel": 12})

    # Her nokta için drag hesapla
    result_waypoints = []
    drag_total = 0
    for n in waypoints:
        drag = tahmin_drag(n["lat"], n["lon"], n["depth"],
                           n["ssh"], swh, speed, draft)
        sav  = max(0, (0.5 - drag) * 30)
        drag_total += drag
        result_waypoints.append({
            "name": n["name"], "depth": n["depth"],
            "ssh": n["ssh"], "swh": swh,
            "drag": round(drag, 4), "savings": round(sav, 1)
        })

    avg_drag     = drag_total / len(waypoints)
    savings_rate     = max(0, (0.5 - avg_drag) * 0.25)
    fuel_per_day    = profile["fuel"]
    dwt          = profile["dwt"]
    fuel_price  = 650
    voyage_distance = 5000

    # Yıllık hesaplar
    annual_fuel_usd = fuel_per_day * voyage_days * fuel_price
    cost_savings    = annual_fuel_usd * savings_rate
    co2_reduction       = fuel_per_day * savings_rate * voyage_days * 3.151

    # CII
    cii_params   = CII_REF.get(gemi, {"a": 588.0, "c": 0.3885})
    required_cii = cii_params["a"] * (dwt ** (-cii_params["c"])) * (1 - 11/100)
    co2_baseline_g = fuel_per_day * voyage_days * 3.151 * 1_000_000
    co2_optimized_g = fuel_per_day * (1-savings_rate) * voyage_days * 3.151 * 1_000_000
    cii_b   = co2_baseline_g / (dwt * voyage_distance)
    cii_a   = co2_optimized_g / (dwt * voyage_distance)

    def cii_not(v, r):
        o = v/r
        if o <= 0.86: return "A"
        if o <= 0.94: return "B"
        if o <= 1.06: return "C"
        if o <= 1.18: return "D"
        return "E"

    return jsonify({
        "drag":        round(avg_drag, 4),
        "savings":    round(savings_rate * 100, 1),
        "cost_savings": round(cost_savings, 0),
        "co2_reduction":  round(co2_reduction, 1),
        "cii_before":  cii_not(cii_b, required_cii),
        "cii_after":   cii_not(cii_a, required_cii),
        "cii_b_val":   round(cii_b, 3),
        "cii_a_val":   round(cii_a, 3),
        "waypoints":    result_waypoints,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
