# BATIMETRIX

**Proactive Hydrodynamic Drag Prediction Engine for Maritime Fuel Optimization**

Batimetrix predicts hydrodynamic drag **3 nautical miles ahead** of a vessel's position using real data from **three NASA satellites**, enabling proactive propeller pitch and engine optimization. Target: **8–12% net fuel savings** and measurable **IMO CII rating improvements**.

Built from scratch by a 18-year-old student from Turkey.

---

## Why Batimetrix?

Existing systems (Wartsila, Kongsberg, ZeroNorth) are **reactive** — they respond to conditions the ship is already in. Batimetrix is **proactive**: it computes underwater resistance before the vessel reaches the coordinate, giving the propulsion system time to adapt.

No other system combines:

| Component | Batimetrix | Competitors |
|---|---|---|
| NASA SWOT sea surface height | YES | No |
| NASA GPM storm prediction | YES | No |
| NASA MODIS sea surface temperature | YES | No |
| GEBCO 2026 bathymetry | YES | No |
| Physics-Informed Neural Network | YES | Rare |
| 3 NM look-ahead prediction | YES | No |
| IMO CII compliance module | YES | Partial |
| Open source | YES | No |

---

## Data Sources — Three NASA Satellites

| Satellite | Measurement | Update Cycle | Role in Batimetrix |
|---|---|---|---|
| **SWOT** | Sea surface height (ssh_karin) | 21 days | Current estimation via SSH anomaly |
| **GPM** | Precipitation (IMERG) | 30 minutes | Significant wave height prediction |
| **MODIS** | Sea surface temperature | Daily | Kinematic viscosity correction |
| **GEBCO 2026** | Bathymetry (15 arc-sec) | Static | Shallow-water resistance effects |

The model was fine-tuned on **296,526 real SWOT measurements** over the Black Sea.

---

## Model Architecture

**Physics-Informed Neural Network (PINN)** — 1,657,025 parameters

- Input: 7 features (lat, lon, depth, SSH anomaly, SWH, speed, draft)
- 6 residual blocks, 512 neurons each, GELU activation, LayerNorm
- Output: normalized drag score in [0, 1]

**Physics constraints in the loss function:**

- ITTC 1957 friction line: `Cf = 0.075 / (log10(Re) - 2)^2`
- Reynolds number scaling
- Morison-type wave loading factor
- Shallow-water resistance term
- Incompressibility (continuity) penalty

**Training:** 100,000 synthetic samples (100 epochs, AdamW, CosineAnnealingLR) followed by fine-tuning on real NASA SWOT data.

**Deployment:** Exported to ONNX (12.2 KB, max deviation vs PyTorch: 2.98e-08) for the Rust onboard inference engine.

---

## Two-Layer Architecture 
---

## Results

- Average drag score (Black Sea, calm): **0.13–0.14**
- Estimated fuel savings: **8–12%** depending on conditions
- CII rating improvement: **E to D** (Black Sea cargo), **B to A** (Handy bulk)
- Route analysis: Istanbul–Trabzon, Istanbul–Novorossiysk, Odessa–Istanbul, Batumi–Constanta

Fuel consumption figures calibrated against Clarkson Research / MAN Energy Solutions reference data (VLCC: 120 t/day, Panamax: 80 t/day, Capesize: 40 t/day).

---

## Repository Contents

| File | Description |
|---|---|
| `guclu_egitim2.py` | Main PINN training (100K samples) |
| `swot_gercek.py` | Real SWOT NetCDF download and processing |
| `nasa_gercek_egitim.py` | Fine-tuning on 296K real SWOT points |
| `gpm_entegre.py` | GPM precipitation to SWH integration |
| `modis_test.py` | MODIS SST to viscosity correction |
| `gpm_modis_gercek.py` | Combined 3-satellite inference |
| `cii_modul.py` | IMO CII rating calculator (MEPC.354(78)) |
| `gemi_tipleri.py` | Vessel-type analysis with realistic fuel data |
| `gemi_harita.py` | Multi-route interactive map (folium) |
| `grafik.py` | 7-panel analysis visualization |
| `guclu_export.py` | ONNX export and validation |
| `app.py` | Flask web dashboard |

---

## Quick Start

```bash
pip install torch numpy netCDF4 requests folium matplotlib flask
python guclu_egitim2.py      # train the model
python app.py                # launch the web dashboard
# open http://127.0.0.1:5000
```

NASA data access requires a free [Earthdata](https://urs.earthdata.nasa.gov/) account.

---

## Roadmap

- [x] PINN model with physics-based loss
- [x] Real NASA SWOT data integration (296K measurements)
- [x] GPM + MODIS integration (3 satellites)
- [x] ONNX export for edge deployment
- [x] IMO CII compliance module
- [x] Web dashboard
- [ ] Pilot validation with real vessel voyage data
- [ ] Rust onboard inference engine (full)
- [ ] AIS real-time position integration
- [ ] NASA SWOT Early Adopters program application

---

## Acknowledgments

- **NASA / CNES** — SWOT mission data via [PO.DAAC](https://podaac.jpl.nasa.gov)
- **NASA** — GPM IMERG and MODIS SST products
- **GEBCO** — General Bathymetric Chart of the Oceans 2026
- **Raissi, Perdikaris, Karniadakis (2019)** — the original PINN framework

---

## Author

**Mehmet Sinan** — Arizona State University (online) 

*This research uses data from NASA's SWOT mission. SWOT data are freely available under a Creative Commons Zero license.*