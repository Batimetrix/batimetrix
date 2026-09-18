# BATIMETRIX — Thalassa Maritime Intelligence Platform

**Proactive Hydrodynamic Drag Prediction Engine for Maritime Fuel Optimization**

🌐 **Live Demo: [batimetrix.onrender.com](https://batimetrix.onrender.com)**
📦 **GitHub: [github.com/Batimetrix/batimetrix](https://github.com/Batimetrix/batimetrix)**

Batimetrix predicts hydrodynamic drag 3 nautical miles ahead of a vessel's position using real data from **9 NASA satellites and data sources**, enabling proactive propulsion optimization.

**Target: 8-12% net fuel savings and measurable IMO CII rating improvements.**

Built from scratch by an 18-year-old independent researcher from Turkey.

---

## Why Batimetrix?

Existing systems (Wärtsilä, Kongsberg, ZeroNorth) are **reactive** — they respond to conditions the ship is already in. Batimetrix is **proactive**: it computes underwater resistance before the vessel reaches the coordinate, giving the propulsion system time to adapt.

| Component | Batimetrix | Competitors |
|---|---|---|
| 9 NASA satellite data sources | YES | No |
| Physics-Informed Neural Network | YES | Rare |
| 3 NM look-ahead prediction | YES | No |
| IMO CII compliance (MEPC.354(78)) | YES | Partial |
| 60 global trade routes | YES | No |
| 15 vessel classes | YES | No |
| Fleet Dashboard | YES | No |
| 3D SSH Anomaly Globe | YES | No |
| Ocean Intelligence module | YES | No |
| Open source | YES | No |

---

## Data Sources — 9 NASA Satellites & Data Sources

| Satellite | Measurement | Last Data | Role in Batimetrix |
|---|---|---|---|
| SWOT (NASA/CNES) | Sea surface height (ssh_karin) | 2025-05-03 | SSH anomaly → drag estimation |
| Sentinel-6 (NASA/ESA) | SSH continuity | 2026-01-16 | 10-day cycle SSH complement |
| GPM IMERG | Precipitation / wave height | 2025-09-30 | Significant wave height prediction |
| MODIS | Sea surface temperature | Daily | Kinematic viscosity correction |
| CYGNSS (8 satellites) | Ocean wind speed NRT | 2025-10-31 | Wave loading factor |
| PACE OCI | Ocean color / chlorophyll NRT | 2026-09-16 | Viscosity correction |
| SMAP | Sea surface salinity NRT | 2026-09-17 | Water density correction |
| ICESat-2 | Arctic ocean height | 2026-05-18 | Arctic route optimization |
| GEBCO 2026 | Bathymetry (15 arc-sec) | April 2026 | Shallow-water resistance |

The model was fine-tuned on **296,526 real SWOT measurements** over the Black Sea.

---

## Model Architecture

**Physics-Informed Neural Network (PINN) — 1,657,025 parameters**

- Input: 7 features (lat, lon, depth, SSH anomaly, SWH, speed, draft)
- 6 residual blocks, 512 neurons each, GELU activation, LayerNorm
- Output: normalized drag score in [0, 1]

Physics constraints in the loss function:
- ITTC 1957 friction line: `Cf = 0.075 / (log10(Re) - 2)^2`
- Reynolds number scaling
- Morison-type wave loading factor
- Shallow-water resistance term
- Incompressibility (continuity) penalty

Training: 100,000 synthetic samples (100 epochs, AdamW, CosineAnnealingLR) + fine-tuning on 296,526 real NASA SWOT measurements.

Deployment: Exported to ONNX (12.2 KB, max deviation vs PyTorch: 2.98e-08).

---

## Platform Features

- **60 global trade routes** — Black Sea, Mediterranean, Suez, Cape, Arctic NSR, Trans-Pacific, Trans-Atlantic
- **15 vessel classes** — VLCC, Suezmax, Aframax, MR, LNG, Capesize, Panamax, ULCV, Feeder and more
- **Fleet Dashboard** — multi-vessel portfolio analysis
- **Ocean Intel** — 16 global maritime zone SSH anomaly monitoring
- **3D Globe** — interactive SSH anomaly visualization
- **IMO CII** — MEPC.354(78) compliant rating calculation
- **7 languages** — EN, TR, EL, ZH, RU, ES, FR
- **PDF report** — downloadable analysis report

---

## Results

- Average drag score (Black Sea, calm): 0.13-0.14
- Estimated fuel savings: 8-12% depending on conditions
- CII rating improvement: E→D (Black Sea cargo), B→A (Handy bulk)
- 60 global trade routes analyzed

Fuel consumption figures calibrated against Lloyd's List / MAN Energy Solutions reference data.

---

## Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch PINN → ONNX export |
| Backend | Python / Flask |
| Frontend | HTML/CSS/JS, Leaflet.js |
| NASA Data | PO.DAAC CMR API (9 sources) |
| Deployment | Render.com |

---

## Contact

**Mehmet Sinan Bilgeç**
mbilgec@asu.edu
Independent Researcher | ASU Online Student Turkey

---

*Batimetrix is the foundation of Thalassa — a maritime intelligence platform vision combining NASA satellite data, physics-informed AI, and ocean analytics for commercial and defense applications.*
