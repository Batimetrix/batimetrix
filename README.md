# BATIMETRIX
**Proactive Hydrodynamic Drag Prediction Engine for Maritime Fuel Optimization**

Batimetrix predicts hydrodynamic drag 3 nautical miles ahead of a vessel's position using real data from three NASA satellites, enabling proactive propeller pitch and engine optimization. Target: 8-12% net fuel savings and measurable IMO CII rating improvements.

Built from scratch by an 18-year-old student from Turkey.

---

## Why Batimetrix?

Existing systems (Wartsila, Kongsberg, ZeroNorth) are reactive - they respond to conditions the ship is already in. Batimetrix is **proactive**: it computes underwater resistance before the vessel reaches the coordinate, giving the propulsion system time to adapt.

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

## Data Sources - Three NASA Satellites

| Satellite | Measurement | Update Cycle | Role in Batimetrix |
|---|---|---|---|
| SWOT | Sea surface height (ssh_karin) | 21 days | Current estimation via SSH anomaly |
| GPM | Precipitation (IMERG) | 30 minutes | Significant wave height prediction |
| MODIS | Sea surface temperature | Daily | Kinematic viscosity correction |
| GEBCO 2026 | Bathymetry (15 arc-sec) | Static | Shallow-water resistance effects |

The model was fine-tuned on **296,526 real SWOT measurements** over the Black Sea.

---

## Model Architecture

**Physics-Informed Neural Network (PINN) - 1,657,025 parameters**

- Input: 7 features (lat, lon, depth, SSH anomaly, SWH, speed, draft)
- 6 residual blocks, 512 neurons each, GELU activation, LayerNorm
- Output: normalized drag score in [0, 1]

**Physics constraints in the loss function:**
- ITTC 1957 friction line: Cf = 0.075 / (log10(Re) - 2)^2
- Reynolds number scaling
- Morison-type wave loading factor
- Shallow-water resistance term
- Incompressibility (continuity) penalty

**Training:** 100,000 synthetic samples (100 epochs, AdamW, CosineAnnealingLR) followed by fine-tuning on real NASA SWOT data.

**Deployment:** Exported to ONNX (12.2 KB, max deviation vs PyTorch: 2.98e-08) for the Rust onboard inference engine.

---

## Results

- Average drag score (Black Sea, calm): 0.13-0.14
- Estimated fuel savings: 8-12% depending on conditions
- CII rating improvement: E to D (Black Sea cargo), B to A (Handy bulk)
- Route analysis: Istanbul-Trabzon, Istanbul-Novorossiysk, Odessa-Istanbul, Batumi-Constanta

Fuel consumption figures calibrated against Clarkson Research / MAN Energy Solutions reference data (VLCC: 120 t/day, Panamax: 80 t/day, Capesize: 40 t/day).

---

## Repository Contents

| File | Description |
|---|---|
| 	raining_v2.py | Main PINN training (100K synthetic samples) |
| swot_real.py | Real SWOT NetCDF download and processing |
| 
asa_real_training.py | Fine-tuning on 296K real SWOT measurements |
| gpm_integrate.py | GPM precipitation to SWH integration |
| modis_test.py | MODIS SST to kinematic viscosity correction |
| gpm_modis_real.py | Combined 3-satellite inference pipeline |
| cii_module.py | IMO CII rating calculator (MEPC.354(78)) |
| scenario_test.py | Vessel-type analysis with realistic fuel data |
| 
oute_map.py | Multi-route interactive map generator (folium) |
| onnx_export_v2.py | ONNX export and cross-validation |
| pp.py | Flask web dashboard |

---

## Quick Start

\\ash
pip install torch numpy netCDF4 requests folium matplotlib flask
python training_v2.py        # train the PINN model
python app.py                # launch the web dashboard
# open http://127.0.0.1:5000
\
> NASA data access requires a free [Earthdata account](https://urs.earthdata.nasa.gov/).

---

## Roadmap

- [x] PINN model with physics-based loss functions
- [x] Real NASA SWOT data integration (296K measurements)
- [x] GPM + MODIS integration (3 satellites)
- [x] ONNX export for edge deployment
- [x] IMO CII compliance module
- [x] Flask web dashboard
- [ ] Pilot validation with real vessel voyage data
- [ ] Rust onboard inference engine (complete)
- [ ] AIS real-time position integration
- [ ] NASA SWOT Early Adopters program application

---

## Acknowledgments

- [NASA / CNES](https://swot.jpl.nasa.gov/) - SWOT mission data via PO.DAAC
- NASA - GPM IMERG and MODIS SST products
- [GEBCO](https://www.gebco.net/) - General Bathymetric Chart of the Oceans 2026
- Raissi, Perdikaris, Karniadakis (2019) - Physics-Informed Neural Networks

---

## Author

**Mehmet Sinan** - Independent Researcher, Turkey

*This research uses data from NASA SWOT mission. SWOT data are freely available under a Creative Commons Zero license.*
