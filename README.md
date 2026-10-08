# BATIMETRIX — Thalassa Maritime Intelligence Platform

**Proactive Hydrodynamic Drag Prediction Engine for Maritime Fuel Optimization**

🌐 **Live Demo: [batimetrix.onrender.com](https://batimetrix.onrender.com)**
📦 **GitHub: [github.com/Batimetrix/batimetrix](https://github.com/Batimetrix/batimetrix)**

Batimetrix predicts hydrodynamic drag 3 nautical miles ahead of a vessel's position using a physics-informed neural network fine-tuned on real NASA SWOT measurements, and monitors **10 NASA satellite data sources** for data availability.

**Target: 8-12% net fuel savings and measurable IMO CII rating improvements.**

Built from scratch by an 18-year-old independent researcher from Turkey.

---

## Positioning & Related Work

Vessel performance prediction and voyage optimization are established fields. Companies such as **Bearing AI**, **ZeroNorth** and **StormGeo** already predict fuel consumption and optimize voyages using weather, oceanographic and vessel data, and some report high accuracy against real fleet data.

Batimetrix does not claim to replace these systems. It explores a narrower research question:

> **Can satellite ocean observations — in particular SSH-derived surface currents from NASA SWOT and Sentinel-6 — combined with a physics-informed neural network, improve hydrodynamic drag prediction ahead of a vessel's position?**

What Batimetrix currently offers:

| Component | Status |
|---|---|
| Physics-informed neural network (ITTC 1957, Reynolds, shallow-water terms) | Implemented, fine-tuned on real SWOT data |
| NASA satellite data source monitoring (10 sources) | Implemented |
| IMO CII / EU ETS / FuelEU calculators | Implemented (based on estimated savings) |
| 60 trade routes, 15 vessel classes, fleet dashboard | Implemented |
| Live SSH inputs at route waypoints | Implemented (NASA-SSH simple grid, weekly, 0.5°; 181/188 waypoints) |
| SSH gradient → geostrophic current | Computed (anomaly component, 118/188 waypoints); not yet a model input |
| Validation against real vessel fuel data | Planned (seeking pilot shipowner) |
| Open source | Yes |



---

## Data Sources — 10 NASA Satellites & Data Sources

| Satellite | Measurement | Last Data | Intended role |
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
| NISAR (NASA/ISRO) | SAR Maritime + Wake Detection | 2026-09-20 | Status monitoring; wake detection planned |

The model was fine-tuned on **296,526 real SWOT measurements** over the Black Sea. Other sources are currently monitored for data availability; integrating them as live model inputs is part of the roadmap.

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

Deployment: Exported to ONNX (12.2 KB graph + 6.6 MB external weights, max deviation vs PyTorch: 2.98e-08).

---

## Platform Features

- **60 global trade routes** — Black Sea, Mediterranean, Suez, Cape, Arctic NSR, Trans-Pacific, Trans-Atlantic
- **15 vessel classes** — VLCC, Suezmax, Aframax, MR, LNG, Capesize, Panamax, ULCV, Feeder and more
- **Fleet Dashboard** — multi-vessel portfolio analysis
- **Ocean Intel** — 16 global maritime zones with live SSH anomaly values (NASA-SSH, weekly; 15/16 zones)
- **3D Globe** — interactive visualization of live SSH anomaly values (weekly)
- **IMO CII** — MEPC.354(78) compliant rating calculation
- **7 languages** — EN, TR, EL, ZH, RU, ES, FR
- **PDF report** — downloadable analysis report
- **Palantir-style sidebar** — professional intelligence platform UI
- **EU ETS Carbon Calculator** — 2026 scope (100% of emissions)
- **FuelEU Maritime Calculator** — GHG intensity compliance
- **Speed Optimization Wizard** — NASA SSH-adjusted optimal speed
- **Execution Gap Analyzer** — planned vs actual voyage analysis
- **Route Comparison** — dual route efficiency analysis
- **NISAR SAR Integration** — *planned*: vessel wake detection (NASA/ISRO)

---

## Current Estimates & Assumptions

- Average model drag score (Black Sea, calm): 0.13-0.14
- Fuel savings shown in the platform (8-12%) are **estimates**, derived from the drag score through an assumed linear conversion factor and bounded to an 8-15% range. They are not yet measured results.
- CII rating changes (e.g. E→D, B→A) are computed from these estimated savings.
- Baseline daily fuel consumption per vessel class is based on Lloyd's List / MAN Energy Solutions reference values.

---

## Limitations & Roadmap

Batimetrix is an early-stage research platform. Known limitations:

- **No real-world validation yet.** Training targets are generated from the physics model; the network has not been validated against measured vessel drag or fuel consumption. Next step: validation against ship noon-report data (daily position, speed, fuel) from a pilot shipowner.
- **Coarse SSH resolution.** Route waypoints now use live sea surface height anomaly from the NASA-SSH simple gridded product (weekly, 0.5°). This resolution cannot resolve narrow straits or coastal areas; 7 of 188 waypoints fall back to reference values. Next step: higher-resolution SWOT data near straits.
- **Physical interpretation.** SSH itself has a small direct effect on hull resistance; the meaningful link is through SSH gradients, which indicate geostrophic surface currents. Anomaly geostrophic currents are now derived from SSHA gradients, but they represent deviations from the mean circulation, not total currents, and are not yet a model input. Next step: add mean dynamic topography and retrain the model with current as an input.
- **NISAR.** SAR-based vessel wake detection and surface roughness estimation are planned, starting with NISAR-formatted Sentinel-1 and simulated-NISAR UAVSAR products.

Feedback and collaboration are welcome.

---

## Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch PINN → ONNX export |
| Backend | Python / Flask |
| Frontend | HTML/CSS/JS, Leaflet.js |
| NASA Data | PO.DAAC CMR API (10 sources) |
| Deployment | Render.com |

---

## Contact

**Mehmet Sinan Bilgeç**
mbilgec@asu.edu
Independent Researcher | ASU Online Student Turkey

---

*Batimetrix is the foundation of Thalassa — a maritime intelligence platform vision combining NASA satellite data, physics-informed AI, and ocean analytics for commercial and defense applications.*
