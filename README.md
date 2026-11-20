# BATIMETRIX — Thalassa Maritime Intelligence Platform

**Proactive Hydrodynamic Drag Prediction Engine for Maritime Fuel Optimization**

🌐 **Live Demo: [batimetrix.onrender.com](https://batimetrix.onrender.com)**
📦 **GitHub: [github.com/Batimetrix/batimetrix](https://github.com/Batimetrix/batimetrix)**

Batimetrix predicts hydrodynamic drag 3 nautical miles ahead of a vessel's position using a physics-informed neural network trained on physics-based synthetic data, with live NASA sea surface height and surface current inputs, uses live NASA sea surface height and surface current data at every route waypoint (updated automatically each week), and monitors **10 NASA satellite data sources** for data availability.

**Target: 8-12% net fuel savings and measurable IMO CII rating improvements.**

Built from scratch by an 18-year-old independent researcher from Turkey.

---

## Positioning & Related Work

Vessel performance prediction and voyage optimization are established fields. Companies such as **Bearing AI**, **ZeroNorth** and **StormGeo** already predict fuel consumption and optimize voyages using weather, oceanographic and vessel data, and some report high accuracy against real fleet data.

Batimetrix does not claim to replace these systems. It explores a narrower research question:

> **Can satellite ocean observations — in particular sea surface height (NASA SWOT, NASA-SSH) and satellite-derived surface currents (NASA OSCAR) — combined with a physics-informed neural network, improve hydrodynamic drag prediction ahead of a vessel's position?**

What Batimetrix currently offers:

| Component | Status |
|---|---|
| Physics-informed neural network (ITTC 1957, Reynolds, shallow-water terms) | Implemented; trained on physics-based synthetic data (v3 adds surface current) |
| NASA satellite data source monitoring (10 sources) | Implemented |
| IMO CII / EU ETS / FuelEU calculators | Implemented (based on estimated savings) |
| 60 trade routes, 15 vessel classes, fleet dashboard | Implemented |
| Live SSH inputs at route waypoints | Implemented (NASA-SSH simple grid, weekly, 0.5°; 181/188 waypoints) |
| Surface currents | Implemented: total surface current from NASA OSCAR (daily, 0.25°; 188/190 points), used as a model input via PINN v3 (along-track current) and shown in the telemetry table |
| Validation of satellite currents against real ship speeds (AIS) | Done: Florida Straits, 11 days in 2025, see Early Validation |
| Validation against real vessel fuel data | First test on one public cruise ship dataset (FuelCast, research-only licence); pilot shipowner still needed |
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

**Live inputs used in the platform today:**

| Dataset | Measurement | Update | Role in Batimetrix |
|---|---|---|---|
| NASA-SSH Simple Gridded SSH (V1.1) | Sea surface height anomaly, 0.5° | Weekly | SSH input at 181/188 waypoints and 15/16 Ocean Intel zones |
| OSCAR v2.0 NRT (ESR / NASA) | Total surface current, 0.25° | Daily (refreshed weekly) | Current speed at 188/190 points, shown in telemetry |

SWOT data (296,526 measurements over the Black Sea) was used in an early training step; see Model Architecture for its limits. Other sources are currently monitored for data availability; integrating them as live model inputs is part of the roadmap.

---

## Early Validation: Satellite Currents vs. Real Ship Speeds

**Question:** Do satellite-derived surface currents (NASA OSCAR) explain real changes in vessel speed?

**Data:** NOAA MarineCadastre AIS (U.S. waters, 2025, 1-minute positions) in the Florida Straits (23.5-27.5 N, 81.5-79 W), where the Gulf Stream flows north at up to about 2 m/s, and OSCAR v2.0 daily surface currents (0.25 deg).

**Method:** Large cargo ships and tankers (100 m or longer, AIS types 70-89) under way at 8-25 kn, excluding positions within 15 km of major ports. For each position, the current component along the ship's course was computed. Within each vessel, deviations in speed over ground were regressed on deviations in along-track current, so every ship is compared only with itself. One day per month, January-November 2025.

**Results:**
- On all 11 days the effect has the expected sign: 0.42-0.85 kn of speed change per 1 kn of current (all days: 0.59, r = 0.41, 330,966 positions, 78-125 vessels per day).
- Northbound ships (with the Gulf Stream) were 1.2-3.4 kn faster than southbound ships on every day; the gap was among the smallest on the day with the weakest current (Oct 15).

![Validation chart](docs/validation_slope_2025.png)

### Speed through water (speed over ground minus satellite current)

- **North-south gap:** northbound ships were on average 2.5 kn faster
  over ground; after subtracting the OSCAR along-track current the gap is
  0.7 kn (mean absolute, 11 days). About 73% of the gap is explained
  by the satellite current.
- **Full subtraction over-corrects:** on 9 of 11 days southbound ships
  then appear faster. This matches the slope below 1. One possible cause
  (not yet tested): 0.25 deg OSCAR cells blur the sharp western edge of the
  Gulf Stream, which southbound ships tend to avoid.
- **Within-vessel speed variance:** subtracting the full current reduces it by
  8.3% on average (range -9.4 to 37.0%). Subtracting a
  scaled current, with the scale learned on the other 10 days
  (leave-one-day-out), reduces it by 15.9%
  (range 8.9 to 33.8%).
- **Implication:** currents should enter the drag model as a learned input,
  not as a direct subtraction.

Table: `validation/stw_2025.csv`.

### Replication: Cape Hatteras (North Carolina)

The same analysis was repeated in a second, independent region where the
Gulf Stream leaves the U.S. coast (33.5-36.5 N, 77-74 W), same 11 days.

- Slope positive on all 11 days: 0.4 to 1.47
  kn per kn of current (mean of days 0.82).
- Cross-validated within-vessel speed variance reduction: 23.4%
  on average; clearly positive on 9 of 11 days, near zero on the rest.
- Fewer vessels than in the Florida Straits (23-40
  per day), so day-to-day results are noisier. The Gulf Stream meanders
  strongly here; on days with a small north-south speed gap ships were
  probably outside its core.

Table: `validation/hatteras_2025.csv`.

### First fuel test: one cruise ship (FuelCast dataset)

**Data:** CPS Poseidon from the public FuelCast dataset (Krohne Digital; Hugging Face
`krohnedigital/FuelCast`; licence CC BY-NC-ND 4.0, used here for non-commercial research only and
not redistributed). 12 months of 5-minute onboard data: flowmeter fuel consumption, GPS speed over
ground (SOG), speed log speed through water (STW), heading, and model ocean currents from Open-Meteo.
60,318 under-way records. Vessel position is not included.

**Test 1, current check:** the current measured by the ship (SOG - STW) correlates with the
Open-Meteo along-track current: r = 0.60, slope 0.62 (assuming current in m/s).

**Test 2, fuel prediction:** fuel rate modelled from speed and wave height (cubic in speed),
cross-validated over 5 consecutive time blocks:

| Speed used | Fuel error (RMSE, % of mean) |
|---|---|
| SOG only | 11.57% |
| SOG - 0.7 x along-track current (fixed k, as in the platform) | 9.18% |
| SOG - 1.0 x along-track current (fixed k, physics) | 8.91% |
| SOG - k x current, k fitted per block (0.90-0.95) | 8.93% |
| Measured STW (speed log) | 9.24% |

Accounting for the current reduced the fuel prediction error by about 21-23% without fitting k.
Current-corrected SOG did slightly better than the speed log, which reads on average 0.23 m/s
above SOG; this probably reflects speed log bias rather than better-than-perfect information.

**Limits:** one cruise ship; total fuel includes hotel load; currents from Open-Meteo, not NASA
OSCAR; position unknown; current units assumed. Open question: propulsion shaft power did not
show the same improvement (errors about 38% for all speed inputs, dominated by other factors).
Code: `fuelcast_test.py`, `fuelcast_robust.py`.

Full table: `validation/results_2025.csv`. Code: `validation/validate_currents.py`.

**Interpretation and limits:** A slope below 1 is expected because OSCAR smooths currents (0.25 deg, daily, top 30 m) and ships adjust engine settings. This validates the importance of satellite currents for vessel speed; it does not yet measure fuel savings. Consecutive AIS positions are correlated, so statistical precision is overstated; the consistency across 11 days is the main evidence. The Gulf Stream is exceptionally strong, so effects in the Black Sea and Baltic will be smaller.

---

## Model Architecture

**Physics-Informed Neural Network (PINN) — 1,657,025 parameters**

- Input: 7 features (lat, lon, depth, SSH anomaly, SWH, speed, draft)
- 6 residual blocks, 512 neurons each, GELU activation, LayerNorm
- Output: normalized drag score in [0, 1]

**PINN v3: current-aware (November 2026)**: `training_v3.py`, `batimetrix_v3_current.pt`
- Adds an 8th input: surface current along the ship's track (kn).
- Target: drag per nautical mile over ground, `drag x STW / SOG`, with `SOG = STW + k x current`.
- `k = 0.7` was measured from real AIS data against NASA OSCAR currents (Florida Straits 0.59,
  Cape Hatteras 0.82; see Early Validation). It is a calibration for OSCAR currents, not a physical
  constant, and may partly reflect ships adjusting engine power.
- Physics constraint in the loss: more following current must not increase drag per nm.
- An unphysical latitude factor was removed from the synthetic training targets.
- In the platform, v3 provides a current adjustment factor (drag with current / drag with zero
  current) that multiplies the existing drag score, so existing thresholds remain valid.
- Check against the formula: the model gets the direction right at all tested currents but
  overestimates the effect of following currents (about -15% instead of -10% at +2 kn).

Physics constraints in the loss function:
- ITTC 1957 friction line: `Cf = 0.075 / (log10(Re) - 2)^2`
- Reynolds number scaling
- Morison-type wave loading factor
- Shallow-water resistance term
- Incompressibility (continuity) penalty

Training (v2): 100,000 synthetic samples generated from the physics formulas above (100 epochs, AdamW, CosineAnnealingLR). A second step used 10,000 points sampled from 296,526 SWOT measurements over the Black Sea, but its targets were computed from the same formula with fixed speed, draft and depth, so it adds little real-world information. All training targets are synthetic; the model has not yet been trained on measured vessel data.

Deployment: the live platform runs the PyTorch models directly. An ONNX export (12.2 KB graph + 6.6 MB external weights, max deviation vs PyTorch 2.98e-08) exists but is not used in production.

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

- Model drag scores depend on vessel type, speed, draft and sea state; current outputs are visible in the platform. Status labels are relative, not absolute: EFFICIENT / NOMINAL / HIGH DRAG thresholds (0.52 / 0.65) are set at the 33rd and 66th percentiles of model outputs across all 60 routes under typical speeds, drafts and sea states.
- Fuel savings shown in the platform are **estimates**, derived from the drag score through an assumed linear conversion factor and capped at 15% (no minimum, so high-drag routes show lower savings). They are not yet measured results.
- CII rating changes (e.g. E→D, B→A) are computed from these estimated savings.
- Baseline daily fuel consumption per vessel class is based on Lloyd's List / MAN Energy Solutions reference values.

---

## Limitations & Roadmap

Batimetrix is an early-stage research platform. Known limitations:

- **Limited fuel validation.** A first test on one cruise ship (FuelCast) shows that current-corrected speed improves fuel prediction. Satellite currents have been validated against real ship speeds (see Early Validation), but the drag model itself has not yet been validated against measured vessel drag or fuel consumption; its training targets are generated from the physics model. Next step: validation against ship noon-report data (daily position, speed, fuel) from a pilot shipowner.
- **Coarse SSH resolution.** Route waypoints now use live sea surface height anomaly from the NASA-SSH simple gridded product (weekly, 0.5°). This resolution cannot resolve narrow straits or coastal areas; 7 of 188 waypoints fall back to reference values. Next step: higher-resolution SWOT data near straits.
- **Physical interpretation.** SSH itself has a small direct effect on hull resistance; the meaningful link is through SSH gradients, which indicate geostrophic surface currents. Total surface currents now come from NASA OSCAR v2.0 (geostrophic + Ekman + thermal wind, top 30 m, daily, 0.25°). Since PINN v3 they enter the drag estimate as along-track current. OSCAR (0.25 deg) cannot resolve narrow straits: at the Istanbul Strait it shows a weak northward current, while the real surface flow is southward.
- **SSH sensitivity and training assumptions.** The model takes SSH anomaly (ssha) as input, normalized as (ssha + 2) / 4. Real anomalies are typically within +/-0.1 m, so this input varies very little and has only a small effect on predictions. In the training data, wave height was estimated from SSH with an assumed relation (1.5 + 0.3 * |ssha|) rather than measured. Next step: retrain using measured significant wave height (swh_karin from SWOT) and a normalization matched to the real SSH range.
- **NISAR.** SAR-based vessel wake detection and surface roughness estimation are planned, using calibrated NISAR L-band products now publicly available through the ASF DAAC.

Feedback and collaboration are welcome.

---

## Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch PINN → ONNX export |
| Backend | Python / Flask |
| Frontend | HTML/CSS/JS, Leaflet.js |
| NASA Data | PO.DAAC CMR API (10 sources) |
| Data access | NASA Earthdata / earthaccess (PO.DAAC) |
| Automation | GitHub Actions — weekly NASA data refresh and redeploy |
| Deployment | Render.com |

---

## Contact

**Mehmet Sinan Bilgeç**
mbilgec@asu.edu
Independent Researcher | ASU Online Student Turkey

---

*Batimetrix is the foundation of Thalassa — a maritime intelligence platform vision combining NASA satellite data, physics-informed AI, and ocean analytics for commercial and defense applications.*

---

## Data Credits

- Sea surface height: NASA-SSH Simple Gridded SSH (NASA PO.DAAC), CC BY 4.0.
- Surface currents: OSCAR v2.0 Near Real Time, Earth & Space Research / NASA PO.DAAC, doi:10.5067/OSCAR-25N20.
- SWOT data: NASA / CNES via NASA PO.DAAC.
- AIS (validation): Office for Coastal Management, Nationwide Automatic Identification System 2025, https://www.fisheries.noaa.gov/inport/item/77594 (accessed November 2026).
