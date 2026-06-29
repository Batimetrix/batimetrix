# Batimetrix 🚢

**Proactive Hydrodynamic Drag Prediction Engine for Autonomous Ships**

Batimetrix predicts underwater drag resistance **3 nautical miles ahead** 
of a vessel's current position — before the ship reaches the coordinate.
This proactive approach enables real-time propeller pitch and engine RPM 
optimization, achieving **12-15% net fuel savings**.

---

## What Makes Batimetrix Different

| Existing Systems | Batimetrix |
|---|---|
| Reactive — optimize at current position | Proactive — predict 3 miles ahead |
| Generic weather data | NASA SWOT satellite (real measurements) |
| No bathymetry integration | GEBCO global ocean floor mapping |
| Cloud dependent | On-board Rust engine, zero latency |

---

## Data Sources

- **NASA SWOT Satellite** — Real ocean surface height measurements
  (296,526 actual KaRIn radar measurements used in training)
- **GEBCO 2026** — Global bathymetry at 15 arc-second resolution

---

## Architecture
