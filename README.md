# Multi-Source Deep Learning for Solar Irradiance Forecasting in Colombia

This project aims to develop a deep learning-based forecasting model that combines heterogeneous data sources — including satellite imagery (GOES, NOAA GFC), ground-based sky images, and local meteorological measurements — to accurately predict short-term solar irradiance in Colombia.

⚡ **Goal**: Enable photovoltaic (PV) plants to minimize penalties in Colombia’s intraday energy market by improving the reliability of solar generation forecasts.

---

## 🌞 Motivation

Colombia is expanding its renewable energy capacity, especially through solar PV. However, regulatory frameworks (e.g., CREG 060 of 2019) impose strict penalties for deviations between scheduled and actual energy delivery. Accurate irradiance forecasting is therefore crucial to reduce risk and economic loss.

Existing models relying solely on tabular or numerical weather prediction data (e.g., GFS) often fail to capture rapid local atmospheric transitions. This project explores the integration of satellite and sky image data with local sensor readings to improve forecast accuracy.

---

## 🧠 Core Ideas

- **Multi-source data fusion**: Combine tabular, image, and satellite data streams.
- **Deep learning**: Leverage architectures like LSTM, Bi-LSTM, CNN-LSTM.
- **Spatiotemporal modeling**: Capture cloud movement, atmospheric trends, and site-specific weather patterns.
- **Regulatory context**: Reduce market penalties by enhancing forecast resolution and precision.

---

## 🗂 Project Structure

```bash
irradiance-fusion-forecast/
├── data/             # Local datasets (raw/interim) — typically gitignored
├── models/           # Trained weights / checkpoints — typically gitignored
├── reports/          # Reports and generated outputs
│   └── figures/      # Plots/figures (consider only committing finals)
├── Notebooks/        # Jupyter notebooks
├── environment.yml   # Primary (Conda) environment spec
├── requirements.txt  # Optional pip fallback (if you need it)
└── README.md

