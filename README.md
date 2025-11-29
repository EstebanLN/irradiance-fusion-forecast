# 🌞 Multi-Source Deep Learning for Solar Irradiance Forecasting in Colombia

This repository contains the code and documentation for a project on **short-term solar irradiance forecasting (GHI)** for a reference photovoltaic site in Colombia.

The work focuses on forecasting **global horizontal irradiance (GHI)** at a **60-minute horizon** using:
- Ground-based tabular data (irradiance and meteorological measurements, plus engineered features and solar geometry).
- Models that include both purely tabular approaches and hybrid architectures that incorporate information derived from satellite products.

The repository is organized to support a reproducible pipeline: data preprocessing, feature engineering, model training, and reporting of results.

**Goal**: Enable photovoltaic (PV) plants to minimize penalties in Colombia’s intraday energy market by improving the reliability of solar generation forecasts.

---

## Motivation

Colombia is expanding its renewable energy capacity, especially through solar PV. However, regulatory frameworks (e.g., CREG 060 of 2019) impose strict penalties for deviations between scheduled and actual energy delivery. Accurate irradiance forecasting is therefore crucial to reduce risk and economic loss.

Existing models relying solely on tabular or numerical weather prediction data (e.g., GFS) often fail to capture rapid local atmospheric transitions. This project explores the integration of satellite and sky image data with local sensor readings to improve forecast accuracy.

---

## Core Ideas

- **Multi-source data fusion**: Combine tabular, image, and satellite data streams.
- **Deep learning**: Leverage architectures like LSTM, Bi-LSTM, CNN-LSTM.
- **Spatiotemporal modeling**: Capture cloud movement, atmospheric trends, and site-specific weather patterns.
- **Regulatory context**: Reduce market penalties by enhancing forecast resolution and precision.

---

## Project Structure

```bash
irradiance-fusion-forecast/
├── Notebooks/         # Jupyter notebooks for exploration, preprocessing and modeling
│
├── data_raw/          # Original input data as obtained from external sources
├── data_interim/      # Intermediate datasets after cleaning / alignment
├── data_processed/    # Final modeling-ready datasets (e.g. train/val/test splits)
│
├── models/            # Model definitions, training scripts and saved artifacts
│
├── reports/           # Figures, tables and manuscript-oriented outputs
│
├── .gitignore
├── environment.yml    # Conda environment specification
├── requirements.txt   # Python dependencies (pip)
└── README.md
```

This repository does not include real datasets for confidentiality and size reasons.
The folder structure mirrors the structure used during development so that the pipeline remains fully reproducible if data is provided.
