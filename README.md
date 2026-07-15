# Hourly Electricity Demand Forecasting — Multi-Model Benchmarking Study
 
**Authors:** Maximillian Ermler & Hamzah Harsani  
**Oversight: Prof. Eunshin Byon

---

## Project Overview

Compared classical statistical models, shrinkage estimators, robust regression methods, and deep learning against 50,000+ hours of real hourly electricity demand data from Toledo Edison (FirstEnergy), covering January 2019 through February 2026. The goal was to identify the best-performing forecasting approach for day-ahead utility load prediction and quantify the real-world cost of forecast error.

---

## Data

- **Load data:** Toledo Edison hourly demand by customer class (Residential, Commercial, Industrial) sourced from Ohio PUCO filings — 62,784 raw rows
- **Weather data:** Open-Meteo reanalysis, 19 hourly variables (temperature, humidity, wind, radiation, soil moisture, VPD, etc.) for the Toledo grid cell
- **Final modeling dataset:** 50,235 hourly rows × 21 columns after cleaning and COVID window removal (March 2020 – June 2021 dropped)

---

## Models Tested

| Approach | Models |
|----------|--------|
| Diagnostic Regression (A1) | OLS with B-spline temperature, holiday/heatwave flags, lag-1 term |
| Estimator Benchmarking (A2) | OLS, Ridge, LASSO, PCR, WLS, Huber, LAD, LTS, SARIMA, SARIMAX |
| Deep Learning (A3) | Two-layer PyTorch LSTM, hidden size 200, sequence length 24 |

---

## Key Results

| Model | Test RMSE |
|-------|-----------|
| LSTM (full features) | **78,432 KW** |
| Huber (best linear) | 86,528 KW |
| OLS / Ridge / LASSO | ~87,000 KW |
| SARIMAX | 318,583 KW |
| SARIMA (baseline) | 337,752 KW |

- LSTM outperformed the best linear model by ~10%
- Full-feature LSTM was 77% better than the SARIMA baseline
- At PJM 2025 LMP of $50.73/MWh, the improvement represents ~$115M/yr in reduced energy-market exposure

---

## Tools

- **R** — OLS, Ridge, LASSO, PCR, WLS, Huber, LAD, LTS, SARIMA, SARIMAX, EDA, diagnostics
- **Python / PyTorch** — LSTM deep learning model
- **Libraries:** tidyverse, forecast, glmnet, MASS, splines, torch

---

## Files

| File | Description |
|------|-------------|
| `Approach1_Full_Final.R` | R code for diagnostic regression pipeline |
| `IOE562proj_DeepLearning.py` | PyTorch LSTM implementation |
| `IOE562_Final_Presentation_v2.pdf` | Final presentation slides |
| `IOE562_Final_Report_v3.docx` | Full written report |
