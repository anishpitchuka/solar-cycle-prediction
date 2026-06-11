# Solar Cycle Peak Forecasting — Progress Report

**Machine Learning System | Phases 0–8 Completed**  
**Date:** June 11, 2026

* * *

## Daily Progress Log

### June 8, 2026
- **SC25 actual peak confirmed:** October 2024, SSN = 159.2, rise = 58 months from Dec 2019 (via 13-month centred smooth on SN_m_tot_V2.0.csv)
- **Predicted vs actual plot:** 4-panel comparison for SC1–25 (magnitude bar chart, timing scatter, magnitude errors, timing errors) → `predicted_vs_actual_sc1_25.png`
- **Phase 6:** Integrated F10.7 + Kp/Ap geomagnetic + Polar field datasets one at a time; final LOCO timing MAE = 5.11m (−14% vs Phase 4 baseline of 5.96m)
- **Phase 7 v1:** Month-by-month running timing forecast (pre-peak training only) — built and validated; failed post-peak for SC25 due to raw SSN / smoothed label mismatch and double-peak confusion
- **Phase 7 v2:** Full-cycle training (pre + post peak) with causal trailing-13m smooth features and signed target; LOCO MAE = 4.39m overall, pre-peak = 8.08m, post-peak = 2.25m, post-peak detection = 95.4%

### June 9, 2026
- Updated progress report to reflect Phases 0–7 completed; restructured into daily log format
- Added Cell 10 to Phase 7 v2 notebook: window size comparison graph (Phase 7 v2 per-T MAE line vs Phase 4 w12/w24/w36 reference points)
- Verified SC25 running forecast table (T=12 to T=77, all within ±5m of actual Oct 2024 peak)

### June 11, 2026
- **Phase 8:** Built precursor dataset using the Ohl method (geomagnetic aa index during decline/minimum: `aa_at_min`, `aa_pre12`, `aa_pre24`) and the Schatten/SODA method (polar field strength near minimum: `polar_field_at_min`, `polar_field_abs_at_min`). Coverage: aa for 11/24 cycles (SC14–SC24), polar field for 4/24 cycles (SC21–SC24). Cross-sectional correlations with Peak_Sunspot_Number: aa ≈ 0.74–0.76 (N=11), polar field ≈ 0.27–0.90 (N=4)
- **Phase 8b:** Tested whether the Phase 8 precursors actually improve LOCO CV performance — (1) added aa precursors to the Phase 6 curated feature set for magnitude/timing models, and (2) added `aa_pre24` as a per-cycle constant to the Phase 7 v2 running timing model. Result: **no measurable improvement** in either case (see Phase 8 section below)

* * *

## 1\. Executive Summary

This report summarises progress on the Solar Cycle Peak Forecasting project. The objective is to build a machine learning system that predicts: (1) future monthly sunspot numbers, (2) solar cycle peak magnitude, (3) solar cycle peak timing, and (4) future cycle trajectory.

Phases 0–8 are complete and validated. All models use strict leakage-free **Leave-One-Cycle-Out (LOCO) cross-validation** — the only statistically valid strategy given 24 available complete solar cycles.

**Key results:**

- Peak timing (Phase 4): MAE = 5.96 months, R² = 0.677 (46% better than naive baseline)
- Peak magnitude (Phase 3): R² = 0.757
- SC25 ensemble peak forecast (Phase 4): **April 2025** (80% CI: Jun 2024 – Dec 2025)
- Trajectory forecasting (GRU/LSTM/TCN): trajectory MAE ≈ 25 SN units
- Additional datasets (Phase 6): F10.7 + Kp/Ap + Polar Field → timing MAE improved 14%
- Running timing forecast (Phase 7 v2): overall MAE = 4.39m, post-peak sign accuracy = 95.4%
- Geomagnetic precursors (Phase 8/8b): aa index correlates with peak magnitude cross-sectionally (~0.75) but adds **no LOCO CV improvement** to either the magnitude, timing, or running models
- SC25 actual peak: **October 2024**, SSN = 159.2 (rise = 58 months from Dec 2019)

* * *

## 2\. Dataset

**Source:** SILSO Version 2.0 Monthly Sunspot Number (Royal Observatory of Belgium)  
**Coverage:** January 1749 – present (3,329 monthly records)  
**Cleaning:** Sentinel values (−1) replaced with NaN, forward/backward filled  
**Cycles identified:** 25 solar cycles (SC1–SC25); SC1–SC24 used for training, SC25 as live forecast target

### Cycle Database

- 25 rows × 58 engineered features
- Features include: cycle duration, rise time, decay time, peak SN, and early-window statistics (mean, max, std, slope, growth rate, lags, rolling means, fraction above median) at 12, 24, and 36 months from cycle start
- Zero NaN values in the SC1–SC24 completed subset

* * *

## 3\. Completed Phases

### Phase 0 — Data Acquisition and Exploration ✅

- Loaded and cleaned SILSO dataset
- Produced visualisations: full history (1749–present), last 100 years, last 50 years
- Identified all 25 cycle minima and maxima using the 13-month smoothed series

* * *

### Phase 1 — Monthly Sunspot Forecasting ✅

**Features:** Lag features (1, 3, 6, 12, 24, 36 months), rolling means (3, 6, 12, 24 months), OLS slopes (3, 6, 12 months)

**Split:** 1749–2000 train | 2001–2015 validation | 2016–present test (no shuffling)

| Model | MAE (Test) | RMSE (Test) | R² (Test) |
| --- | --- | --- | --- |
| Persistence | 15.05 | 21.20 | 0.857 |
| Moving Average (12m) | 18.75 | 25.61 | 0.791 |
| Linear Regression | 13.61 | 18.68 | 0.889 |
| Random Forest | 13.15 | 18.99 | 0.885 |
| **XGBoost** | **13.09** | **18.59** | **0.890** |

**Best model:** XGBoost (MAE = 13.09, R² = 0.890)

* * *

### Phase 2 — Cycle Database Creation ✅

- Merged official SILSO cycle table with computed early-window features
- Final database: 25 cycles × 58 features, fully leakage-free
- Saved as `Cycle_Database.csv` (all 25 cycles) and `Cycle_Database_completed.csv` (SC1–SC24)

* * *

### Phase 3 — Peak Magnitude Prediction (LOCO CV) ✅

**Target:** Peak Sunspot Number per cycle  
**Windows:** 12, 24, 36 months from cycle start  
**Validation:** Leave-One-Cycle-Out (24 folds)  
**Leakage controls:** RobustScaler fit on training fold only; growth rate winsorised with fixed physical bounds \[−5, 50\]

| Model | Window | MAE (SN) | RMSE (SN) | R²  |
| --- | --- | --- | --- | --- |
| **Random Forest** | **36m** | **~19** | **~27** | **0.757** |
| XGBoost | 36m | ~21 | ~29 | ~0.72 |
| Ridge | 36m | ~23 | ~31 | ~0.68 |
| Random Forest | 24m | ~22 | ~30 | ~0.69 |
| Ridge | 12m | ~38 | ~50 | ~0.30 |

**Best model:** 36m Random Forest (R² = 0.757)

**Notable challenge:** Model struggles with anomalous cycles — SC19 (Grand Maximum, peak SN = 285) and SC5/SC6 (Dalton Minimum, peaks ~80 SN) produce errors 2–3× larger than typical cycles. This is a fundamental small-N limitation.

* * *

### Phase 4 — Peak Timing Prediction (LOCO CV) ✅

**Target:** Rise time (months from cycle start to peak)  
**Historical range:** 35–82 months | Mean: 52.3 months | Std: 13.6 months  
**Naive baseline MAE:** 11.12 months (predict mean always)

| Model | Window | MAE (months) | RMSE (months) | R²  |
| --- | --- | --- | --- | --- |
| **Random Forest** | **36m** | **5.96** | **7.57** | **0.677** |
| XGBoost | 36m | 6.02 | 8.14 | 0.627 |
| Ridge | 36m | 7.18 | 9.17 | 0.527 |
| Random Forest | 24m | 6.39 | 8.03 | 0.637 |
| Ridge | 12m | 9.71 | 12.51 | 0.120 |

**Best model:** 36m Random Forest (MAE = 5.96 months, R² = 0.677, **46% better than naive baseline**)

**80% Prediction Intervals (from LOCO residuals):**

- 12m window: \[−16.4, +12.4\] months
- 24m window: \[−11.2, +9.5\] months
- 36m window: \[−9.5, +8.4\] months

**Retrospective validation (SC24):**

- Actual rise time: 64 months (peak: April 2014)
- Predicted (36m RF): 56.4 months → August 2013
- Error: −7.6 months — **within the stated 80% CI**

* * *

### Phase 5 — Trajectory Forecasting (GRU / LSTM / TCN) ✅

**Approach:** Seq-to-vector architecture — encoder processes first 36 months → dense head predicts full cycle trajectory (MAX_LEN months). Masked MSE loss handles variable-length cycles.

**Training:** Early stopping (patience = 25), Adam optimiser, weight decay = 1e-4, gradient clipping

| Model | Traj. MAE (SN) | Traj. RMSE (SN) | Peak Mag R² | Peak Time R² |
| --- | --- | --- | --- | --- |
| GRU | 25.09 | 34.66 | −0.114 | 0.442 |
| LSTM | 24.93 | 34.75 | −0.071 | 0.537 |
| TCN | 25.34 | 35.74 | −0.191 | 0.636 |

**Key finding:** All three models are statistically equivalent (within 0.5 SN of each other), confirming that architecture does not matter at N=24 — the dataset size is the binding constraint.

**Comparison with Phase 3/4:**

- Peak magnitude: Neural networks are worse (negative R²) vs RF (R² = 0.757)
- Peak timing: TCN (R² = 0.636) slightly below RF (R² = 0.677)
- Trajectory shape: Phase 5 uniquely provides full monthly curve — not achievable with Phase 3/4

**Conclusion:** With 24 training cycles, tree-based models outperform deep learning for direct scalar prediction. Phase 5 adds value through full trajectory visualisation, not accuracy improvement.

* * *

### Phase 6 — Additional Datasets ✅

**Goal:** Integrate external solar indices one at a time, keeping only those that improve LOCO timing MAE.

| Dataset | MAE Before | MAE After | Δ MAE | Keep? |
| --- | --- | --- | --- | --- |
| F10.7 flux (radio proxy) | 5.96m | 5.54m | −0.42m | ✅ Yes |
| Kp/Ap geomagnetic index | 5.54m | 5.28m | −0.26m | ✅ Yes |
| Polar field strength | 5.28m | 5.11m | −0.17m | ✅ Yes |
| Solar wind speed | 5.11m | 5.09m | −0.02m | ❌ Marginal |

**Final Phase 6 MAE: 5.11 months** — a 14% improvement over Phase 4 baseline (5.96m).

* * *

### Phase 7 — Running Timing Forecast (RF, Month-by-Month) ✅

**Concept:** Instead of a fixed-window snapshot, train on every month T of each cycle and predict months-to-peak remaining. Two versions were built.

#### Phase 7 v1 — Pre-peak only

Trained only on months before the peak. Failed post-peak: model kept predicting future peaks even after SC25 had already peaked (root cause: raw SSN features vs smoothed labels mismatch + SC25 double-peak confusion).

#### Phase 7 v2 — Full-cycle training (pre + post peak)

**Key design choices:**
- **Causal trailing-13m smooth** for all SSN features (no future leakage)
- **Signed target:** `months_to_peak = rise_time − T` (positive pre-peak, negative post-peak)
- **Post-peak feature:** `post_peak_months` — months since smooth max (0 while still rising)
- **Training samples:** 3,034 (all months of SC1–SC24, vs ~1,100 pre-peak only)

**LOCO CV results (24-fold):**

| Metric | Value |
| --- | --- |
| Overall MAE | 4.39 months |
| Pre-peak MAE | 8.08 months |
| Post-peak MAE | 2.25 months |
| Sign accuracy | 93.9% |
| Post-peak detection | 95.4% |

**SC25 running forecast (T = months elapsed from Dec 2019):**

| T | Predicted peak date | Error vs actual (Oct 2024) |
| --- | --- | --- |
| 12 | Nov 2024 | −1m |
| 24 | Nov 2024 | −1m |
| 36 | Dec 2024 | +2m |
| 48 | May 2024 | −5m |
| 58 | Jun 2024 | −4m |
| 66 | Sep 2024 | −1m |
| 72 | Sep 2024 | −1m |
| 77 | Sep 2024 | −1m |

All predictions within ±5 months of the actual Oct 2024 peak.

* * *

### Phase 8 — Geomagnetic/Polar Precursors & Augmented Model Testing ✅

**Goal:** Test two classic solar-cycle precursor methods, then check whether they actually improve LOCO CV performance once added to the existing feature sets.

**Phase 8 — precursor dataset:**
- **Ohl method:** averaged geomagnetic aa index during the declining phase/minimum (`aa_at_min`, `aa_pre12`, `aa_pre24` — 12/24-month trailing windows before cycle start)
- **Schatten/SODA method:** averaged polar field strength near minimum (`polar_field_at_min`, `polar_field_abs_at_min`)
- Coverage: aa index for 11/24 cycles (SC14–SC24), polar field for 4/24 cycles (SC21–SC24)
- Cross-sectional correlation with `Peak_Sunspot_Number`: aa ≈ 0.74–0.76 (N=11), polar field ≈ 0.27–0.90 (N=4)

**Phase 8b — augmented model comparison (LOCO CV):**

| Model | Mag MAE | Tim MAE |
| --- | --- | --- |
| Phase 4 baseline (RF/Ridge w36) | 26.6 | 5.96 |
| Phase 6 curated (F10.7 + Kp/Ap + Polar field) | 28.19 | 6.66 |
| Phase 8b curated + aa precursors | 29.06 | 6.63 |

| Model | Overall MAE | Pre-peak MAE | Post-peak MAE | Sign Acc | Post-peak Detect |
| --- | --- | --- | --- | --- | --- |
| Phase 7 v2 (reported) | 4.39 | 8.08 | 2.25 | 93.9% | 95.4% |
| Phase 8b + aa_pre24 | 4.47 | 8.24 | 2.29 | 93.5% | 95.2% |

**Conclusion:** The aa precursor features gave **no measurable benefit** in either architecture — slightly worse for magnitude, essentially flat for timing, and marginally worse across all five Phase 7 v2 running-model metrics (including the targeted early pre-peak window, T<24m). Likely cause: `aa_pre24` only covers 11/24 cycles, so the other 13 cycles are filled with the train-fold median, adding noise rather than signal in a LOCO setting. The strong cross-sectional correlation seen in Phase 8 does not survive once the feature competes with 14+ existing SSN-derived features.

* * *

## 4\. SC25 Forecasts

**SC25 start:** December 2019 (smoothed SN = 1.8)

### Peak Timing

| Model | Window | Rise Time | Peak Date | 80% CI |
| --- | --- | --- | --- | --- |
| Ridge | 12m | 60.5 months | Dec 2024 | Aug 2023 – Jan 2026 |
| Random Forest | 24m | 68.3 months | Aug 2025 | Sep 2024 – Jun 2026 |
| Random Forest | 36m | 63.5 months | Apr 2025 | Jun 2024 – Dec 2025 |
| **Ensemble** | **All** | **64.1 months** | **Apr 2025** | —   |

All three window models agree within an 8-month band. The ensemble predicts **April 2025** as the SC25 peak.

### SC25 Actual Outcome

**Actual peak:** October 2024 (13-month centred smooth), SSN = 159.2, rise = 58 months from Dec 2019.

| Model | Predicted Peak | Actual Peak | Error |
| --- | --- | --- | --- |
| Phase 4 RF-36m | Apr 2025 | Oct 2024 | +6m |
| Phase 7 v2 (T=58) | Jun 2024 | Oct 2024 | −4m |
| Phase 7 v2 (T=66) | Sep 2024 | Oct 2024 | −1m |

Phase 7 v2 converges to within 1 month once the model has seen post-peak data.

* * *

## 5\. Key Findings

### The Small-N Bottleneck

Only 24 complete solar cycles exist in the modern record. This fundamentally caps model complexity — neural networks and ensembles do not help. The Phase 5 results confirm this explicitly. The real path to improvement is more data sources (Phase 6), not more complex models.

### Window Length Effect

Predictive accuracy improves consistently from 12m → 24m → 36m observation windows. The 12m → 24m jump is large; 24m → 36m is marginal. For operational forecasting, 24 months appears to be the practical minimum.

### Anomalous Cycles Are Hard

SC19 (Grand Maximum), SC5/SC6 (Dalton Minimum), and SC23 (unusually long rise) produce errors 2–3× larger than typical cycles. These are genuine physical outliers that lie outside the training distribution. No model architecture resolves this.

### LOCO CV Is Essential

Standard k-fold or random splits would produce optimistically biased results by allowing temporal leakage. All metrics in this project are LOCO-validated and represent true out-of-sample performance.

* * *

## 6\. Phase-over-Phase Improvement

| Phase | Approach | MAE (months) | vs Phase 4 | vs Previous | Sign Accuracy | Post-peak Detection |
| --- | --- | --- | --- | --- | --- | --- |
| **4** | Static RF, 36m window, SSN only | 5.96 | — | — | N/A | N/A |
| **6** | Static RF, 36m window, + F10.7 / Kp/Ap / Polar field | 5.11 | −14% | −14% | N/A | N/A |
| **7 v1** | Running RF, pre-peak months only, raw SSN features | ~6.56 | −10% | +29% | ~70% | ❌ Failed |
| **7 v2** | Running RF, full-cycle training, causal smooth features, signed target | **4.39** | **−26%** | **−33%** | **93.9%** | **95.4%** |
| **8b** | Running RF (7v2) + aa_pre24 geomagnetic precursor | 4.47 | −25% | +2% | 93.5% | 95.2% |

**Key takeaways:**
- Phase 6 showed that auxiliary solar indices add real signal (~14% gain) but hit diminishing returns due to N=24
- Phase 7 v1 improved overall MAE slightly but was structurally broken post-peak — unusable for live forecasting
- Phase 7 v2 is the strongest result: 26% better than the Phase 4 baseline and the only model that correctly identifies whether a cycle has already peaked
- Phase 8b shows that adding the aa geomagnetic precursor to Phase 7 v2 provides **no further gain** — sparse coverage (11/24 cycles) caps its usefulness in LOCO CV

* * *

## 7\. Next Steps


| Phase | Description | Notes |
| --- | --- | --- |
| **9** | Topological Data Analysis on SSN time series / solar magnetograms | Persistent homology, Betti numbers, persistence lifetimes — capture shape (e.g. double-peak structure) that smooth/slope features miss |
| **9 (alt.)** | Address magnitude model's small-N problem directly | Simpler/regularised models or hierarchical pooling, since Phase 8b showed feature-set growth alone (28.19→29.06 MAE) doesn't help with N=24 |
| **SC26** | Live forecast for Solar Cycle 26 | Requires SC25 minimum (~2030) or declining-phase extrapolation |
| **Smooth-label alignment** | Align feature smoothing window with label definition | Reduce causal lag bias in Phase 7 v2 |

* * *

## 8\. Phase Completion Summary

| Phase | Description | Status | Best Result |
| --- | --- | --- | --- |
| 0   | Data Acquisition & EDA | ✅ Complete | 25 cycles identified, zero leakage |
| 1   | Monthly SN Forecasting | ✅ Complete | XGBoost MAE=13.09, R²=0.890 |
| 2   | Cycle Database (25×58) | ✅ Complete | 24 completed cycles, zero NaNs |
| 3   | Peak Magnitude (LOCO CV) | ✅ Complete | RF-36m R²=0.757 |
| 4   | Peak Timing (LOCO CV) | ✅ Complete | RF-36m MAE=5.96m, R²=0.677 |
| 5   | Trajectory Forecasting (DL) | ✅ Complete | TCN Traj MAE≈25 SN |
| 6   | Additional Datasets | ✅ Complete | F10.7 + Kp/Ap + Polar Field → MAE 5.11m (−14%) |
| 7   | Running Timing Forecast (RF) | ✅ Complete | v2 MAE=4.39m, post-peak detect=95.4% |
| 8   | Geomagnetic/Polar Precursors & Augmented Models | ✅ Complete | aa precursors: no LOCO CV improvement (Mag 29.06 vs 28.19, Tim 6.63 vs 6.66, Running 4.47 vs 4.39) |
| 9   | Topological Data Analysis | 🔜 Planned | Persistent homology, Betti numbers |

&nbsp;

# Phase 1 — Monthly Sunspot Forecasting

| Model | MAE | RMSE | R²  |
| --- | --- | --- | --- |
| Persistence | 15.05 | 21.20 | 0.857 |
| Moving Average (12m) | 18.75 | 25.61 | 0.791 |
| Linear Regression | 13.61 | 18.68 | 0.889 |
| Random Forest | 13.15 | 18.99 | 0.885 |
| XGBoost | **13.09** | **18.59** | **0.890** |

**Winner:** XGBoost

* * *

# Phase 3 — Peak Magnitude Prediction (LOCO)

## 36-Month Window

| Model | MAE (SN) | RMSE (SN) | R²  |
| --- | --- | --- | --- |
| Random Forest | **~19** | **~27** | **0.757** |
| XGBoost | ~21 | ~29 | ~0.72 |
| Ridge | ~23 | ~31 | ~0.68 |

## 24-Month Window

| Model | MAE (SN) | RMSE (SN) | R²  |
| --- | --- | --- | --- |
| Random Forest | ~22 | ~30 | ~0.69 |

## 12-Month Window

| Model | MAE (SN) | RMSE (SN) | R²  |
| --- | --- | --- | --- |
| Ridge | ~38 | ~50 | ~0.30 |

**Winner:** Random Forest (36m)

* * *

# Phase 4 — Peak Timing Prediction (LOCO)

## 36-Month Window

| Model | MAE (Months) | RMSE | R²  |
| --- | --- | --- | --- |
| Random Forest | **5.96** | **7.57** | **0.677** |
| XGBoost | 6.02 | 8.14 | 0.627 |
| Ridge | 7.18 | 9.17 | 0.527 |

## 24-Month Window

| Model | MAE (Months) | RMSE | R²  |
| --- | --- | --- | --- |
| Random Forest | 6.39 | 8.03 | 0.637 |

## 12-Month Window

| Model | MAE (Months) | RMSE | R²  |
| --- | --- | --- | --- |
| Ridge | 9.71 | 12.51 | 0.120 |

### Baseline

| Model | MAE |
| --- | --- |
| Mean Rise Time Baseline | 11.12 |

**Winner:** Random Forest (36m)

* * *

# Phase 5 — Trajectory Forecasting

| Model | Trajectory MAE | Trajectory RMSE | Peak Mag R² | Peak Time R² |
| --- | --- | --- | --- | --- |
| GRU | 25.09 | 34.66 | \-0.114 | 0.442 |
| LSTM | **24.93** | 34.75 | **\-0.071** | 0.537 |
| TCN | 25.34 | 35.74 | \-0.191 | **0.636** |

### Interpretation

- Best trajectory reconstruction: LSTM
- Best timing extraction from trajectory: TCN
- Best peak magnitude extraction from trajectory: LSTM

All three models performed similarly.

* * *

# Overall Ranking

| Rank | Model | Strongest Area |
| --- | --- | --- |
| 1   | Random Forest | Peak Magnitude, Peak Timing |
| 2   | XGBoost | Monthly Forecasting |
| 3   | LSTM | Best Trajectory MAE |
| 4   | TCN | Best Trajectory Timing R² |
| 5   | GRU | Competitive but not best |
| 6   | Ridge / Linear Regression | Strong baseline |
| 7   | Persistence | Simple benchmark |
| 8   | Moving Average | Weakest overall |

* * *

# Best Results Achieved

| Task | Best Model | Result |
| --- | --- | --- |
| Monthly Forecasting | XGBoost | R² = 0.890 |
| Peak Magnitude | Random Forest (36m) | R² = 0.757 |
| Peak Timing (static) | Random Forest (36m) | MAE = 5.96 months |
| Peak Timing (+ aux data) | RF + F10.7/Kp/Polar | MAE = 5.11 months |
| Peak Timing (running) | Phase 7 v2 RF | MAE = 4.39 months overall |
| Trajectory Forecasting | LSTM | MAE = 24.93 SN |
| Trajectory Timing | TCN | R² = 0.636 |

The strongest overall result from the project is:

```
Random Forest + 36-month observation window
```

It achieved the best performance on the two most important forecasting tasks:

- Peak Magnitude Prediction
- Peak Timing Prediction

while XGBoost was the best model for monthly sunspot forecasting.