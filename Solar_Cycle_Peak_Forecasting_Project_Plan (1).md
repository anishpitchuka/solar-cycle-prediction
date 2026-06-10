# Solar Cycle Peak Forecasting Project

## PRIMARY OBJECTIVE

Build a machine learning system that predicts:
1. Future monthly sunspot numbers
2. Solar cycle peak magnitude
3. Solar cycle peak timing
4. Future cycle trajectory

---

# PHASE 0: DATA ACQUISITION AND EXPLORATION

## Step 0.1
Download SILSO Monthly Sunspot Number Dataset (Version 2.0)

Required columns:
- Date
- Sunspot_Number

Period:
- 1749–Present

## Step 0.2
Load dataset into Pandas DataFrame.

Requirements:
- Parse dates correctly
- Sort chronologically

## Step 0.3
Perform data quality checks:
- Missing values
- Duplicate rows
- Invalid dates
- Negative values

Generate report.

## Step 0.4
Create visualizations:
- Full history (1749–Present)
- Last 100 years
- Last 50 years

## Step 0.5
Identify cycle minima and maxima.

For each cycle determine:
- Cycle ID
- Start date
- Peak date
- End date
- Peak value

Store results.

---

# PHASE 1: MONTHLY SUNSPOT FORECASTING

## Goal
Predict:
- Sunspot_Number(t+1)

using historical observations.

## Step 1.1 Feature Engineering

Create lag features:
- lag1
- lag3
- lag6
- lag12
- lag24
- lag36

## Step 1.2 Rolling Features

Create:
- rolling_mean_3
- rolling_mean_6
- rolling_mean_12
- rolling_mean_24

## Step 1.3 Trend Features

Create:
- slope_3
- slope_6
- slope_12

Use linear regression over rolling windows.

## Step 1.4 Dataset Construction

Target:
- Sunspot_Number(t+1)

Features:
- All lag features
- All rolling features
- All trend features

## Step 1.5 Train/Test Split

Rules:
- No random shuffling

Example:
- 1749–2000 → Train
- 2001–2015 → Validation
- 2016–Present → Test

## Step 1.6 Baselines

Train:
- Persistence
- Moving Average
- Linear Regression

## Step 1.7 Tree Models

Train:
- Random Forest
- XGBoost

Tune hyperparameters.

## Step 1.8 Evaluation

Compute:
- MAE
- RMSE
- R²

Store results.

---

# PHASE 2: CYCLE DATABASE CREATION

Create one record per solar cycle.

Store:
- Cycle_ID
- Start_Date
- End_Date
- Peak_Date
- Peak_Sunspot_Number
- Cycle_Length
- Rise_Time
- Decay_Time

Output:
- Cycle_Database.csv

---

# PHASE 3: PEAK MAGNITUDE PREDICTION

Goal:
- Predict Peak_Sunspot_Number

Using early-cycle observations.

Feature windows:
- First 12 months
- First 24 months
- First 36 months

Features:
- Mean Sunspot Number
- Maximum Sunspot Number
- Growth Rate
- Lag Features
- Rolling Means
- Trend Features

Models:
- Linear Regression
- Random Forest
- XGBoost

Validation:
- Leave-One-Cycle-Out Cross Validation

Metrics:
- Peak Magnitude Error
- MAE
- RMSE

---

# PHASE 4: PEAK TIMING PREDICTION

Goal:
- Predict Peak_Date

Target:
- Months_From_Start_To_Peak

Models:
- Random Forest
- XGBoost

Validation:
- Leave-One-Cycle-Out

Metrics:
- Peak Timing Error
- MAE
- RMSE

---

# PHASE 5: TRAJECTORY FORECASTING

Goal:
- Predict remaining cycle trajectory

Input:
- Observed months up to time t

Output:
- Future monthly sunspot numbers

Models:
- GRU
- LSTM
- TCN

Evaluation:
- Trajectory RMSE
- Trajectory MAE
- R²

Peak Extraction:
- peak_value = max(prediction)
- peak_month = argmax(prediction)

Store:
- Predicted Peak Magnitude
- Predicted Peak Timing

---

# PHASE 6: ADDITIONAL DATASETS

Add one at a time:
1. F10.7
2. Polar Field
3. Solar Wind
4. IMF
5. Kp
6. Dst

For each:
- Merge dataset
- Retrain best model
- Compare results
- Keep only if performance improves

---

# PHASE 7: IMAGE FEATURES

Data Sources:
- SOHO
- SDO

Extract:
- Entropy
- Fractal Dimension
- Active Region Count
- Magnetic Flux

Aggregate monthly.

Integrate into best-performing model.

---

# PHASE 8: TOPOLOGICAL DATA ANALYSIS (TDA)

Input:
- Solar magnetograms

Compute:
- Persistent Homology
- Betti Numbers
- Persistence Lifetimes

Evaluate impact on:
- Peak Magnitude Prediction
- Peak Timing Prediction
- Trajectory Forecasting

---

# VALIDATION RULES

Always enforce:
- No random train/test split
- No future information leakage
- Time-aware validation only
- Leave-One-Cycle-Out for cycle prediction

---

# FINAL OUTPUTS

Generate:
- Monthly Forecasting Model
- Peak Magnitude Model
- Peak Timing Model
- Trajectory Forecasting Model
- Feature Importance Analysis
- SC25 Forecast
- SC26 Experimental Forecast

---

# CURRENT TASK

Execute only PHASE 0.

Do not proceed until:
- Dataset acquired
- Dataset cleaned
- EDA completed
- Cycle minima/maxima identified
