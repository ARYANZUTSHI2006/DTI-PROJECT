# AI-Driven Runoff Prediction for Flood Forecasting and Climate Resilience

## 1. Project Overview

This project focuses on developing an AI-based runoff prediction system using historical meteorological and hydrological data. The goal is to forecast daily river discharge and evaluate model robustness under varying climatic conditions to support flood forecasting and climate resilience analysis.

The study utilizes 31 years (1979–2009) of daily data from the Kasol basin.

---

## 2. Objectives

1. Develop machine learning and deep learning models for daily runoff prediction.
2. Compare traditional ML models with deep learning time-series models.
3. Evaluate model performance during extreme rainfall and peak discharge events.
4. Analyze long-term discharge trends for climate resilience assessment.

---

## 3. Dataset Description

The dataset is stored in `Kasol.xlsx` and contains multiple sheets:

### Sheet1 – Daily Hydrometeorological Data (Main Dataset)

**Time Range:** 1979-01-01 to 2009-12-31
**Total Records:** 11,323 daily observations

**Columns:**

* `DATE` – Daily timestamp
* `Discharge (CUMEC)` – Daily river discharge (target variable)
* `PCP` – Precipitation
* `TMAX` – Maximum temperature
* `TMIN` – Minimum temperature
* `rh` – Relative humidity
* `solar` – Solar radiation
* `wind` – Wind speed
* `P1`, `P2`, `P3` – Rainfall at multiple stations/sub-basins

---

### Sheet2 – Yearly Average Discharge

Contains:

* Year
* Average annual discharge

Used for:

* Long-term trend analysis
* Climate variability assessment
* Wet vs dry year classification

---

## 4. Methodology

### 4.1 Data Preprocessing

* Convert DATE to datetime format
* Handle minimal missing values (interpolation or forward fill)
* Sort data chronologically
* Verify time continuity (confirmed: no missing dates)

### 4.2 Feature Engineering

Hydrology-based enhancements:

* Lag features (t-1, t-2, t-3 rainfall)
* Rolling rainfall accumulation (3-day, 7-day)
* Antecedent Precipitation Index (API)
* Mean temperature = (TMAX + TMIN)/2
* Seasonal cyclic encoding (sin/cos transformation)

---

## 5. Modeling Approach

### 5.1 Baseline Model

* Linear Regression

### 5.2 Machine Learning Models

* Random Forest Regressor
* XGBoost Regressor

### 5.3 Deep Learning Model

* LSTM (Long Short-Term Memory)

  * Input: Previous 7–14 days meteorological data
  * Output: Next-day discharge

---

## 6. Forecasting Strategy

Primary task:

* 1-day ahead discharge forecasting

Optional extension:

* Multi-day forecasting (3-day ahead)
* Flood threshold classification

---

## 7. Evaluation Metrics

For regression performance:

* RMSE (Root Mean Square Error)
* MAE (Mean Absolute Error)
* R² (Coefficient of Determination)
* NSE (Nash–Sutcliffe Efficiency)

Extreme event evaluation:

* Performance on top 10% discharge days
* Peak flow error percentage

---

## 8. Climate Resilience Analysis

Using yearly average discharge (Sheet2):

* Long-term discharge trend analysis
* Comparison of early vs recent decades
* Identification of extreme years
* Sensitivity testing under increased rainfall scenarios (+10%, +20%)

---

## 9. Project Workflow

Data Cleaning
→ Feature Engineering
→ Time-Based Train-Test Split
→ Model Training (RF, XGBoost, LSTM)
→ Performance Evaluation
→ Extreme Event Analysis
→ Climate Trend Assessment

---

## 10. Tools and Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* TensorFlow / Keras (for LSTM)
* Matplotlib / Seaborn

---

## 11. Expected Outcomes

* Accurate daily runoff forecasting model
* Comparative performance analysis of ML vs DL
* Insights into extreme event prediction capability
* Assessment of long-term hydrological trends

---

## 12. Project Significance

This project integrates hydrology and artificial intelligence to improve flood forecasting and climate resilience planning. It demonstrates the application of data-driven methods in environmental risk management and sustainable water resource planning.

---
