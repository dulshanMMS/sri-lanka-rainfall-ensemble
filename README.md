# Sri Lanka Daily Rainfall Prediction - Ensemble Learning Dashboard

**AI tools used:** This code was generated with AI assistance (Antigravity / Claude). I reviewed it and can explain it.

**Data source:** [Sri Lanka Weather Dataset on Kaggle (`rasulmah/sri-lanka-weather-dataset`)](https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset) - Daily historical weather records (Open-Meteo API) across 30 Sri Lankan cities from `2010-01-01` to `2023-06-17`.

---

## Project Overview

This project builds and evaluates an **Ensemble Learning Pipeline and Interactive Streamlit Dashboard** for predicting daily rainfall (`rain_sum`, in mm) across five representative Sri Lankan cities spanning all major climatic zones:
- **Colombo** (Western Coastal Wet Zone)
- **Kandy** (Central Highlands / Hill Country)
- **Ratnapura** (Sabaragamuwa Inland Wet Zone - peak monsoon rainfall)
- **Galle** (Southwest Coast)
- **Jaffna** (Northern Dry Zone)

All models are trained on `log1p(rain_sum)` using **20 meteorological, geographic, seasonal, and per-city lag-1 features** (after strictly dropping leaky variables `precipitation_sum`, `precipitation_hours`, `weathercode`, `snowfall_sum`, `sunrise`, `sunset`), tuned with `RandomizedSearchCV` using `TimeSeriesSplit`, and evaluated on the original **millimeter (mm)** scale on a strict **2-year chronological test split** (`2021-06-18` to `2023-06-17`, `3,650` rows, no shuffling).

---

## Model Comparison Results (Test Set: 2021-06-18 to 2023-06-17)

| Rank | Model | Ensemble Paradigm | RMSE (mm) | MAE (mm) | R² |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | **Stacking Ensemble (`RF + XGB + LGBM + CatBoost -> RidgeCV`)** | **Stacking / Hybridization** | **6.0326** | **2.7983** | **0.5481** |
| 2 | LightGBM (`LGBMRegressor`) | Boosting | 6.0535 | 2.8200 | 0.5450 |
| 3 | XGBoost (`XGBRegressor`) | Boosting | 6.0731 | 2.8186 | 0.5420 |
| 4 | CatBoost (`CatBoostRegressor`) | Boosting | 6.0980 | 2.8238 | 0.5382 |
| 5 | Random Forest (`RandomForestRegressor`) | Bagging | 6.4423 | 2.9975 | 0.4846 |
| 6 | Naive Baseline (Yesterday's Rain `rain_sum_lag1`) | Baseline | 9.8039 | 4.8907 | -0.1935 |

---

## Repository Structure

```text
.
├── SriLanka_Weather_Dataset.csv   # Raw Open-Meteo Sri Lankan daily weather dataset (147,480 rows)
├── notebook.ipynb                 # Executed Jupyter Notebook (EDA, Feature Engineering, Bagging/Boosting/Stacking, Analysis)
├── app.py                         # 4-Page Streamlit Dashboard (loads pre-trained artifacts from models/ only)
├── models/                        # Saved joblib models, feature lists, metadata, and processed dataset
│   ├── rf_model.joblib
│   ├── xgb_model.joblib
│   ├── lgbm_model.joblib
│   ├── catboost_model.joblib
│   ├── stacking_model.joblib
│   ├── feature_cols.joblib
│   ├── city_mapping.joblib
│   ├── city_metadata.joblib
│   ├── feature_importances.joblib
│   ├── results_df.joblib
│   ├── results_table.csv
│   └── processed_weather.csv
├── requirements.txt               # Pinned package versions
└── README.md                      # Project documentation & run instructions
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Open / Run the Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## Credits, Data Attribution & AI Usage Declaration

- **AI Usage Declaration:** Code implementation and the Streamlit dashboard frontend were developed with AI assistance (Antigravity / Claude) as permitted by the module guidelines. Dataset validation, feature selection, model comparison, and results interpretation were performed and verified by me (Dulshan).
- **Dataset Credit:** [Sri Lanka Weather Dataset](https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset) by `rasulmah` on Kaggle, collected via the [Open-Meteo Historical Weather API](https://open-meteo.com/) for 30 Sri Lankan cities (2010-2023).
- **Libraries & Frameworks:** `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `pandas`, `matplotlib`, `seaborn`, `Plotly`, and `Streamlit`.
