"""
Sri Lanka Daily Rainfall Prediction - Streamlit Dashboard

AI Usage Declaration:
The Python code and Streamlit dashboard UI in this file were generated with AI
assistance (Antigravity / Claude) as permitted by the module guidelines.
Dataset validation, feature selection, model comparison, and results
interpretation were reviewed and verified by me (Dulshan).

Data & Libraries Credit:
- Dataset: Sri Lanka Weather Dataset by rasulmah on Kaggle
  (https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset),
  collected via the Open-Meteo Historical Weather API (2010-2023).
- Libraries: scikit-learn, XGBoost, LightGBM, CatBoost, pandas, Plotly, Streamlit.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

MODEL_PRED_COLS = {
    "Stacking Ensemble": "pred_stacking",
    "LightGBM": "pred_lgbm",
    "XGBoost": "pred_xgb",
    "CatBoost": "pred_catboost",
    "Random Forest": "pred_rf",
    "Naive Baseline (Lag-1)": "pred_naive",
}

CATEGORY_COLORS = {
    "Stacking": "#7c3aed",
    "Boosting": "#0d9488",
    "Bagging": "#2563eb",
    "Baseline": "#94a3b8",
}

FEATURE_LABELS = {
    "month": "Month (1-12)",
    "day_of_year": "Day of Year (1-366)",
    "city_encoded": "City Code",
    "latitude": "Latitude (°N)",
    "longitude": "Longitude (°E)",
    "elevation": "Elevation (m)",
    "temperature_2m_max": "Max Temperature (°C)",
    "temperature_2m_min": "Min Temperature (°C)",
    "temperature_2m_mean": "Mean Temperature (°C)",
    "apparent_temperature_max": "Max Apparent Temp (°C)",
    "apparent_temperature_min": "Min Apparent Temp (°C)",
    "apparent_temperature_mean": "Mean Apparent Temp (°C)",
    "shortwave_radiation_sum": "Shortwave Radiation (MJ/m²)",
    "windspeed_10m_max": "Max Wind Speed (km/h)",
    "windgusts_10m_max": "Max Wind Gusts (km/h)",
    "winddirection_10m_dominant": "Dominant Wind Direction (°)",
    "et0_fao_evapotranspiration": "FAO Evapotranspiration (mm)",
    "daylight_duration": "Daylight Duration (hrs)",
    "rain_sum_lag1": "Yesterday's Rainfall (mm)",
    "temp_mean_lag1": "Yesterday's Mean Temp (°C)",
}


@st.cache_resource
def load_artifacts():
    """Load all pre-trained models, metadata, and processed dataset from models/."""
    df = pd.read_csv(os.path.join(MODELS_DIR, "processed_weather.csv"), parse_dates=["time"])
    results_df = joblib.load(os.path.join(MODELS_DIR, "results_df.joblib"))
    fi_df = joblib.load(os.path.join(MODELS_DIR, "feature_importances.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    city_mapping = joblib.load(os.path.join(MODELS_DIR, "city_mapping.joblib"))
    city_metadata = joblib.load(os.path.join(MODELS_DIR, "city_metadata.joblib"))
    best_params = joblib.load(os.path.join(MODELS_DIR, "best_params.joblib"))

    models = {
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib")),
        "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib")),
        "LightGBM": joblib.load(os.path.join(MODELS_DIR, "lgbm_model.joblib")),
        "CatBoost": joblib.load(os.path.join(MODELS_DIR, "catboost_model.joblib")),
        "Stacking Ensemble": joblib.load(os.path.join(MODELS_DIR, "stacking_model.joblib")),
    }

    return {
        "df": df,
        "results_df": results_df,
        "fi_df": fi_df,
        "feature_cols": feature_cols,
        "city_mapping": city_mapping,
        "city_metadata": city_metadata,
        "best_params": best_params,
        "models": models,
    }


def render_data_overview(artifacts):
    """Render Page 1: Dataset summary, seasonality, city comparison, and target transform."""
    df = artifacts["df"]
    feature_cols = artifacts["feature_cols"]

    st.title("1. Data Overview - Sri Lanka Daily Rainfall (2010-2023)")
    st.markdown(
        "Daily weather records from the Open-Meteo Sri Lanka Weather Dataset across **5 distinct climatic zones**: "
        "**Colombo** (west coast wet), **Ratnapura** (inland wet), **Kandy** (central highlands), "
        "**Galle** (southwest coast), and **Jaffna** (northern dry zone)."
    )

    test_count = int((df["split"] == "test").sum())
    train_count = int((df["split"] == "train").sum())
    missing_count = int(df[feature_cols + ["rain_sum"]].isna().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned Daily Rows", f"{len(df):,}", f"{df['city'].nunique()} cities")
    c2.metric("Engineered Features", f"{len(feature_cols)}", f"{missing_count} missing values")
    c3.metric("Chronological Train Split", f"{train_count:,}", "2010-01-02 to 2021-06-17")
    c4.metric("Chronological Test Split", f"{test_count:,}", "2021-06-18 to 2023-06-17")

    st.divider()

    # Rainfall by Month (Overall + By City)
    col_left, col_right = st.columns(2)

    monthly_all = (
        df.groupby("month", as_index=False)["rain_sum"]
        .mean()
        .assign(Month=lambda d: d["month"].map(lambda m: MONTH_NAMES[m - 1]))
    )
    fig_month = px.bar(
        monthly_all,
        x="Month",
        y="rain_sum",
        title="Average Daily Rainfall by Month (All 5 Cities)",
        labels={"rain_sum": "Mean Rainfall (mm/day)", "Month": "Month"},
        color="rain_sum",
        color_continuous_scale="Blues",
    )
    fig_month.add_vrect(
        x0=3.5, x1=5.5, fillcolor="rgba(37, 99, 235, 0.08)", line_width=0,
        annotation_text="SW Monsoon Peak", annotation_position="top left",
    )
    fig_month.add_vrect(
        x0=8.5, x1=11.5, fillcolor="rgba(13, 148, 136, 0.08)", line_width=0,
        annotation_text="NE / Inter-Monsoon", annotation_position="top left",
    )
    fig_month.update_layout(coloraxis_showscale=False, height=400)
    col_left.plotly_chart(fig_month, use_container_width=True)

    monthly_city = (
        df.groupby(["month", "city"], as_index=False)["rain_sum"]
        .mean()
        .assign(Month=lambda d: d["month"].map(lambda m: MONTH_NAMES[m - 1]))
    )
    fig_month_city = px.line(
        monthly_city,
        x="Month",
        y="rain_sum",
        color="city",
        markers=True,
        title="Monthly Monsoon Seasonality by City",
        labels={"rain_sum": "Mean Rainfall (mm/day)", "city": "City"},
    )
    fig_month_city.update_layout(height=400)
    col_right.plotly_chart(fig_month_city, use_container_width=True)

    # Rainfall by City + Raw vs log1p Distribution
    col_a, col_b = st.columns(2)

    city_avg = (
        df.groupby("city", as_index=False)["rain_sum"]
        .mean()
        .sort_values("rain_sum", ascending=True)
    )
    fig_city = px.bar(
        city_avg,
        x="rain_sum",
        y="city",
        orientation="h",
        text=city_avg["rain_sum"].map(lambda v: f"{v:.2f} mm"),
        title="Average Daily Rainfall by City (2010-2023)",
        labels={"rain_sum": "Mean Daily Rainfall (mm)", "city": "City"},
        color="rain_sum",
        color_continuous_scale="Teal",
    )
    fig_city.update_traces(textposition="outside")
    fig_city.update_layout(
        coloraxis_showscale=False,
        xaxis_range=[0, float(city_avg["rain_sum"].max()) * 1.2],
        height=390,
    )
    col_a.plotly_chart(fig_city, use_container_width=True)

    dist_view = col_b.radio(
        "Target Distribution View",
        ["Raw rain_sum (mm)", "Log-Transformed log1p(rain_sum)"],
        horizontal=True,
    )
    if dist_view.startswith("Raw"):
        fig_dist = px.histogram(
            df,
            x="rain_sum",
            nbins=60,
            title=f"Raw Rainfall Distribution (Zero-rain days: {(df['rain_sum'] == 0).mean() * 100:.1f}%)",
            labels={"rain_sum": "Daily Rainfall (mm)"},
            color_discrete_sequence=["#2563eb"],
        )
    else:
        log_df = pd.DataFrame({"log1p_rain_sum": np.log1p(df["rain_sum"])})
        fig_dist = px.histogram(
            log_df,
            x="log1p_rain_sum",
            nbins=50,
            title="Log-Transformed Target Distribution: log1p(rain_sum)",
            labels={"log1p_rain_sum": "log(1 + rain_sum)"},
            color_discrete_sequence=["#0d9488"],
        )
    fig_dist.update_layout(height=345, yaxis_title="Number of Days")
    col_b.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.caption(
        "Data Credit: Sri Lanka Weather Dataset by rasulmah on Kaggle (Open-Meteo Historical Weather API, 2010-2023). "
        "Code and dashboard frontend developed with AI assistance (Antigravity / Claude); "
        "feature selection, model comparison, and interpretation verified by me (Dulshan)."
    )


def render_model_comparison(artifacts):
    """Render Page 2: Metrics comparison, test-period overlay chart, and ensemble analysis."""
    results_df = artifacts["results_df"]
    df = artifacts["df"]
    test_df = df[df["split"] == "test"].copy()

    st.title("2. Model Comparison - Bagging vs. Boosting vs. Stacking")

    winner = results_df.iloc[0]
    naive = results_df[results_df["Model"].str.contains("Naive")].iloc[0]
    rmse_gain = ((naive["RMSE (mm)"] - winner["RMSE (mm)"]) / naive["RMSE (mm)"]) * 100.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top Performing Model", winner["Model"], winner["Category"])
    m2.metric("Best Test RMSE", f"{winner['RMSE (mm)']:.4f} mm", f"-{rmse_gain:.1f}% vs Naive")
    m3.metric("Best Test MAE", f"{winner['MAE (mm)']:.4f} mm")
    m4.metric("Best Test R²", f"{winner['R²']:.4f}")

    st.subheader("Test Set Metrics (2021-06-18 to 2023-06-17, Original mm Scale)")
    st.dataframe(
        results_df.style.format({"RMSE (mm)": "{:.4f}", "MAE (mm)": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Bar charts for RMSE, MAE, R2
    c1, c2, c3 = st.columns(3)
    for col, metric in zip([c1, c2, c3], ["RMSE (mm)", "MAE (mm)", "R²"]):
        sorted_df = results_df.sort_values(metric, ascending=(metric != "R²"))
        fig_m = px.bar(
            sorted_df,
            x=metric,
            y="Model",
            color="Category",
            orientation="h",
            text=sorted_df[metric].map(lambda v: f"{v:.3f}"),
            color_discrete_map=CATEGORY_COLORS,
            title=metric,
        )
        fig_m.update_traces(textposition="outside")
        fig_m.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": list(reversed(sorted_df["Model"]))},
            showlegend=(metric == "R²"),
            height=360,
        )
        col.plotly_chart(fig_m, use_container_width=True)

    st.divider()
    st.subheader("Actual vs. Predicted Rainfall Over the 2-Year Test Period")

    f1, f2, f3 = st.columns([1, 1, 2])
    selected_city = f1.selectbox("City", ["All Cities (Average)", "Colombo", "Kandy", "Ratnapura", "Galle", "Jaffna"])
    agg_mode = f2.selectbox("Aggregation", ["Monthly Average", "Weekly Average", "Daily Window"])
    selected_models = f3.multiselect(
        "Models to Overlay",
        list(MODEL_PRED_COLS.keys()),
        default=["Stacking Ensemble", "LightGBM", "Random Forest"],
    )

    sub = test_df.copy()
    if selected_city != "All Cities (Average)":
        sub = sub[sub["city"] == selected_city]

    if agg_mode == "Daily Window":
        # Show the last 90 days by default or allow user selection
        min_d = sub["time"].min().date()
        max_d = sub["time"].max().date()
        date_range = st.slider(
            "Select Daily Date Window",
            min_value=min_d,
            max_value=max_d,
            value=(pd.Timestamp("2022-10-01").date(), pd.Timestamp("2022-12-31").date()),
        )
        sub = sub[(sub["time"].dt.date >= date_range[0]) & (sub["time"].dt.date <= date_range[1])]
        grouped = sub.groupby("time", as_index=False)[["rain_sum"] + list(MODEL_PRED_COLS.values())].mean()
    elif agg_mode == "Weekly Average":
        grouped = (
            sub.set_index("time")[["rain_sum"] + list(MODEL_PRED_COLS.values())]
            .resample("W")
            .mean()
            .reset_index()
        )
    else:
        grouped = (
            sub.set_index("time")[["rain_sum"] + list(MODEL_PRED_COLS.values())]
            .resample("ME")
            .mean()
            .reset_index()
        )

    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=grouped["time"],
            y=grouped["rain_sum"],
            mode="lines+markers",
            name="Actual Rainfall",
            line=dict(color="#0f172a", width=2.8),
        )
    )
    model_line_colors = {
        "Stacking Ensemble": "#7c3aed",
        "LightGBM": "#0d9488",
        "XGBoost": "#0284c7",
        "CatBoost": "#f59e0b",
        "Random Forest": "#2563eb",
        "Naive Baseline (Lag-1)": "#94a3b8",
    }
    for m_name in selected_models:
        col_key = MODEL_PRED_COLS[m_name]
        fig_ts.add_trace(
            go.Scatter(
                x=grouped["time"],
                y=grouped[col_key],
                mode="lines",
                name=m_name,
                line=dict(color=model_line_colors.get(m_name, "#6366f1"), width=2, dash="dash" if "Naive" in m_name else "solid"),
            )
        )
    fig_ts.update_layout(
        title=f"Actual vs. Predicted Rainfall ({selected_city} - {agg_mode})",
        xaxis_title="Date",
        yaxis_title="Mean Rainfall (mm/day)",
        hovermode="x unified",
        height=430,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.info(
        f"**Why {winner['Model']} Won (Bagging vs. Boosting vs. Stacking):**\n\n"
        "- **Bagging (`Random Forest`)** averages deep parallel trees trained on bootstrap samples, cutting prediction variance on noisy daily weather.\n"
        "- **Boosting (`XGBoost`, `LightGBM`, `CatBoost`)** fits trees sequentially on prior residuals, reducing bias and capturing sharp non-linear monsoon thresholds.\n"
        "- **Stacking (`StackingRegressor` + `RidgeCV`)** blends Bagging and Boosting out-of-fold predictions, combining variance reduction with low-bias residual fitting.\n\n"
        "**Tabular vs. Image/Text Data Strengths & Limitations:**\n"
        "Tree ensembles dominate structured tabular meteorology with mixed units, zero-inflation, and threshold interactions without heavy feature scaling, "
        "whereas deep neural networks excel on unstructured spatial/sequential signals (satellite imagery, radar, or text). "
        "Main limitation: squared-error training on `log1p(rain_sum)` dampens rare >100 mm extreme monsoon cloudbursts."
    )


def render_feature_importance(artifacts):
    """Render Page 3: Bagging vs. Boosting feature importance comparison."""
    fi_df = artifacts["fi_df"].copy()
    fi_df["Feature Label"] = fi_df["feature"].map(lambda c: FEATURE_LABELS.get(c, c))

    st.title("3. Feature Importance - Meteorological Drivers")
    st.markdown(
        "Compare relative feature importances (normalized to sum to 100%) between **Random Forest (Bagging)** "
        "and your choice of **Boosting** model."
    )

    boost_choice = st.selectbox("Select Boosting Model to Compare", ["LightGBM", "XGBoost", "CatBoost"])

    col1, col2 = st.columns(2)
    for col, model_name, color in [
        (col1, "Random Forest", "#2563eb"),
        (col2, boost_choice, "#0d9488"),
    ]:
        top12 = fi_df.sort_values(model_name, ascending=True).tail(12)
        fig_fi = px.bar(
            top12,
            x=model_name,
            y="Feature Label",
            orientation="h",
            text=top12[model_name].map(lambda v: f"{v:.1f}%"),
            title=f"Top 12 Features - {model_name}",
            labels={model_name: "Relative Importance (%)", "Feature Label": ""},
            color_discrete_sequence=[color],
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(xaxis_range=[0, float(top12[model_name].max()) * 1.2], height=440)
        col.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Full Feature Importance Table (%)")
    display_fi = fi_df[["feature", "Feature Label", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]].sort_values(
        boost_choice, ascending=False
    )
    st.dataframe(
        display_fi.style.format({
            "Random Forest": "{:.2f}%",
            "XGBoost": "{:.2f}%",
            "LightGBM": "{:.2f}%",
            "CatBoost": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**Meteorological Interpretation:** Rainy monsoon days in Sri Lanka feature dense cloud cover that sharply lowers "
        "`shortwave_radiation_sum` and `et0_fao_evapotranspiration`, narrows the diurnal temperature range (`temperature_2m_max` vs. `min`), "
        "elevates `windgusts_10m_max`, and reflects multi-day wet-spell persistence via `rain_sum_lag1`."
    )


def _apply_preset(city, month, day_of_year, city_metadata, overrides=None, set_selectors=True):
    meta = city_metadata[city]
    meds = meta["monthly_medians"][month]
    if set_selectors:
        st.session_state["lp_city"] = city
        st.session_state["lp_month"] = month
        st.session_state["lp_doy"] = day_of_year
    for feat, val in meds.items():
        st.session_state[f"lp_{feat}"] = float(val)
    if overrides:
        for feat, val in overrides.items():
            st.session_state[f"lp_{feat}"] = float(val)
    st.session_state["lp_last_city"] = city
    st.session_state["lp_last_month"] = month


def render_live_prediction(artifacts):
    """Render Page 4: Interactive weather inputs & live rainfall predictions across all models."""
    city_metadata = artifacts["city_metadata"]
    feature_cols = artifacts["feature_cols"]
    models = artifacts["models"]

    st.title("4. Live Rainfall Prediction")
    st.markdown(
        "Configure city, season, and daily meteorological indicators, or load a preset scenario, to predict "
        "daily rainfall (**mm**) across every trained ensemble model."
    )

    cities = list(city_metadata.keys())

    # Initialize session defaults cleanly
    if "lp_city" not in st.session_state:
        _apply_preset("Colombo", 5, 135, city_metadata)

    # Preset buttons
    p1, p2, p3 = st.columns([1.2, 1.2, 1.6])
    if p1.button("🌧️ Wet Monsoon Day (Ratnapura)", use_container_width=True):
        _apply_preset("Ratnapura", 5, 140, city_metadata)
    if p2.button("☀️ Dry Season Day (Jaffna)", use_container_width=True):
        _apply_preset("Jaffna", 2, 45, city_metadata)
    if p3.button("🔄 Reset to City-Month Median", use_container_width=True):
        cur_c = st.session_state.get("lp_city", "Colombo")
        cur_m = int(st.session_state.get("lp_month", 5))
        _apply_preset(cur_c, cur_m, min(365, (cur_m - 1) * 30 + 15), city_metadata)

    # Primary Location & Calendar Selectors
    c1, c2, c3 = st.columns(3)
    selected_city = c1.selectbox("City", cities, key="lp_city")
    selected_month = c2.selectbox(
        "Month",
        list(range(1, 13)),
        format_func=lambda m: f"{m:02d} - {MONTH_NAMES[m - 1]}",
        key="lp_month",
    )
    selected_doy = c3.slider("Day of Year", min_value=1, max_value=366, key="lp_doy")

    # Auto-update weather defaults if user changed city or month manually
    if (
        st.session_state.get("lp_last_city") != selected_city
        or st.session_state.get("lp_last_month") != selected_month
    ):
        _apply_preset(selected_city, selected_month, selected_doy, city_metadata, set_selectors=False)

    meta = city_metadata[selected_city]
    bounds = meta["feature_bounds"]
    slider_feats = [
        f for f in feature_cols
        if f not in ("month", "day_of_year", "city_encoded", "latitude", "longitude", "elevation")
    ]

    st.subheader(f"Weather Inputs ({selected_city} - {MONTH_NAMES[selected_month - 1]})")
    st.caption(
        f"Static geography: Lat {meta['latitude']:.2f}°N, Lon {meta['longitude']:.2f}°E, "
        f"Elevation {meta['elevation']:.0f} m (encoded={meta['city_encoded']})"
    )

    input_values = {
        "month": selected_month,
        "day_of_year": selected_doy,
        "city_encoded": meta["city_encoded"],
        "latitude": meta["latitude"],
        "longitude": meta["longitude"],
        "elevation": meta["elevation"],
    }

    cols = st.columns(3)
    for idx, feat in enumerate(slider_feats):
        col = cols[idx % 3]
        f_min = float(bounds[feat]["min"])
        f_max = float(bounds[feat]["max"])
        cur_val = float(st.session_state.get(f"lp_{feat}", meta["monthly_medians"][selected_month][feat]))
        cur_val = float(np.clip(cur_val, f_min, f_max))
        st.session_state[f"lp_{feat}"] = cur_val
        step = 1.0 if feat in ("winddirection_10m_dominant",) else 0.1
        input_values[feat] = col.slider(
            FEATURE_LABELS.get(feat, feat),
            min_value=f_min,
            max_value=f_max,
            step=step,
            key=f"lp_{feat}",
        )

    X_input = pd.DataFrame([[input_values[c] for c in feature_cols]], columns=feature_cols)

    # Predict in log1p space -> invert via expm1 clipped at 0
    pred_rows = []
    model_order = [
        ("Stacking Ensemble", "Stacking"),
        ("LightGBM", "Boosting"),
        ("XGBoost", "Boosting"),
        ("CatBoost", "Boosting"),
        ("Random Forest", "Bagging"),
    ]
    for m_name, cat in model_order:
        pred_log = float(models[m_name].predict(X_input)[0])
        pred_mm = float(np.clip(np.expm1(pred_log), 0.0, None))
        pred_rows.append({"Model": m_name, "Category": cat, "Predicted Rainfall (mm)": round(pred_mm, 2)})

    # Naive baseline prediction equals yesterday's rainfall
    pred_rows.append({
        "Model": "Naive Baseline (Lag-1)",
        "Category": "Baseline",
        "Predicted Rainfall (mm)": round(float(input_values["rain_sum_lag1"]), 2),
    })

    pred_df = pd.DataFrame(pred_rows)

    st.divider()
    st.subheader("Predicted Daily Rainfall (mm)")

    metric_cols = st.columns(len(pred_rows))
    for mcol, row in zip(metric_cols, pred_rows):
        mcol.metric(row["Model"], f"{row['Predicted Rainfall (mm)']:.2f} mm", row["Category"])

    left_out, right_out = st.columns([1.4, 1.0])
    fig_pred = px.bar(
        pred_df,
        x="Model",
        y="Predicted Rainfall (mm)",
        color="Category",
        text=pred_df["Predicted Rainfall (mm)"].map(lambda v: f"{v:.2f} mm"),
        color_discrete_map=CATEGORY_COLORS,
        title="Model Predictions Comparison",
    )
    fig_pred.update_traces(textposition="outside")
    fig_pred.update_layout(
        yaxis_range=[0, max(5.0, float(pred_df["Predicted Rainfall (mm)"].max()) * 1.25)],
        height=370,
    )
    left_out.plotly_chart(fig_pred, use_container_width=True)
    right_out.dataframe(
        pred_df.style.format({"Predicted Rainfall (mm)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


def main():
    st.set_page_config(
        page_title="Sri Lanka Rainfall Ensemble Dashboard",
        page_icon="🌧️",
        layout="wide",
    )
    artifacts = load_artifacts()

    st.sidebar.title("🌧️ Sri Lanka Rainfall")
    st.sidebar.caption("Ensemble Learning Comparison (2010-2023)")
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Data Overview",
            "2. Model Comparison",
            "3. Feature Importance",
            "4. Live Prediction",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Models Loaded:**\n"
        "- Random Forest (Bagging)\n"
        "- XGBoost, LightGBM, CatBoost (Boosting)\n"
        "- StackingRegressor (`RidgeCV`)\n"
        "- Naive Lag-1 Baseline"
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Credits & Declaration**")
    st.sidebar.info(
        "**AI Declaration:** Code and dashboard UI developed with AI assistance (Antigravity / Claude). "
        "Model selection, comparison, and interpretation by me (Dulshan).\n\n"
        "**Data Credit:** Kaggle (`rasulmah/sri-lanka-weather-dataset`) & Open-Meteo Historical Weather API "
        "(Sri Lanka, 2010-2023)."
    )

    if page == "1. Data Overview":
        render_data_overview(artifacts)
    elif page == "2. Model Comparison":
        render_model_comparison(artifacts)
    elif page == "3. Feature Importance":
        render_feature_importance(artifacts)
    else:
        render_live_prediction(artifacts)


if __name__ == "__main__":
    main()
