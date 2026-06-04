import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ── Page Configuration ──────────────────────────────────
st.set_page_config(
    page_title="SmartCity Traffic Forecast",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Model, Features, and Historical Data ───────
@st.cache_data
def load_assets():
    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir / 'models' / 'lightgbm_traffic_forecaster.pkl')
    features = joblib.load(base_dir / 'models' / 'feature_columns.pkl')
    # Load original raw data to act as our "historical database"
    hist_df = pd.read_csv(base_dir / 'data' / 'raw' / 'train.csv')
    hist_df['DateTime'] = pd.to_datetime(hist_df['DateTime'])
    return model, features, hist_df

# Load assets
model, FEATURES, hist_df = load_assets()

# ── App Header ──────────────────────────────────────────
st.markdown("""
<style>
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .header-subtitle {
        font-size: 1.1em;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f0f8ff;
        border: 1px solid #1f77b4;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
<div class="header-title">🚦 Smart City Traffic Forecasting System</div>
<div class="header-subtitle">Intelligent 24-Hour Traffic Volume Prediction</div>
""", unsafe_allow_html=True)

# ── Sidebar Controls ────────────────────────────────────
st.sidebar.markdown("## Configuration")
st.sidebar.markdown("---")
junction = st.sidebar.selectbox("📍 Select Junction", [1, 2, 3, 4])
default_date = hist_df['DateTime'].max().date() + pd.Timedelta(days=1)
date = st.sidebar.date_input("📅 Select Forecast Date", value=default_date)

# ── Forecast Horizon Selection ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Forecast Duration")
forecast_option = st.sidebar.selectbox(
    "How far ahead to predict?",
    options=["24 Hours (1 Day)", "7 Days (1 Week)", "14 Days (2 Weeks)", "30 Days (1 Month)", "90 Days (3 Months)"],
    index=0
)

# Parse forecast option
forecast_mapping = {
    "24 Hours (1 Day)": 1,
    "7 Days (1 Week)": 7,
    "14 Days (2 Weeks)": 14,
    "30 Days (1 Month)": 30,
    "90 Days (3 Months)": 90
}
forecast_days = forecast_mapping[forecast_option]
forecast_hours = forecast_days * 24

# ── Information Panel ──────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Information")
st.sidebar.info(f"""
**Latest Data Date:** {hist_df['DateTime'].max().strftime('%B %d, %Y')}

**Total Records:** {len(hist_df):,}

**Junctions Covered:** 1, 2, 3, 4
""")

# ── Main Content Area ──────────────────────────────────
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Prediction Settings")
    st.write(f"**Selected Junction:** Junction {junction}")
    st.write(f"**Forecast Date:** {date.strftime('%A, %B %d, %Y')}")

with col2:
    st.markdown("### Quick Stats")
    j_data = hist_df[hist_df['Junction'] == junction]
    st.metric("Avg. Vehicles/Hour", f"{j_data['Vehicles'].mean():.0f}")
    st.metric("Peak Traffic", f"{j_data['Vehicles'].max():.0f}")

st.markdown("---")

# ── 3. Prediction Pipeline ─────────────────────────────
if st.button("🔮 Generate Forecast", use_container_width=True):
    
    with st.spinner(f"⏳ Calculating features and generating {forecast_days}-day forecast..."):
        # Step A: Generate the future hours
        future_hours = pd.date_range(str(date), periods=forecast_hours, freq='h')
        pred_df = pd.DataFrame({'DateTime': future_hours, 'Junction': junction})
        
        # Step B: Get historical data for context (at least 168 hours/7 days)
        # Use more data if making longer forecasts for better seasonal context
        context_hours = max(168, forecast_hours)
        j_hist = hist_df[hist_df['Junction'] == junction].tail(context_hours).copy()
        
        # Step C: Stitch them together
        combined_df = pd.concat([j_hist, pred_df], ignore_index=True)
        
        # Step D: Run the Feature Engineering Pipeline
        # Time Features
        combined_df['hour']         = combined_df['DateTime'].dt.hour
        combined_df['day_of_week']  = combined_df['DateTime'].dt.dayofweek
        combined_df['month']        = combined_df['DateTime'].dt.month
        combined_df['year']         = combined_df['DateTime'].dt.year
        combined_df['is_weekend']   = (combined_df['day_of_week'] >= 5).astype(int)
        combined_df['quarter']      = combined_df['DateTime'].dt.quarter
        combined_df['is_rush_hour'] = combined_df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        combined_df['is_night']     = (combined_df['hour'].between(22, 23) | combined_df['hour'].between(0, 5)).astype(int)
        
        # Cyclical Encoding
        combined_df['hour_sin']  = np.sin(2 * np.pi * combined_df['hour'] / 24)
        combined_df['hour_cos']  = np.cos(2 * np.pi * combined_df['hour'] / 24)
        combined_df['dow_sin']   = np.sin(2 * np.pi * combined_df['day_of_week'] / 7)
        combined_df['dow_cos']   = np.cos(2 * np.pi * combined_df['day_of_week'] / 7)
        combined_df['month_sin'] = np.sin(2 * np.pi * combined_df['month'] / 12)
        combined_df['month_cos'] = np.cos(2 * np.pi * combined_df['month'] / 12)
        
        # Lag Features
        combined_df['lag_1h']    = combined_df['Vehicles'].shift(1)
        combined_df['lag_2h']    = combined_df['Vehicles'].shift(2)
        combined_df['lag_24h']   = combined_df['Vehicles'].shift(24)
        combined_df['lag_168h']  = combined_df['Vehicles'].shift(168)
        combined_df['rolling_mean_3h']  = combined_df['Vehicles'].rolling(3).mean()
        combined_df['rolling_mean_24h'] = combined_df['Vehicles'].rolling(24).mean()
        combined_df['rolling_std_24h']  = combined_df['Vehicles'].rolling(24).std()
        combined_df['rolling_mean_7d']  = combined_df['Vehicles'].rolling(168).mean()
        
        # Step E: Extract ONLY the future hours we want to predict
        final_pred_df = combined_df[combined_df['Vehicles'].isna()].copy()
        
        # Step E1: Handle NaN values in lag features and rolling stats for future dates
        # Forward-fill lag features using the last available value
        for lag_col in ['lag_1h', 'lag_2h', 'lag_24h', 'lag_168h']:
            final_pred_df[lag_col] = final_pred_df[lag_col].ffill()  # Pandas 3.0+ syntax
            # If still NaN, use the mean of historical data
            if final_pred_df[lag_col].isna().any():
                final_pred_df[lag_col] = final_pred_df[lag_col].fillna(
                    combined_df[combined_df['Vehicles'].notna()][lag_col].mean()
                )
        
        # Forward-fill rolling statistics
        for roll_col in ['rolling_mean_3h', 'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_7d']:
            final_pred_df[roll_col] = final_pred_df[roll_col].ffill()  # Pandas 3.0+ syntax
            # If still NaN, use the mean of historical data
            if final_pred_df[roll_col].isna().any():
                final_pred_df[roll_col] = final_pred_df[roll_col].fillna(
                    combined_df[combined_df['Vehicles'].notna()][roll_col].mean()
                )
        
        # Step F: Predict!
        final_pred_df['prediction'] = model.predict(final_pred_df[FEATURES])
        
        st.success("✅ Forecast generated successfully!")
    
    # ── Results Display ────────────────────────────────
    st.markdown("---")
    forecast_days_text = "Day" if forecast_days == 1 else "Days"
    st.markdown(f"## 📈 {forecast_days}-{forecast_days_text} Forecast Results")
    
    # Step G: Plot the Results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=final_pred_df['DateTime'], 
        y=final_pred_df['prediction'],
        mode='lines+markers', 
        name='Predicted Traffic',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, color='#ff7f0e')
    ))
    
    fig.update_layout(
        title=f"Junction {junction} — {forecast_days}-Day Traffic Forecast starting {date.strftime('%B %d, %Y')}",
        xaxis_title="Time Period",
        yaxis_title="Predicted Vehicle Count",
        template="plotly",
        hovermode="x unified",
        height=500,
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f5f5f5"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Statistics Panel ────────────────────────────────
    st.markdown("### 📊 Forecast Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Vehicles", f"{final_pred_df['prediction'].mean():.0f}")
    with col2:
        st.metric("Peak Traffic", f"{final_pred_df['prediction'].max():.0f}")
    with col3:
        st.metric("Minimum Traffic", f"{final_pred_df['prediction'].min():.0f}")
    with col4:
        st.metric("Std Deviation", f"{final_pred_df['prediction'].std():.0f}")
    
    # ── Raw Forecast Data ──────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Detailed Forecast Data")
    
    display_df = final_pred_df[['DateTime', 'prediction']].copy()
    display_df.columns = ['Time', 'Predicted Vehicles']
    
    # Format time based on forecast length
    if forecast_days == 1:
        display_df['Time'] = display_df['Time'].dt.strftime('%H:%M')
    else:
        display_df['Time'] = display_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
    
    display_df['Predicted Vehicles'] = display_df['Predicted Vehicles'].round(0).astype(int)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ── Download Option ────────────────────────────────
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"traffic_forecast_junction_{junction}_{forecast_days}days_{date}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ── Footer ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em; margin-top: 20px;">
    <p>🚦 Smart City Traffic Forecasting System</p>
    <p>Powered by LightGBM Machine Learning Model</p>
    <p style="color: #999; font-size: 0.85em;">Last Updated: {}</p>
</div>
""".format(datetime.now().strftime('%B %d, %Y at %H:%M')), unsafe_allow_html=True)

