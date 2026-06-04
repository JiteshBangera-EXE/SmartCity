# 🚦 SmartCity Traffic Forecasting

A sophisticated machine learning system for predicting traffic volume at urban traffic junctions using LightGBM and advanced time-series feature engineering.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Results & Visualizations](#results--visualizations)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Project Overview

SmartCity Traffic Forecasting is an intelligent traffic prediction system designed to forecast vehicle traffic volume at urban junctions for the next 24 hours. The system uses a pre-trained LightGBM machine learning model combined with sophisticated feature engineering to provide accurate traffic predictions.

### Key Capabilities:
- ✅ Predict traffic volume for up to 24 hours ahead
- ✅ Support for 4 different traffic junctions
- ✅ Interactive web-based UI using Streamlit
- ✅ Real-time visualizations with Plotly
- ✅ Time-series aware feature engineering
- ✅ Rush hour and seasonal pattern detection

---

## ✨ Features

### Core Prediction Features

#### 1. **Time-Based Features**
- Hour of day (0-23)
- Day of week (0-6)
- Month (1-12)
- Year
- Quarter (Q1-Q4)
- Cyclical encoding for hours, days, and months (sin/cos transformations)

#### 2. **Traffic Pattern Features**
- **Rush Hour Detection**: Identifies peak traffic periods (7-9 AM, 4-7 PM)
- **Night Time Detection**: Flags low-traffic periods (10 PM - 6 AM)
- **Weekend Detection**: Binary flag for weekends (Saturday-Sunday)

#### 3. **Lag Features** (Historical Context)
- 1-hour lag
- 2-hour lag
- 24-hour lag (same time yesterday)
- 168-hour lag (same time last week)

#### 4. **Rolling Statistics**
- 3-hour rolling mean
- 24-hour rolling mean
- 24-hour rolling standard deviation
- 7-day (168-hour) rolling mean

---

## 📁 Project Structure

```
SmartCity/
├── app.py                              # Streamlit web application
├── Folder.py                           # Directory structure initialization
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── data/                               # Data directory
│   ├── raw/                           # Original dataset (train.csv)
│   ├── processed/                     # Cleaned/processed data
│   └── features/                      # Engineered features
│
├── models/                             # Pre-trained models
│   ├── lightgbm_traffic_forecaster.pkl # Trained LightGBM model
│   └── feature_columns.pkl             # Feature columns mapping
│
├── notebooks/                          # Jupyter notebooks for ML pipeline
│   ├── 01_data_loading.ipynb          # Data loading and exploration
│   ├── 02_eda.ipynb                   # Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb   # Feature creation and transformation
│   ├── 04_modeling.ipynb              # Model training and tuning
│   └── 05_evaluation.ipynb            # Model evaluation and metrics
│
├── reports/                            # Output reports and visualizations
│   └── figures/                       # Generated charts and graphs
│
└── traffic_env/                        # Python virtual environment
```

---

## 🛠️ Technology Stack

### Core Libraries
- **Streamlit** (v4.x) - Interactive web application framework
- **Pandas** (v3.x) - Data manipulation and analysis
- **NumPy** (v2.x) - Numerical computations
- **LightGBM** (v4.x) - Gradient boosting machine learning
- **Scikit-learn** (v1.x) - ML utilities and preprocessing
- **Plotly** (v6.x) - Interactive visualizations

### Additional ML Libraries
- **Prophet** (v1.3) - Time-series forecasting alternative
- **XGBoost** (v3.x) - Alternative boosting algorithm
- **Statsmodels** (v0.14) - Statistical modeling
- **Scipy** (v1.17) - Scientific computing

### Development & Analysis
- **Jupyter** (v7.x) - Interactive notebook environment
- **JupyterLab** (v4.x) - Enhanced notebook interface
- **Matplotlib** (v3.x) - Static visualizations
- **Seaborn** (v0.13) - Statistical graphics

### Utilities
- **Joblib** - Model serialization and caching
- **Python-dateutil** - Date/time utilities
- **Holidays** - Holiday calendar detection
- **Requests** - HTTP library

---

## 📊 Data Pipeline

### Stage 1: Data Loading (`01_data_loading.ipynb`)
- Load raw traffic dataset from CSV
- Parse datetime columns
- Explore dataset shape and basic statistics
- Identify missing values and data quality issues

### Stage 2: Exploratory Data Analysis (`02_eda.ipynb`)
- Visualize traffic patterns across junctions
- Analyze temporal trends (hourly, daily, weekly, seasonal)
- Identify traffic anomalies and outliers
- Examine distribution of vehicle counts

### Stage 3: Feature Engineering (`03_feature_engineering.ipynb`)
- Create time-based features (hour, day, month, cyclical encoding)
- Engineer traffic pattern features (rush hour, night time, weekends)
- Calculate lag features from historical data
- Compute rolling statistics and aggregations
- Handle missing values for lag features

### Stage 4: Model Training (`04_modeling.ipynb`)
- Split data into training and validation sets
- Train LightGBM regression model
- Hyperparameter tuning and optimization
- Feature importance analysis
- Model serialization (save as .pkl)

### Stage 5: Evaluation (`05_evaluation.ipynb`)
- Model performance metrics (MAE, RMSE, R²)
- Cross-validation results
- Prediction error analysis
- Visualization of actual vs predicted values
- Residual analysis

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SmartCity
   ```

2. **Create virtual environment**
   ```bash
   python -m venv traffic_env
   
   # Activate (Windows)
   traffic_env\Scripts\activate
   
   # Activate (macOS/Linux)
   source traffic_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create project directories**
   ```bash
   python Folder.py
   ```

---

## 📱 Usage

### Running the Web Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the Interface

1. **Select Junction**: Choose from 4 available traffic junctions (1-4)
2. **Select Date**: Pick the forecast date (defaults to the day after training data ends)
3. **Click "🔮 Forecast Next 24 Hours"**: Generate predictions
4. **View Results**: 
   - Interactive chart showing predicted traffic volume
   - Raw forecast data table with hourly predictions

### Example Workflow

```python
# The app automatically:
1. Loads the pre-trained LightGBM model
2. Retrieves historical data for the selected junction
3. Generates 24 hourly timestamps for the forecast date
4. Calculates all engineered features
5. Makes predictions using the model
6. Visualizes results with Plotly
7. Displays raw prediction numbers
```

---

## 🤖 Model Architecture

### Model Type
**LightGBM Regression** (Gradient Boosting Decision Trees)

### Model Characteristics
- **Framework**: Light Gradient Boosting Machine
- **Task**: Regression (continuous traffic volume prediction)
- **Target Variable**: Vehicle count at each hour
- **Input Features**: 25+ engineered features
- **Training Data**: Historical traffic records with multiple junctions

### Why LightGBM?
✅ Fast training and inference
✅ Excellent performance with time-series data
✅ Built-in feature importance
✅ Handles non-linear relationships
✅ Memory efficient
✅ Supports categorical features (junction ID)

### Model Files
- `models/lightgbm_traffic_forecaster.pkl` - Trained model weights
- `models/feature_columns.pkl` - Expected feature list for predictions

---

## 🔧 Feature Engineering Details

### Time Encoding Strategy

#### Linear Features
```
hour: 0-23
day_of_week: 0-6 (Monday=0, Sunday=6)
month: 1-12
year: Integer year
quarter: 1-4
```

#### Cyclical Encoding (Sin/Cos)
Captures circular nature of time:
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
dow_sin = sin(2π × day_of_week / 7)
dow_cos = cos(2π × day_of_week / 7)
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

#### Binary Indicators
```
is_weekend: 1 if Saturday/Sunday, 0 otherwise
is_rush_hour: 1 if 7-9 AM or 4-7 PM, 0 otherwise
is_night: 1 if 10 PM-6 AM, 0 otherwise
```

### Lag Features (Historical Context)
Captures temporal dependencies:
- **lag_1h**: Vehicle count 1 hour ago
- **lag_2h**: Vehicle count 2 hours ago
- **lag_24h**: Vehicle count at same time yesterday
- **lag_168h**: Vehicle count at same time last week

### Rolling Aggregations (Statistical Context)
```
rolling_mean_3h: Average of last 3 hours
rolling_mean_24h: Average of last 24 hours
rolling_std_24h: Standard deviation of last 24 hours
rolling_mean_7d: Average of last 7 days (168 hours)
```

---

## 📈 Results & Visualizations

### Output Dashboard
The Streamlit app displays:

1. **Interactive Time Series Chart**
   - X-axis: Hourly timestamps
   - Y-axis: Predicted vehicle count
   - Line plot with markers
   - Dark theme for clarity
   - Hover information for each point

2. **Raw Forecast Table**
   - DateTime column
   - Predicted vehicle count for each hour
   - Sortable and filterable data

### Typical Patterns Detected
- ⬆️ **Morning Rush**: Peak traffic 7-9 AM
- ⬇️ **Mid-day Dip**: Lower traffic 10 AM-3 PM
- ⬆️ **Evening Rush**: Peak traffic 4-7 PM
- ⬇️ **Night Trough**: Minimal traffic 11 PM-6 AM
- 📅 **Weekly Cycles**: Differences between weekdays and weekends
- 🌡️ **Seasonal Variations**: Monthly and quarterly patterns

---

## 🎓 Notebook Descriptions

### 01_data_loading.ipynb
**Purpose**: Initial data exploration and loading
- Load train.csv from raw data directory
- Convert DateTime to proper format
- Display dataset info (shape, columns, dtypes)
- Check for missing values
- Statistical summary of traffic volumes

### 02_eda.ipynb
**Purpose**: Understand traffic patterns through visualization
- Traffic distribution by junction
- Hourly traffic patterns
- Daily and weekly trends
- Seasonal analysis
- Outlier detection
- Correlation analysis

### 03_feature_engineering.ipynb
**Purpose**: Create predictive features
- Time feature extraction
- Cyclical encoding implementation
- Lag feature calculation
- Rolling statistics computation
- Feature scaling and normalization
- Handling NaN values from lag features

### 04_modeling.ipynb
**Purpose**: Train and optimize the LightGBM model
- Data splitting (train/validation)
- LightGBM hyperparameter tuning
- Model training with cross-validation
- Feature importance ranking
- Model serialization to .pkl files

### 05_evaluation.ipynb
**Purpose**: Assess model performance
- Prediction accuracy metrics (MAE, RMSE, R²)
- Cross-validation scores
- Prediction vs actual visualization
- Residual analysis and error distribution
- Per-junction performance breakdown

---

## 🔮 Prediction Workflow in app.py

```
User Input (Junction, Date)
    ↓
Load Model & Historical Data
    ↓
Generate 24 Future Timestamps
    ↓
Fetch Last 168 Hours Historical Data (Context)
    ↓
Combine Historical + Future Data
    ↓
Calculate All Features:
  - Time Features (hour, day, month, cyclical)
  - Traffic Patterns (rush hour, night, weekend)
  - Lag Features (1h, 2h, 24h, 168h)
  - Rolling Statistics (3h, 24h, 7d)
    ↓
Extract Only Future Hours (NaN Vehicles)
    ↓
Model Prediction (LightGBM)
    ↓
Visualize Results (Plotly)
    ↓
Display Raw Predictions (Table)
```

---

## 💡 Key Implementation Details

### Caching Strategy
The Streamlit app uses `@st.cache_data` to cache:
- Pre-trained LightGBM model
- Feature columns list
- Historical traffic data

This prevents reloading these large assets on every user interaction.

### Feature Consistency
The app maintains feature consistency by:
1. Using pre-saved feature column names from training
2. Ensuring exact feature order matches model expectations
3. Calculating all features for both historical and future data
4. Filtering only necessary columns before prediction

### Data Stitching
For accurate lag features during forecasting:
1. Loads last 168 hours (7 days) of historical data
2. Concatenates with 24 future timestamps
3. Calculates lag features on the combined dataset
4. Extracts only the 24 future predictions

---

## 🚀 Future Enhancements

### Potential Improvements
- [ ] Add multiple traffic junction comparison dashboard
- [ ] Implement confidence intervals for predictions
- [ ] Add external features (weather, events, holidays)
- [ ] Support for longer forecast horizons (48h, 7-day)
- [ ] Model retraining pipeline with new data
- [ ] Anomaly detection for traffic incidents
- [ ] Real-time data integration from traffic sensors
- [ ] Automatic model performance monitoring
- [ ] A/B testing for different model architectures
- [ ] REST API for programmatic access
- [ ] Database backend for historical predictions
- [ ] Mobile-friendly responsive design

### Potential Model Improvements
- Ensemble methods combining LightGBM with Prophet
- LSTM/GRU neural networks for sequence learning
- Attention mechanisms for temporal dependencies
- External regressors (weather, events)
- Hierarchical time-series forecasting (city-level to junction-level)

---

## 📝 Configuration & Customization

### Adjusting Prediction Parameters

**Edit `app.py` to customize:**
```python
# Line 25: Change available junctions
junction = st.sidebar.selectbox("Select Junction", [1, 2, 3, 4])

# Line 35: Change forecast periods
future_hours = pd.date_range(str(date), periods=24, freq='H')

# Line 53-54: Modify rush hour definition
combined_df['is_rush_hour'] = combined_df['hour'].isin([7, 8, 9, 16, 17, 18, 19])

# Line 54: Modify night time definition
combined_df['is_night'] = (combined_df['hour'].between(22, 23) | ...)
```

### Feature Customization

Add new features in the feature engineering section:
```python
combined_df['holiday'] = combined_df['DateTime'].dt.dayofyear.isin([...])
combined_df['custom_feature'] = combined_df['some_column'].apply(...)
```

---

## 🔗 Dependencies

See `requirements.txt` for complete list. Key dependencies:
- streamlit==1.x
- pandas==3.0.3
- numpy==2.4.6
- lightgbm==4.6.0
- plotly==6.7.0
- scikit-learn==1.8.0
- joblib==1.5.3

---

## 📄 License

This project is provided as-is for educational and development purposes.

---

## 👤 Author

SmartCity Traffic Forecasting System
Created as part of intelligent urban traffic management research.

---

## 📧 Support & Questions

For questions or issues, please refer to the project documentation or examine the Jupyter notebooks for detailed implementation explanations.

---

**Last Updated**: June 2026  
**Project Status**: Active Development
