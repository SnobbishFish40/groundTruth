from analysis import DataPreprocessor, ProphetForecaster, ForecastVisualizer
import pandas as pd

# =============================================================================
# CONFIGURATION - Change these to analyze different properties
# =============================================================================
TARGET_COLUMN = 'T2M'  # Change this to any column you want to forecast
                               # Examples: 'T2M', 'PRECTOTCORR', 'GWETTOP', etc.

PLOT_TITLE = "Temperature Forecast"  # Update title to match your property

# Optional: Customize model parameters based on property type
SEASONALITY_MODE = 'additive'  # 'additive' or 'multiplicative'
CHANGEPOINT_PRIOR = 0.05       # Higher = more flexible trend (0.001-0.5)
# =============================================================================

preprocessor = DataPreprocessor('temperature.csv')
prophet_data = preprocessor.prepare_for_prophet(
    preprocessor.df, 
    target_column=TARGET_COLUMN, 
    year_column='YEAR', 
    doy_column='DOY'
)

forecaster = ProphetForecaster(
    seasonality_mode=SEASONALITY_MODE,
    changepoint_prior_scale=CHANGEPOINT_PRIOR
)
forecaster.create_model(
    yearly_seasonality=True,
    weekly_seasonality=False,
#    growth='flat'  # Flat trend - focuses on seasonality only
)
forecaster.fit(prophet_data)
forecast = forecaster.predict(periods=365, frequency='D', include_history=True)

print(forecast)

visualizer = ForecastVisualizer()
visualizer.plot_forecast(forecaster.model, forecast, historical_data=prophet_data, 
                        title=PLOT_TITLE)
visualizer.plot_components(forecaster.model, forecast)
visualizer.create_dashboard(forecaster.model, forecast, prophet_data)
