import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and preprocess data
df = pd.read_csv('sales_data_processed.csv')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
df.set_index('datetime', inplace=True)

# Filter business hours
df = df.between_time('11:00', '21:00')

# Create exogenous features
df_features = df.copy()
df_features['hour'] = df_features.index.hour
df_features['day_of_week'] = df_features.index.dayofweek
df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
df_features['day_of_month'] = df_features.index.day

# Define target and exogenous variables
y = df['amount']
exog = df_features[['hour', 'day_of_week', 'is_weekend', 'day_of_month']]

# Auto ARIMA to find best hyperparameters
print("Running auto_arima to find best parameters...")
auto_model = auto_arima(
    y,
    exogenous=exog,
    seasonal=True,
    m=11,  # 11 business hours per day
    stepwise=True,
    suppress_warnings=True,
    trace=True
)

print(f"\nBest ARIMA order: {auto_model.order}")
print(f"Best Seasonal order: {auto_model.seasonal_order}")

# Fit SARIMAX with best parameters
model = SARIMAX(
    y,
    exog=exog,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

print("\nFitting SARIMAX model...")
results = model.fit(disp=False)

# In-sample prediction
y_pred = results.predict(start=exog.index[0], end=exog.index[-1], exog=exog)

# Evaluate
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print(f"\nIn-sample R² Score: {r2:.4f}")
print(f"In-sample MAE: {mae:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(y.index, y, label='Actual', alpha=0.7)
plt.plot(y_pred.index, y_pred, label='Predicted', linestyle='--')
plt.title(f"SARIMAX Fit with Auto ARIMA (R² = {r2:.4f}, MAE = {mae:.2f})")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to a file instead of displaying it
plt.savefig('plots/sarimax_results.png', dpi=300, bbox_inches='tight')
plt.close()
