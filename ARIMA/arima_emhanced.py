import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ==== Load and process sales data ====
df = pd.read_csv('sales_data_processed.csv')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00')
df.set_index('datetime', inplace=True)

# Filter business hours
df = df.between_time('11:00', '21:00')

# ==== Load and process external datasets ====

# Weather data
weather = pd.read_csv('cleaned_weather_data.csv')
weather['datetime'] = pd.to_datetime(weather['date'] + ' ' + weather['hour'].astype(str) + ':00:00')
weather.set_index('datetime', inplace=True)

# Holiday/Event data
holiday = pd.read_csv('event_holiday_data.csv')
holiday['datetime'] = pd.to_datetime(holiday['date'] + ' ' + holiday['hour'].astype(str) + ':00:00')
holiday.set_index('datetime', inplace=True)

# Promotion data
promo = pd.read_csv('promotion_data.csv')
promo['datetime'] = pd.to_datetime(promo['date'] + ' ' + promo['hour'].astype(str) + ':00:00')
promo.set_index('datetime', inplace=True)

# ==== Merge all features ====
df = df.sort_index()
df = df[~df.index.duplicated(keep='first')]  # Remove any duplicate timestamps

df = df.join([
    weather.drop(columns=['date', 'hour'], errors='ignore'),
    holiday[['isHoliday', 'isEvent']],
    promo[['isPromotion']]
], how='left')

# ==== Fill missing values (if any) ====
df.fillna(0, inplace=True)

# ==== Add time-based features ====
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['day_of_month'] = df.index.day

# ==== Define target and exogenous features ====
y = df['amount']
exog_features = [
    'hour', 'day_of_week', 'is_weekend', 'day_of_month',
    'isHoliday', 'isEvent', 'isPromotion',
    'rainfall', 'tempreture', 'Clear', 'Clouds', 'Fog', 'Rain'
]
exog = df[exog_features]

# ==== Auto ARIMA ====
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

# ==== Fit SARIMAX ====
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

# ==== In-sample prediction and evaluation ====
y_pred = results.predict(start=exog.index[0], end=exog.index[-1], exog=exog)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print(f"\nIn-sample R² Score: {r2:.4f}")
print(f"In-sample MAE: {mae:.2f}")

# ==== Plot actual vs predicted ====
plt.figure(figsize=(14, 6))
plt.plot(y.index, y, label='Actual', alpha=0.7)
plt.plot(y_pred.index, y_pred, label='Predicted', linestyle='--')
plt.title(f"SARIMAX Forecast with Weather & Promotion (R² = {r2:.4f}, MAE = {mae:.2f})")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

