import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# === Load and Clean Data ===
df = pd.read_csv("processed_data_1.csv").dropna()

# Convert date to datetime first
df['date'] = pd.to_datetime(df['date'])

# Create train and test sets based on date ranges
train_mask = (df['date'] >= '2024-12-01') & (df['date'] <= '2025-01-31')
test_mask = (df['date'] >= '2025-02-01') & (df['date'] <= '2025-02-28')


# select the features
features = [
        "tempreture",  # temperature
        "isPromotion",  # discount (50% & 20%)
        "isEvent",  # isEvent
        "isHoliday",  # isHoliday
        "Rain",  # weather_rain
        "Fog",  # weather_fog
        "Clouds",  # weather_cloudy
        "Clear",  # weather_clear
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",  # weekdays
    ] + [f"hour_{h}" for h in range(11, 22)]  # hours
print(features)

# Split features and target AFTER creating masks
X = df[features]
y = df["amount"]

# split the data into train and test
X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

# === Use Regression to reduce multicollinearity effect ===
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === Evaluation ===
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# === Bootstrapped Coefficient Uncertainty ===
np.random.seed(1)
n_bootstraps = 1000

coefs = np.array([
    Ridge(alpha=1.0).fit(*resample(X, y)).coef_
    for _ in range(n_bootstraps)
])

mean_coef = np.mean(coefs, axis=0)
std_err = np.std(coefs, axis=0)

coef_summary = pd.DataFrame({
    "Feature": X.columns,
    "Effect": mean_coef.round(2),
    "Uncertainty (±)": std_err.round(2)
}).sort_values(by="Effect", key=abs, ascending=False)

print("\n📊 Bootstrapped Coefficient Summary (Ridge):")
print(coef_summary)

# === Plotting ===
comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "date": df[test_mask]["date"]  # Add the date column
}).reset_index(drop=True)

plt.figure(figsize=(14, 6))
plt.plot(comparison_df["date"], comparison_df["Actual"], label="Actual", marker='o', linestyle='-')
plt.plot(comparison_df["date"], comparison_df["Predicted"], label="Predicted", marker='x', linestyle='--')
plt.title("Actual vs Predicted Bubble Tea Sales (Ridge Regression)")
plt.xlabel("Date")
plt.ylabel("Amount ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()
plt.show()