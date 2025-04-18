import pandas as pd

# Load CSVs
weather = pd.read_csv("weather_data.csv")
events = pd.read_csv("event_holiday_data.csv")
promo = pd.read_csv("promotion_data.csv")
sales = pd.read_csv("sales_data_processed.csv")

# Only keep 'amount' in sales
sales = sales[["date", "hour", "amount"]]

# Merge everything on date + hour
merged = weather.merge(events, on=["date", "hour"], how="outer") \
                .merge(promo, on=["date", "hour"], how="outer") \
                .merge(sales, on=["date", "hour"], how="outer")

# Filter to business hours (11–21)
merged = merged[(merged["hour"] >= 11) & (merged["hour"] <= 21)]

# Add day of week
merged["date"] = pd.to_datetime(merged["date"])
merged["day_of_week"] = merged["date"].dt.dayofweek  # 0=Monday, 6=Sunday

# Convert temperature to Celsius
merged["temperature_C"] = merged["feels_like"] - 273.15

# Clean column names
merged = merged.rename(columns={
    "isPromotion (means proportion of discont)": "promotion",
})

# Drop unneeded columns
merged = merged.drop(columns=["year", "timezone_x", "timezone_y", "feels_like","clouds_all", "Snow"])

# One-hot encoding: day_of_week and hour
merged = pd.get_dummies(merged, columns=["day_of_week", "hour"], prefix=["dow", "hr"])

# Convert all boolean columns (from one-hot encoding) to integers (0/1)
merged = merged.astype({col: int for col in merged.columns if merged[col].dtype == 'bool'})

# Save for future use
merged.to_csv("final_merged_for_model.csv", index=False)

# Preview result
print(merged.head())

