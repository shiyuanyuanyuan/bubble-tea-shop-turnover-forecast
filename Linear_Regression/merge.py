import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# File paths
BASE_PATH = "/Users/chenshiyuan/Desktop/NEU/5800/bubble-tea-shop-turnover-forecast/raw_data"
SALES_DATA_PATH = f"{BASE_PATH}/sales_data_processed.csv"
EVENT_HOLIDAY_DATA_PATH = f"{BASE_PATH}/event_holiday_data.csv"
WEATHER_DATA_PATH = f"{BASE_PATH}/weather_data.csv"
PROMOTION_DATA_PATH = f"{BASE_PATH}/promotion_data.csv"


def load_and_process_data():
    """Load and process data from all sources"""
    # Load data
    sales_df = pd.read_csv(SALES_DATA_PATH)
    event_holiday_df = pd.read_csv(EVENT_HOLIDAY_DATA_PATH)
    weather_df = pd.read_csv(WEATHER_DATA_PATH)
    promotion_df = pd.read_csv(PROMOTION_DATA_PATH)

    # Process dates
    for df in [sales_df, event_holiday_df, weather_df, promotion_df]:
        df["date"] = pd.to_datetime(df["date"])

    # Merge data
    df = sales_df.merge(event_holiday_df, on=["date", "hour"], how="left")
    df = df.merge(weather_df, on=["date", "hour"], how="left")
    df = df.merge(promotion_df, on=["date", "hour"], how="left")

    # Fill missing values
    df = df.fillna(
        {
            "tempreture": df["tempreture"].mean(),
            "isPromotion": 0,
            "isEvent": 0,
            "isHoliday": 0,
            "Rain": 0,
            "Fog": 0,
            "Clouds": 0,
            "Clear": 0,
        }
    )

    # Add time features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Create weekday one-hot encoding
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, day in enumerate(weekdays):
        df[day] = (df["day_of_week"] == i).astype(int)

    # Create hour one-hot encoding
    hours = range(11, 22)
    for hour in hours:
        df[f"hour_{hour}"] = (df["hour"] == hour).astype(int)

    # Add interaction terms
    # 1. Weather and time interactions
    for hour in hours:
        df[f"rain_hour_{hour}"] = df["Rain"] * df[f"hour_{hour}"]
        df[f"clouds_hour_{hour}"] = df["Clouds"] * df[f"hour_{hour}"]
        df[f"clear_hour_{hour}"] = df["Clear"] * df[f"hour_{hour}"]

    # 2. Promotion and time interactions
    for hour in hours:
        df[f"promotion_hour_{hour}"] = df["isPromotion"] * df[f"hour_{hour}"]

    # 3. Holiday and time interactions
    for hour in hours:
        df[f"holiday_hour_{hour}"] = df["isHoliday"] * df[f"hour_{hour}"]
        df[f"event_hour_{hour}"] = df["isEvent"] * df[f"hour_{hour}"]

    # remove the hour not in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    df = df[df["hour"].isin(hours)]

    # save the processed data
    df.to_csv("processed_data_1.csv", index=False)


if __name__ == "__main__":
    load_and_process_data()