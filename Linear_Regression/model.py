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
WEATHER_DATA_PATH = f"{BASE_PATH}/cleaned_weather_data.csv"
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

    # remove the hour not in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    df = df[df["hour"].isin(hours)]
    # save the processed data
    df.to_csv("processed_data.csv", index=False)

    # # Add interaction terms
    # # 1. Weather and time interactions
    # for hour in hours:
    #     df[f"rain_hour_{hour}"] = df["Rain"] * df[f"hour_{hour}"]
    #     df[f"clouds_hour_{hour}"] = df["Clouds"] * df[f"hour_{hour}"]
    #     df[f"clear_hour_{hour}"] = df["Clear"] * df[f"hour_{hour}"]

    # # 2. Promotion and time interactions
    # for hour in hours:
    #     df[f"promotion_hour_{hour}"] = df["isPromotion"] * df[f"hour_{hour}"]

    # # 3. Holiday and time interactions
    # for hour in hours:
    #     df[f"holiday_hour_{hour}"] = df["isHoliday"] * df[f"hour_{hour}"]
    #     df[f"event_hour_{hour}"] = df["isEvent"] * df[f"hour_{hour}"]

    # Select features
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
    # # Add interaction features
    # interaction_features = []
    # for hour in hours:
    #     interaction_features.extend(
    #         [
    #             f"rain_hour_{hour}",
    #             f"clouds_hour_{hour}",
    #             f"clear_hour_{hour}",
    #             f"promotion_hour_{hour}",
    #             f"holiday_hour_{hour}",
    #             f"event_hour_{hour}",
    #         ]
    #     )

    # features.extend(interaction_features)

    # Prepare feature matrix and target variable
    X = df[features]
    y = df["amount"]

    return X, y


def train_and_evaluate_model():
    """Train and evaluate the linear regression model"""
    print("Loading and processing data...")
    X, y = load_and_process_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train linear regression model
    print("Training linear regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = lr_model.predict(X_train_scaled)
    y_pred_test = lr_model.predict(X_test_scaled)

    # Evaluate model
    print("\nModel evaluation results:")
    print(f"Training R2 score: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Testing R2 score: {r2_score(y_test, y_pred_test):.4f}")
    print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
    print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

    # Feature coefficients
    print("\nTop 20 feature coefficients:")
    coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": lr_model.coef_})
    print(coefficients.sort_values("Coefficient", ascending=False).head(20))

    # Save model and scaler
    joblib.dump(lr_model, "linear_regression_model.joblib")
    joblib.dump(scaler, "scaler.joblib")


if __name__ == "__main__":
    train_and_evaluate_model()