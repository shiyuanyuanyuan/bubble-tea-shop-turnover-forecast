import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# File paths
BASE_PATH = "/Users/chenshiyuan/Desktop/NEU/5800/bubble-tea-shop-turnover-forecast/raw_data"
SALES_DATA_PATH = f"{BASE_PATH}/sales_data_processed.csv"
EVENT_HOLIDAY_DATA_PATH = f"{BASE_PATH}/event_holiday_data.csv"
WEATHER_DATA_PATH = f"{BASE_PATH}/cleaned_weather_data.csv"
PROMOTION_DATA_PATH = f"{BASE_PATH}/promotion_data.csv"


def create_interaction_features(df):
    """Create interaction features"""
    # Fill missing values
    df = df.fillna(
        {
            "tempreture": df["tempreture"].mean(),
            "Rain": 0,
            "Fog": 0,
            "Clouds": 0,
            "Clear": 0,
            "isPromotion": 0,
            "isEvent": 0,
            "isHoliday": 0,
        }
    )

    # Temperature binning (using fixed boundaries)
    temp_bins = [-np.inf, 270, 275, 280, 285, np.inf]
    temp_labels = ["Very Cold", "Cold", "Moderate", "Hot", "Very Hot"]
    df["temp_bin"] = pd.cut(df["tempreture"], bins=temp_bins, labels=temp_labels)

    # Create temperature and time interaction features
    for hour in range(11, 22):
        df[f"temp_hour_{hour}"] = df["tempreture"] * df[f"hour_{hour}"]

    # Create promotion and holiday interaction features
    df["promotion_holiday"] = df["isPromotion"] * df["isHoliday"]
    df["promotion_event"] = df["isPromotion"] * df["isEvent"]

    # Create weather combination features
    df["bad_weather"] = (df["Rain"] + df["Fog"]).clip(0, 1)
    df["good_weather"] = df["Clear"].astype(float).fillna(0).astype(int)

    # Create weekend feature
    df["is_weekend"] = (df["Sat"] + df["Sun"]).clip(0, 1)

    # Create peak hour feature
    df["is_peak_hour"] = ((df["hour"] >= 11) & (df["hour"] <= 14)).astype(int)

    return df


def load_and_process_data():
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

    # Create interaction features
    df = create_interaction_features(df)

    # Select features
    base_features = [
        "tempreture",
        "isPromotion",
        "isEvent",
        "isHoliday",
        "Rain",
        "Fog",
        "Clouds",
        "Clear",
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ] + [f"hour_{h}" for h in range(11, 22)]

    interaction_features = [
        "promotion_holiday",
        "promotion_event",
        "bad_weather",
        "good_weather",
        "is_weekend",
        "is_peak_hour",
    ] + [f"temp_hour_{h}" for h in range(11, 22)]

    features = base_features + interaction_features

    # Prepare feature matrix and target variable
    X = df[features]
    y = df["amount"]

    return X, y


def train_and_evaluate_model():
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

    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Use grid search to find best parameters
    print("Starting grid search for best parameters...")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring="r2"
    )
    grid_search.fit(X_train_scaled, y_train)

    # Use best parameters model
    print("\nBest parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    # Evaluate model
    print("\nModel evaluation results:")
    print(f"Training R2 score: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Testing R2 score: {r2_score(y_test, y_pred_test):.4f}")
    print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
    print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

    # Feature importance
    print("\nFeature importance:")
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": best_model.feature_importances_}
    )
    print(feature_importance.sort_values("Importance", ascending=False).head(15))

    # Save model and scaler
    joblib.dump(best_model, "random_forest_model.joblib")
    joblib.dump(scaler, "scaler.joblib")


if __name__ == "__main__":
    train_and_evaluate_model()
