import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to construct path relative to script directory
def data_path(filename):
    return os.path.join(script_dir, filename)

# Load sales data
sales_data_dec = pd.read_csv(data_path('sales_data_processed.csv'))
sales_data_feb = pd.read_csv(data_path('sales_data_processed_202502.csv'))  # February data used as test set

# Load event and holiday data
event_holiday = pd.read_csv(data_path('event_holiday_data.csv'))

# Load weather data
weather_data = pd.read_csv(data_path('combined_weather_data.csv'))

# Load promotion data
promo_data = pd.read_csv(data_path('promotion_data.csv'))

# Standardize date and hour formats
def preprocess_datetime_hour(df, date_col='date', hour_col='hour'):
    df[date_col] = pd.to_datetime(df[date_col])
    df[hour_col] = df[hour_col].astype(str).str.zfill(2)
    return df

# Process sales data
sales_data_dec = preprocess_datetime_hour(sales_data_dec)
sales_data_feb = preprocess_datetime_hour(sales_data_feb)
sales_data_dec['weekday'] = sales_data_dec['date'].dt.dayofweek
sales_data_feb['weekday'] = sales_data_feb['date'].dt.dayofweek

# Process event/holiday data
event_holiday = preprocess_datetime_hour(event_holiday)

# Process weather data
weather_data = preprocess_datetime_hour(weather_data)
# Select required weather features
weather_features = ['date', 'hour', 'tempreture', 'Clear', 'Clouds', 'Fog', 'Rain']
weather_data = weather_data[weather_features]

# Process promotion data
promo_data = preprocess_datetime_hour(promo_data)
# Select and rename promotion features to avoid conflicts
promo_data = promo_data[['date', 'hour', 'isPromotion']]


# Add promotion impact feature (can be optionally kept or removed)
sales_data_dec = calculate_promotion_impact(sales_data_dec)
sales_data_feb = calculate_promotion_impact(sales_data_feb)

# Merge all data
def merge_all_data(sales_df, event_df, weather_df, promo_df):
    # Ensure merge key types are consistent (date: datetime, hour: string)
    # sales_df, event_df, weather_df, promo_df have already been processed by preprocess_datetime_hour

    # 1. Merge sales with event/holiday data
    merged = pd.merge(
        sales_df,
        event_df[['date', 'hour', 'isEvent', 'isHoliday']],
        on=['date', 'hour'],
        how='left'
    )
    merged['isEvent'] = merged['isEvent'].fillna(0)
    merged['isHoliday'] = merged['isHoliday'].fillna(0)

    # 2. Merge weather data
    merged = pd.merge(
        merged,
        weather_df,
        on=['date', 'hour'],
        how='left'
    )
    # Fill potential missing weather data (e.g., fill with 0 or use more complex methods like interpolation/forward fill)
    weather_cols_to_fill = ['tempreture', 'Clear', 'Clouds', 'Fog', 'Rain']
    # Simple fill with 0, may need adjustment
    merged[weather_cols_to_fill] = merged[weather_cols_to_fill].fillna(0)

    # 3. Merge promotion data
    merged = pd.merge(
        merged,
        promo_df,
        on=['date', 'hour'],
        how='left'
    )
    # Assume no record = no promotion
    merged['isPromotion'] = merged['isPromotion'].fillna(0)

    return merged

# Merge training and test data
train_data = merge_all_data(sales_data_dec, event_holiday, weather_data, promo_data)
test_data = merge_all_data(sales_data_feb, event_holiday, weather_data, promo_data)

# Create features
def create_features(df):
    # Create one-hot encoding for hour
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour')

    # Create one-hot encoding for weekday
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday')

    # --- One-hot encode isPromotion (Revised) ---
    # 1. Define all possible promotion categories (as strings)
    all_promo_categories = ['0.0', '0.2', '0.5'] # Add any other possible values if they exist

    # 2. Convert isPromotion to string category
    df['isPromotion_cat'] = df['isPromotion'].astype(str)

    # 3. Convert it to Categorical type, specifying all possible categories
    df['isPromotion_cat'] = pd.Categorical(
        df['isPromotion_cat'],
        categories=all_promo_categories,
        ordered=False
    )

    # 4. Now perform one-hot encoding, which will always create columns for all defined categories
    promotion_dummies = pd.get_dummies(df['isPromotion_cat'], prefix='promo')
    # print(f"Promotion Dummies for this slice: {promotion_dummies.columns.tolist()}")
    # ------------------------------------------

    # Select base and new features to include
    # !! Removed original 'isPromotion' !!
    feature_columns = [
        'isEvent', 'isHoliday', 'bill', 'promotion_impact', # Original features (promotion_impact optional)
        # 'isPromotion', # <--- Removed original numerical feature
        'tempreture', 'Clear', 'Clouds', 'Fog', 'Rain' # Added weather features
    ]

    # Check if feature columns exist, might be missing due to merge failure
    existing_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        # Warning: Feature columns {missing_features} are missing in DataFrame and will not be included.
        print(f"Warning: Feature columns {missing_features} are missing in DataFrame and will not be included.")

    # Combine all features (hour, weekday, base features, and new promotion dummies)
    features = pd.concat([
        hour_dummies,
        weekday_dummies,
        df[existing_features],
        promotion_dummies # <--- Added promotion dummies
    ], axis=1)

    # --- Added: Ensure train and test sets have consistent feature columns ---
    # (Although the Categorical method should ensure this, can uncomment as a safety measure or alternative)
    # global train_columns # If train_columns is defined outside the function
    # if 'train_columns' in globals():
    #     features = features.reindex(columns=train_columns, fill_value=0)
    # ------------------------------------------

    return features

# Prepare training data
X_train = create_features(train_data)
y_train = train_data['amount']

# --- Added: Store training columns for alignment --- (Complements the reindex safety measure above)
# train_columns = X_train.columns.tolist()
# print(f"Stored training columns: {train_columns}")
# ----------------------------------

# Prepare test data
# Ensure test data has 'amount' column, adjust if not (e.g., for pure future prediction)
if 'amount' in test_data.columns:
    X_test = create_features(test_data)
    y_test = test_data['amount']
else:
    # Warning: Test data is missing 'amount' column, cannot perform evaluation.
    print("Warning: Test data is missing 'amount' column, cannot perform evaluation.")
    X_test = create_features(test_data)
    # Or handle as needed
    y_test = None

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model (only if y_test exists)
if y_test is not None:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Model Evaluation Results:
    print(f"Model Evaluation Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Analyze feature importance
    feature_names = X_train.columns
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    # Sort by absolute value of coefficient
    coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)

    # Most Important Features (Ranked by Impact Size):
    print("Most Important Features (Ranked by Impact Size):")
    # Display more features
    print(coefficients.head(15))

    # Visualize actual vs predicted values
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    # Note: Plot labels/titles remain in Chinese as requested earlier
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales (with Weather & Promotion Data)')
    # Save with new filename
    plt.savefig(data_path('prediction_results_v2.png'))
    plt.close()
    # Chart prediction_results_v2.png has been saved
    print("Chart prediction_results_v2.png has been saved")

# Save prediction results
results_df = pd.DataFrame({
    'Date': test_data['date'],
    'Hour': test_data['hour'],
    # Handle case where y_test might not exist
    'Actual': y_test if y_test is not None else np.nan,
    'Predicted': y_pred
})
if y_test is not None:
    results_df['Difference'] = results_df['Actual'] - results_df['Predicted']
# Original promotion impact factor
results_df['Promotion_Impact'] = test_data['promotion_impact']
# New promotion flag
results_df['isPromotion'] = test_data['isPromotion']
# Optionally add weather features to the results file
# results = pd.concat([results, test_data[weather_features]], axis=1)

# Save with new filename
results_df.to_csv(data_path('prediction_results_v2.csv'), index=False)
# Prediction results saved to prediction_results_v2.csv
print("Prediction results saved to prediction_results_v2.csv") 
