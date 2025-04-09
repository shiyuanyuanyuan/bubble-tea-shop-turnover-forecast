import pandas as pd
import numpy as np
from datetime import datetime

# Load the datasets - fixing the file paths
# Using '../raw data/' to go up one directory level
sales_df = pd.read_csv('../raw data/sales_data_extracted.csv')
weather_df = pd.read_csv('../raw data/processed_weather_data.csv')

# Transform sales data time format
# Extract the first hour from the hour range (e.g., "11-12" becomes "11")
sales_df['hour'] = sales_df['hour'].str.split('-').str[0].astype(int)

# Convert date format in sales data to match weather data
sales_df['date'] = pd.to_datetime(sales_df['date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

# Now we can merge the datasets on date and hour
merged_df = pd.merge(sales_df, weather_df, on=['date', 'hour'], how='inner')

# Check the shape of the merged dataset
print(f"Original sales data shape: {sales_df.shape}")
print(f"Original weather data shape: {weather_df.shape}")
print(f"Merged data shape: {merged_df.shape}")

# Check for duplicates
print(f"Number of unique date-hour combinations: {merged_df[['date', 'hour']].drop_duplicates().shape[0]}")
print(f"Total rows in merged data: {merged_df.shape[0]}")

# Handle duplicates by aggregating weather data for the same date and hour
cleaned_df = merged_df.groupby(['date', 'hour']).agg({
    'bill': 'first',
    'pax': 'first',
    'amount': 'first',
    'amt_percentage': 'first',
    'timezone': 'first',
    'temperature_feel_like': 'mean',
    'rain_1h': 'mean',
    'clouds_all': 'mean',
    'weather_Fog': 'max',
    'weather_Rain': 'max',
    'weather_Clouds': 'max',
    'weather_Clear': 'max',
    'weather_Snow': 'max',
    'weather_Thunderstorm': 'max'
}).reset_index()

print(f"After handling duplicates, rows: {cleaned_df.shape[0]}")

# Check for missing values
missing_values = cleaned_df.isnull().sum()
print("\nMissing values in cleaned data:")
print(missing_values[missing_values > 0])  # Only show columns with missing values

# Add some useful derived columns
# Convert temperature from Kelvin to Celsius
cleaned_df['temperature_celsius'] = cleaned_df['temperature_feel_like'] - 273.15

# Add day of week
cleaned_df['day_of_week'] = pd.to_datetime(cleaned_df['date']).dt.day_name()

# Add is_weekend flag
cleaned_df['is_weekend'] = pd.to_datetime(cleaned_df['date']).dt.dayofweek >= 5

# Save the cleaned merged dataset
cleaned_df.to_csv('combined_sales_weather.csv', index=False)

print(f"\nCleaned combined dataset saved to 'combined_sales_weather.csv'")
print(f"Sample of combined data:")
print(cleaned_df.head())
