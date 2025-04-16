import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Get the current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Define file paths
weather_data_path = 'extracted_weather_data.csv'
sales_data_path = '../raw data/sales_data_extracted.csv'

# Check if files exist
if not os.path.exists(weather_data_path):
    print(f"Weather data file not found at: {weather_data_path}")
    print(f"Trying alternative path...")
    weather_data_path = 'weatherEDA/extracted_weather_data.csv'
    if not os.path.exists(weather_data_path):
        print(f"Weather data file not found at: {weather_data_path}")
        exit(1)

if not os.path.exists(sales_data_path):
    print(f"Sales data file not found at: {sales_data_path}")
    print(f"Trying alternative path...")
    sales_data_path = 'raw data/sales_data_extracted.csv'
    if not os.path.exists(sales_data_path):
        print(f"Sales data file not found at: {sales_data_path}")
        exit(1)

print(f"Using weather data from: {weather_data_path}")
print(f"Using sales data from: {sales_data_path}")

# Load the datasets
try:
    weather_data = pd.read_csv(weather_data_path)
    print(f"Successfully loaded weather data with {len(weather_data)} rows")
    print(f"Weather data columns: {weather_data.columns.tolist()}")
except Exception as e:
    print(f"Error loading weather data: {e}")
    exit(1)

try:
    sales_data = pd.read_csv(sales_data_path)
    print(f"Successfully loaded sales data with {len(sales_data)} rows")
    print(f"Sales data columns: {sales_data.columns.tolist()}")
except Exception as e:
    print(f"Error loading sales data: {e}")
    exit(1)

# Convert date strings to datetime objects in weather data
weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data['datetime'] = pd.to_datetime(weather_data['date']) + pd.to_timedelta(weather_data['hour'], unit='h')

# Process sales data
# Extract date and convert to datetime
sales_data['date'] = pd.to_datetime(sales_data['date'].str.replace('/', '-'))

# Extract hour range and convert to starting hour
sales_data['hour_start'] = sales_data['hour'].str.split('-').str[0].astype(int)
# Create a new column with just the starting hour (e.g., "11" instead of "11-12")
sales_data['hour_simple'] = sales_data['hour_start']

# Create datetime column for joining
sales_data['datetime'] = pd.to_datetime(sales_data['date']) + pd.to_timedelta(sales_data['hour_start'], unit='h')

# Merge datasets on datetime
combined_data = pd.merge(sales_data, weather_data, on='datetime', how='left')
print(f"Combined data columns: {combined_data.columns.tolist()}")

# Clean up the combined dataset
# Select relevant columns - dynamically check which columns exist
available_columns = ['datetime', 'date_x', 'hour_simple', 'bill', 'pax', 'amount']

# Check for weather columns
for col in ['rain_1h', 'feels_like', 'clouds_all', 'Clear', 'Clouds', 'Fog', 'Rain', 'Snow']:
    if col in combined_data.columns:
        available_columns.append(col)

combined_data = combined_data[available_columns]

# Rename columns for clarity
column_mapping = {
    'date_x': 'date',
    'hour_simple': 'hour',
    'bill': 'num_bills',
    'pax': 'num_customers',
    'amount': 'sales_amount'
}

if 'rain_1h' in combined_data.columns:
    column_mapping['rain_1h'] = 'precipitation'
if 'feels_like' in combined_data.columns:
    column_mapping['feels_like'] = 'temperature'
if 'clouds_all' in combined_data.columns:
    column_mapping['clouds_all'] = 'cloudiness'

combined_data = combined_data.rename(columns=column_mapping)

# Handle missing values
fill_columns = {}
for col in ['precipitation', 'Clear', 'Clouds', 'Fog', 'Rain', 'Snow']:
    if col in combined_data.columns:
        fill_columns[col] = 0

combined_data = combined_data.fillna(fill_columns)

# Filter to include only hours between 11 and 21
combined_data = combined_data[
    (combined_data['hour'] >= 11) & 
    (combined_data['hour'] <= 21)
]

# Save the combined dataset
output_path = '../combined_weather_sales_data.csv'
if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

combined_data.to_csv(output_path, index=False)
print(f"Combined dataset created with {len(combined_data)} rows and saved to {output_path}")
print(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
print(f"Hours included: 11 through 21")
