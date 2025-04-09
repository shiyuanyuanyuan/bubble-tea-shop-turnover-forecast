import pandas as pd
import os
from datetime import datetime

# Path to the raw data file
raw_data_path = "../raw data/2a1f28d91ed648e3c2ddc1620bb978b5.csv"

# Check if the file exists
if not os.path.exists(raw_data_path):
    print(f"Error: File {raw_data_path} not found.")
    exit(1)

# Read the CSV file
try:
    df = pd.read_csv(raw_data_path)
    print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Extract year, date, and hour from dt_iso and convert to Vancouver time (GMT-7)
df['datetime'] = pd.to_datetime(df['dt_iso'], format='%Y-%m-%d %H:%M:%S %z UTC')
# Convert to Vancouver time (GMT-7)
df['datetime_vancouver'] = df['datetime'].dt.tz_convert('America/Vancouver')
df['year'] = df['datetime_vancouver'].dt.year
df['date'] = df['datetime_vancouver'].dt.strftime('%Y-%m-%d')
df['month'] = df['datetime_vancouver'].dt.month
df['hour'] = df['datetime_vancouver'].dt.hour

# Filter data for the specified time period (December 2024 to March 2025) in Vancouver time
filtered_df = df[
    ((df['year'] == 2024) & (df['month'] == 12)) | 
    ((df['year'] == 2025) & (df['month'] >= 1) & (df['month'] <= 3))
]

# Further filter for hours between 11 and 21 (inclusive) in Vancouver time
filtered_df = filtered_df[(filtered_df['hour'] >= 11) & (filtered_df['hour'] <= 21)]

# Create a unique identifier for each timestamp (date + hour)
filtered_df['timestamp'] = filtered_df['date'] + '_' + filtered_df['hour'].astype(str)

# Create a new dataframe with aggregated weather conditions
# First, get the basic columns we want
result_df = filtered_df.groupby('timestamp').agg({
    'year': 'first',
    'date': 'first',
    'hour': 'first',
    'rain_1h': 'first',  # Take max rain value for the hour
    'feels_like': 'first',  # Take average feels_like for the hour
    'clouds_all': 'first'  # Take average cloudiness for the hour
}).reset_index()

# Now create boolean columns for each weather type
# Map weather_main to our consolidated categories
def categorize_weather(weather):
    if weather in ['Rain', 'Drizzle']:
        return 'Rain'
    elif weather in ['Mist', 'Fog', 'Haze', 'Dust', 'Smoke']:
        return 'Fog'
    elif weather == 'Snow':
        print("snow")
        return 'Snow'
    elif weather == 'Clouds':
        return 'Clouds'
    elif weather == 'Clear':
        return 'Clear'
    else:
        return 'Other'

# Apply the mapping
filtered_df['weather_category'] = filtered_df['weather_main'].apply(categorize_weather)

# Create a pivot table to get boolean columns for each weather type
weather_pivot = pd.pivot_table(
    filtered_df, 
    index='timestamp', 
    columns='weather_category', 
    values='weather_main',
    aggfunc=lambda x: 1,  # 1 if present
    fill_value=0  # 0 if not present
)

# Ensure all weather types have columns (even if not present in the data)
for weather_type in ['Rain', 'Fog', 'Snow', 'Clouds', 'Clear']:
    if weather_type not in weather_pivot.columns:
        weather_pivot[weather_type] = 0

# Merge the weather boolean columns with our result dataframe
result_df = result_df.merge(weather_pivot, on='timestamp')

# Drop the timestamp column as it was just for merging
result_df = result_df.drop('timestamp', axis=1)

# Handle missing values in rain_1h (replace NaN with 0)
result_df['rain_1h'] = result_df['rain_1h'].fillna(0)

# Save the extracted data to a new CSV file
output_path = "../weatherEDA/extracted_weather_data.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result_df.to_csv(output_path, index=False)

print(f"Data extraction complete. Saved to {output_path}")

# Display a sample of the extracted data
print("\nSample of extracted data:")
print(result_df.head(10))

# Print some basic statistics
print("\nBasic statistics:")
print(f"Total records: {len(result_df)}")
print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
print(f"Years covered: {sorted(result_df['year'].unique())}")
print(f"Hours covered: {sorted(result_df['hour'].unique())}")
