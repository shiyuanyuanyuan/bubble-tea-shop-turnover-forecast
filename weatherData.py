import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re

# Read the raw data
def process_weather_data():
    # Read the raw data
    raw_data = pd.read_csv('raw data/2a1f28d91ed648e3c2ddc1620bb978b5.csv')
    
    # Clean the dt_iso column to remove the UTC part and then convert to datetime
    def clean_datetime(dt_str):
        # Extract just the date and time part before the timezone info
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', dt_str)
        if match:
            return match.group(1)
        return dt_str
    
    # Clean the datetime strings
    raw_data['dt_iso_clean'] = raw_data['dt_iso'].apply(clean_datetime)
    
    # Convert to datetime
    raw_data['dt_iso'] = pd.to_datetime(raw_data['dt_iso_clean'], format='%Y-%m-%d %H:%M:%S')
    
    # Convert UTC to Vancouver timezone
    vancouver_tz = pytz.timezone('America/Vancouver')
    
    # Function to convert UTC timestamp to Vancouver time
    def convert_to_vancouver(utc_time):
        # Make it timezone aware (UTC)
        utc_time = pytz.utc.localize(utc_time)
        # Convert to Vancouver time
        vancouver_time = utc_time.astimezone(vancouver_tz)
        return vancouver_time
    
    # Apply the conversion
    raw_data['vancouver_time'] = raw_data['dt_iso'].apply(convert_to_vancouver)
    
    # Extract date and hour components
    raw_data['date'] = raw_data['vancouver_time'].dt.strftime('%Y-%m-%d')
    raw_data['hour'] = raw_data['vancouver_time'].dt.hour
    
    # Find the year in the data
    data_year = raw_data['vancouver_time'].dt.year.iloc[0]
    
    # Filter data for winter months - make an explicit copy
    winter_data = raw_data[
        ((raw_data['vancouver_time'].dt.month == 12) & (raw_data['vancouver_time'].dt.day >= 1)) |
        ((raw_data['vancouver_time'].dt.month == 1) & (raw_data['vancouver_time'].dt.day <= 31)) |
        ((raw_data['vancouver_time'].dt.month == 2) & (raw_data['vancouver_time'].dt.day <= 28))
    ].copy()  # Added .copy() to create an explicit copy
    
    # Filter for hours between 11 and 21 (inclusive)
    winter_data = winter_data[(winter_data['hour'] >= 11) & (winter_data['hour'] <= 21)]
    
    # Combine similar weather types using .loc
    # Combine Drizzle with Rain
    winter_data.loc[:, 'weather_main'] = winter_data['weather_main'].replace('Drizzle', 'Rain')
    # Combine Mist, Haze, Dust, and Fog into Fog
    winter_data.loc[:, 'weather_main'] = winter_data['weather_main'].replace(['Mist', 'Haze', 'Dust', 'Fog'], 'Fog')
    
    # Get all unique weather_main values
    unique_weather = winter_data['weather_main'].unique()
    
    # Create binary columns for each weather type
    for weather_type in unique_weather:
        winter_data.loc[:, f'weather_{weather_type}'] = (winter_data['weather_main'] == weather_type).astype(int)
    
    # Group by date and hour to get hourly data
    agg_dict = {
        'feels_like': 'mean',
        'rain_1h': 'mean',
        'clouds_all': 'mean'
    }
    
    # Add aggregation for each weather type column
    for weather_type in unique_weather:
        agg_dict[f'weather_{weather_type}'] = 'max'
    
    grouped_data = winter_data.groupby(['date', 'hour']).agg(agg_dict).reset_index()
    
    # Replace NaN values
    grouped_data['rain_1h'] = grouped_data['rain_1h'].fillna(0)
    
    # Create the final dataframe
    result = pd.DataFrame()
    result['date'] = grouped_data['date']
    result['hour'] = grouped_data['hour']
    result['timezone'] = 'America/Vancouver'
    result['temperature_feel_like'] = grouped_data['feels_like']
    result['rain_1h'] = grouped_data['rain_1h']
    result['clouds_all'] = grouped_data['clouds_all']
    
    # Add the binary weather columns
    for weather_type in unique_weather:
        result[f'weather_{weather_type}'] = grouped_data[f'weather_{weather_type}']
    
    # Adjust the year to 2024-2025
    result['date'] = result['date'].apply(
        lambda x: x.replace(str(data_year), '2024') if '-12-' in x else 
                 x.replace(str(data_year + 1 if data_year + 1 <= raw_data['vancouver_time'].dt.year.max() else data_year), '2025')
    )
    
    # Sort by date and hour
    result = result.sort_values(['date', 'hour'])
    
    # Save to CSV
    result.to_csv('processed_weather_data.csv', index=False)
    print(f"Processed data saved to processed_weather_data.csv")
    print(f"Weather types found: {', '.join(unique_weather)}")
    
    return result

if __name__ == "__main__":
    process_weather_data()
