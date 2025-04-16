import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Create a figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load the combined dataset (already cleaned in combine.py)
df = pd.read_csv('combined_weather_sales_data.csv')

# Convert temperature from Kelvin to Celsius
df['temperature_celsius'] = df['temperature'] - 273.15

# Create a weather type column based on the one-hot encoded columns
df['weather_type'] = 'Unknown'
weather_types = ['Clear', 'Clouds', 'Fog', 'Rain', 'Snow']
for weather in weather_types:
    df.loc[df[weather] == 1, 'weather_type'] = weather

# Extract day of week and hour for time-based analysis
df['datetime'] = pd.to_datetime(df['datetime'])
df['day_of_week'] = df['datetime'].dt.day_name()
df['hour_of_day'] = df['datetime'].dt.hour

# Define a custom color palette for weather types - more elegant colors
weather_colors = {
    'Clear': '#E6A817',    # Golden amber
    'Clouds': '#6E7783',   # Steel blue-gray
    'Fog': '#9A8C98',      # Muted lavender
    'Rain': '#4A6FA5',     # Deep blue-slate
    'Snow': '#DFE6ED'      # Pale ice blue
}
# 3. Average sales by weather type
plt.figure(figsize=(10, 6))
avg_sales_by_weather = df.groupby('weather_type')['sales_amount'].mean().sort_values(ascending=False)
# Create a list of colors in the same order as the sorted weather types
bar_colors = [weather_colors[weather] for weather in avg_sales_by_weather.index]
sns.barplot(x=avg_sales_by_weather.index, y=avg_sales_by_weather.values, palette=bar_colors)
plt.title('Average Sales by Weather Type', fontsize=16)
plt.xlabel('Weather Type', fontsize=14)
plt.ylabel('Average Sales Amount', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/avg_sales_by_weather.png')
plt.close()

# 4. Average sales by hour of day and weather type
plt.figure(figsize=(14, 8))
hour_weather_sales = df.groupby(['hour_of_day', 'weather_type'])['sales_amount'].mean().reset_index()
sns.lineplot(data=hour_weather_sales, x='hour_of_day', y='sales_amount', hue='weather_type', 
             marker='o', linewidth=2.5, markersize=10, palette=weather_colors)
plt.title('Average Sales by Hour of Day and Weather Type', fontsize=16)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Average Sales Amount', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(range(11, 22))
plt.legend(title='Weather Type')
plt.tight_layout()
plt.savefig('figures/avg_sales_by_hour_weather.png')
plt.close()

# 16. Average sales amount with change of temperature
plt.figure(figsize=(14, 8))

# Create more granular temperature bins for a smoother curve
temp_bins = np.arange(-5, 16, 1)  # 1°C increments from -5 to 15
df['temp_bin_exact'] = pd.cut(df['temperature_celsius'], bins=temp_bins)

# Calculate average sales for each temperature bin
temp_sales = df.groupby('temp_bin_exact')['sales_amount'].agg(['mean', 'count', 'std']).reset_index()
temp_sales['temp_midpoint'] = temp_sales['temp_bin_exact'].apply(lambda x: x.mid)

# Sort by temperature for proper line plotting
temp_sales = temp_sales.sort_values('temp_midpoint')

# Plot the average sales by temperature
plt.figure(figsize=(16, 10))

# Main line plot
plt.plot(
    temp_sales['temp_midpoint'], 
    temp_sales['mean'], 
    'o-', 
    color='#0066CC', 
    linewidth=3, 
    markersize=10,
    label='Average Sales'
)

# Add confidence interval (mean ± standard error)
plt.fill_between(
    temp_sales['temp_midpoint'],
    temp_sales['mean'] - temp_sales['std'] / np.sqrt(temp_sales['count']),
    temp_sales['mean'] + temp_sales['std'] / np.sqrt(temp_sales['count']),
    color='#0066CC',
    alpha=0.2,
    label='95% Confidence Interval'
)

# Add data point counts as text
for i, row in temp_sales.iterrows():
    plt.text(
        row['temp_midpoint'], 
        row['mean'] + 5, 
        f'n={row["count"]}',
        ha='center',
        va='bottom',
        fontsize=9,
        alpha=0.7
    )

# Add a polynomial trend line
z = np.polyfit(temp_sales['temp_midpoint'], temp_sales['mean'], 3)
p = np.poly1d(z)
x_trend = np.linspace(temp_sales['temp_midpoint'].min(), temp_sales['temp_midpoint'].max(), 100)
plt.plot(
    x_trend, 
    p(x_trend), 
    '--', 
    color='#FF6600', 
    linewidth=2,
    label='Trend Line (Polynomial)'
)

# Add optimal temperature range indicator
optimal_temp_range = temp_sales.loc[temp_sales['mean'] > temp_sales['mean'].quantile(0.75)]
if not optimal_temp_range.empty:
    min_optimal = optimal_temp_range['temp_midpoint'].min()
    max_optimal = optimal_temp_range['temp_midpoint'].max()
    plt.axvspan(min_optimal, max_optimal, alpha=0.2, color='green', label=f'Optimal Range ({min_optimal:.1f}°C to {max_optimal:.1f}°C)')

plt.title('Average Sales Amount by Temperature', fontsize=18)
plt.xlabel('Temperature (°C)', fontsize=16)
plt.ylabel('Average Sales Amount', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(-5, 16, 1), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('figures/avg_sales_by_temperature.png')
plt.close()

# 18. Average sales amount with change of rainfall
plt.figure(figsize=(14, 8))

# Create rainfall bins (with special handling for zero rainfall)
rain_bins = [0, 0.01, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]
# Fix: We need 9 labels for 10 bin edges (one fewer label than edges)
rain_labels = ['0-0.01', '0.01-0.25', '0.25-0.5', '0.5-1', '1-2', '2-5', '5-10', '10-25', '25+']
df['rain_bin'] = pd.cut(df['precipitation'], bins=rain_bins, labels=rain_labels)

# Calculate average sales for each rainfall bin
rain_sales = df.groupby('rain_bin')['sales_amount'].agg(['mean', 'count', 'std']).reset_index()

# Sort by rainfall for proper plotting
rain_sales['rain_numeric'] = rain_sales.index  # Use index as a proxy for rainfall amount

# Plot the average sales by rainfall
plt.figure(figsize=(16, 10))

# Main bar plot
bars = plt.bar(
    rain_sales['rain_bin'], 
    rain_sales['mean'],
    color='#0066CC',  # Changed to match the temperature chart blue color
    alpha=0.7,
    width=0.7
)

# Add error bars
plt.errorbar(
    x=rain_sales['rain_bin'],
    y=rain_sales['mean'],
    yerr=rain_sales['std'] / np.sqrt(rain_sales['count']),
    fmt='none',
    color='black',
    capsize=5,
    label='95% Confidence Interval'
)

# Add data point counts as text
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height + 5, 
        f'n={rain_sales.iloc[i]["count"]}',
        ha='center',
        va='bottom',
        fontsize=10,
        alpha=0.8
    )

# Add a trend line to show the pattern
rain_midpoints = np.arange(len(rain_sales))
z = np.polyfit(rain_midpoints, rain_sales['mean'], 2)
p = np.poly1d(z)
x_trend = np.linspace(0, len(rain_sales)-1, 100)
plt.plot(
    rain_sales['rain_bin'], 
    p(rain_midpoints), 
    '--', 
    color='#FF6600',  # Same orange color as in temperature chart
    linewidth=2.5,
    label='Trend Line'
)

plt.title('Average Sales Amount by Precipitation Level', fontsize=18)
plt.xlabel('Precipitation (mm)', fontsize=16)
plt.ylabel('Average Sales Amount', fontsize=16)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('figures/avg_sales_by_rainfall.png')
plt.close()

# 19. Average sales amount with change of cloudiness
plt.figure(figsize=(14, 8))

# Create cloudiness bins (0-100%)
cloud_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
df['cloud_bin'] = pd.cut(df['cloudiness'], bins=cloud_bins)

# Calculate average sales for each cloudiness bin
cloud_sales = df.groupby('cloud_bin')['sales_amount'].agg(['mean', 'count', 'std']).reset_index()
cloud_sales['cloud_midpoint'] = cloud_sales['cloud_bin'].apply(lambda x: x.mid)

# Sort by cloudiness for proper line plotting
cloud_sales = cloud_sales.sort_values('cloud_midpoint')

# Plot the average sales by cloudiness
plt.figure(figsize=(16, 10))

# Main line plot
plt.plot(
    cloud_sales['cloud_midpoint'], 
    cloud_sales['mean'], 
    'o-', 
    color='#6600CC', 
    linewidth=3, 
    markersize=10,
    label='Average Sales'
)

# Add confidence interval
plt.fill_between(
    cloud_sales['cloud_midpoint'],
    cloud_sales['mean'] - cloud_sales['std'] / np.sqrt(cloud_sales['count']),
    cloud_sales['mean'] + cloud_sales['std'] / np.sqrt(cloud_sales['count']),
    color='#6600CC',
    alpha=0.2,
    label='95% Confidence Interval'
)

# Add data point counts as text
for i, row in cloud_sales.iterrows():
    plt.text(
        row['cloud_midpoint'], 
        row['mean'] + 5, 
        f'n={row["count"]}',
        ha='center',
        va='bottom',
        fontsize=9,
        alpha=0.7
    )

# Add a polynomial trend line
z = np.polyfit(cloud_sales['cloud_midpoint'], cloud_sales['mean'], 3)
p = np.poly1d(z)
x_trend = np.linspace(cloud_sales['cloud_midpoint'].min(), cloud_sales['cloud_midpoint'].max(), 100)
plt.plot(
    x_trend, 
    p(x_trend), 
    '--', 
    color='#FF6600', 
    linewidth=2,
    label='Trend Line (Polynomial)'
)

plt.title('Average Sales Amount by Cloudiness Percentage', fontsize=18)
plt.xlabel('Cloudiness (%)', fontsize=16)
plt.ylabel('Average Sales Amount', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 101, 10), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('figures/avg_sales_by_cloudiness.png')
plt.close()

# 20. Combined weather factors impact on sales (3D visualization)
from mpl_toolkits.mplot3d import Axes3D

# Create simplified bins for both temperature and cloudiness
temp_simple_bins = [-5, 0, 5, 10, 15]
cloud_simple_bins = [0, 25, 50, 75, 100]

df['temp_simple_bin'] = pd.cut(df['temperature_celsius'], bins=temp_simple_bins)
df['cloud_simple_bin'] = pd.cut(df['cloudiness'], bins=cloud_simple_bins)

# Calculate average sales for each combination
combined_sales = df.groupby(['temp_simple_bin', 'cloud_simple_bin'])['sales_amount'].mean().reset_index()
combined_sales['temp_midpoint'] = combined_sales['temp_simple_bin'].apply(lambda x: x.mid)
combined_sales['cloud_midpoint'] = combined_sales['cloud_simple_bin'].apply(lambda x: x.mid)

# Create the 3D plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot with size proportional to sales
scatter = ax.scatter(
    combined_sales['temp_midpoint'],
    combined_sales['cloud_midpoint'],
    combined_sales['sales_amount'],
    c=combined_sales['sales_amount'],
    cmap='viridis',
    s=combined_sales['sales_amount'] * 0.5,  # Size proportional to sales
    alpha=0.7
)

# Add a colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Average Sales Amount', rotation=270, labelpad=20, fontsize=12)

# Set labels and title
ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Cloudiness (%)', fontsize=12)
ax.set_zlabel('Average Sales Amount', fontsize=12)
plt.title('3D Visualization of Sales by Temperature and Cloudiness', fontsize=16)

# Improve the view angle
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig('figures/3d_sales_temp_cloud.png')
plt.close()

# 21. Average sales amount with change of rainfall (line graph version) - IMPROVED
plt.figure(figsize=(16, 10))

# Create more appropriate rainfall bins with better distribution
rain_bins = [0, 0.01, 0.25, 0.5, 1.0, 2.0, 5.0, 100.0]  # Reduced number of bins for higher rainfall
rain_labels = ['0-0.01', '0.01-0.25', '0.25-0.5', '0.5-1', '1-2', '2-5', '5+']  # Combined higher rainfall bins
df['rain_bin_improved'] = pd.cut(df['precipitation'], bins=rain_bins, labels=rain_labels)

# Calculate average sales for each rainfall bin
rain_sales_improved = df.groupby('rain_bin_improved')['sales_amount'].agg(['mean', 'count', 'std']).reset_index()

# Create a mapping for midpoints based on the bin labels
midpoint_mapping = {
    '0-0.01': 0.005,
    '0.01-0.25': 0.13,
    '0.25-0.5': 0.375,
    '0.5-1': 0.75,
    '1-2': 1.5,
    '2-5': 3.5,
    '5+': 10  # Representative midpoint for the last bin
}
rain_sales_improved['rain_midpoint'] = rain_sales_improved['rain_bin_improved'].map(midpoint_mapping)

# Sort by rainfall midpoint for proper line plotting
rain_sales_improved = rain_sales_improved.sort_values('rain_midpoint')

# Main line plot
plt.plot(
    rain_sales_improved['rain_midpoint'], 
    rain_sales_improved['mean'], 
    'o-', 
    color='#0066CC', 
    linewidth=3, 
    markersize=10,
    label='Average Sales'
)

# Add confidence interval
plt.fill_between(
    rain_sales_improved['rain_midpoint'],
    rain_sales_improved['mean'] - rain_sales_improved['std'] / np.sqrt(rain_sales_improved['count']),
    rain_sales_improved['mean'] + rain_sales_improved['std'] / np.sqrt(rain_sales_improved['count']),
    color='#0066CC',
    alpha=0.2,
    label='95% Confidence Interval'
)

# Add data point counts as text
for i, row in rain_sales_improved.iterrows():
    plt.text(
        row['rain_midpoint'], 
        row['mean'] + 5, 
        f'n={row["count"]}',
        ha='center',
        va='bottom',
        fontsize=12,  # Increased font size for better visibility
        fontweight='bold',
        alpha=0.8
    )

# Add a polynomial trend line
z = np.polyfit(rain_sales_improved['rain_midpoint'], rain_sales_improved['mean'], 2)  # Reduced to quadratic
p = np.poly1d(z)
x_trend = np.linspace(rain_sales_improved['rain_midpoint'].min(), rain_sales_improved['rain_midpoint'].max(), 100)
plt.plot(
    x_trend, 
    p(x_trend), 
    '--', 
    color='#FF6600', 
    linewidth=2.5,
    label='Trend Line (Polynomial)'
)

# Use logarithmic scale for x-axis to better show the distribution of rainfall values
plt.xscale('log')

# Add custom x-tick positions and labels
tick_positions = list(midpoint_mapping.values())
tick_labels = list(midpoint_mapping.keys())
plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=12)

# Add a note about sample sizes
plt.figtext(0.5, 0.01, 
           "Note: Sample sizes (n) are shown above each data point. Points with n<5 may not be statistically reliable.", 
           ha="center", fontsize=12, style='italic')

plt.title('Average Sales Amount by Precipitation Level (Line Graph)', fontsize=18)
plt.xlabel('Precipitation (mm)', fontsize=16)
plt.ylabel('Average Sales Amount', fontsize=16)
plt.grid(True, alpha=0.3)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for the note
plt.savefig('figures/avg_sales_by_rainfall_line_improved.png')
plt.close()

# 22. Combined visualization: Sales by precipitation and temperature - IMPROVED
plt.figure(figsize=(16, 10))

# Create simplified temperature categories for better visualization
temp_categories = [
    (-5, 0, "Very Cold (-5°C to 0°C)"),
    (0, 5, "Cold (0°C to 5°C)"),
    (5, 10, "Moderate (5°C to 10°C)"),
    (10, 15, "Warm (10°C to 15°C)")
]

# Create a new column for temperature category
df['temp_category'] = 'Unknown'
for min_temp, max_temp, label in temp_categories:
    df.loc[(df['temperature_celsius'] >= min_temp) & 
           (df['temperature_celsius'] < max_temp), 'temp_category'] = label

# Calculate average sales by rainfall bin and temperature category
rain_temp_sales = df.groupby(['rain_bin_improved', 'temp_category']).agg({
    'sales_amount': 'mean',
    'datetime': 'count'  # Count for sample size
}).reset_index()
rain_temp_sales.rename(columns={'datetime': 'count'}, inplace=True)

# Create a pivot table for easier plotting
pivot_data = rain_temp_sales.pivot(index='rain_bin_improved', columns='temp_category', values='sales_amount')
pivot_counts = rain_temp_sales.pivot(index='rain_bin_improved', columns='temp_category', values='count')

# Define colors for temperature categories
temp_colors = {
    "Very Cold (-5°C to 0°C)": "#0022FF",  # Deep Blue
    "Cold (0°C to 5°C)": "#00AAFF",        # Light Blue
    "Moderate (5°C to 10°C)": "#00FF00",    # Green
    "Warm (10°C to 15°C)": "#FF6600"        # Orange
}

# Plot each temperature category as a separate line
for temp_category in temp_categories:
    category_label = temp_category[2]
    if category_label in pivot_data.columns:
        # Get data for this temperature category
        category_data = pivot_data[category_label].reset_index()
        category_counts = pivot_counts[category_label].reset_index()
        
        # Map rainfall bins to midpoints for x-axis
        category_data['rain_midpoint'] = category_data['rain_bin_improved'].map(midpoint_mapping)
        
        # Sort by rainfall midpoint
        category_data = category_data.sort_values('rain_midpoint')
        
        # Plot the line for this temperature category
        line = plt.plot(
            category_data['rain_midpoint'],
            category_data[category_label],
            'o-',
            linewidth=2.5,
            markersize=8,
            label=category_label,
            color=temp_colors[category_label]
        )
        
        # Add sample size annotations
        for i, row in category_data.iterrows():
            count = category_counts.loc[category_counts['rain_bin_improved'] == row['rain_bin_improved'], category_label].values[0]
            if not np.isnan(row[category_label]):
                plt.text(
                    row['rain_midpoint'],
                    row[category_label] + 5,
                    f'n={int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=temp_colors[category_label],
                    alpha=0.8
                )

# Use logarithmic scale for x-axis
plt.xscale('log')

# Add custom x-tick positions and labels
plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=12)

# Add a note about sample sizes
plt.figtext(0.5, 0.01, 
           "Note: Sample sizes (n) are shown above each data point. Points with n<5 may not be statistically reliable.", 
           ha="center", fontsize=12, style='italic')

plt.title('Average Sales by Precipitation Level and Temperature Range', fontsize=18)
plt.xlabel('Precipitation (mm)', fontsize=16)
plt.ylabel('Average Sales Amount', fontsize=16)
plt.grid(True, alpha=0.3)
plt.yticks(fontsize=12)
plt.legend(title='Temperature Range', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for the note
plt.savefig('figures/sales_by_rainfall_temperature_improved.png')
plt.close()
