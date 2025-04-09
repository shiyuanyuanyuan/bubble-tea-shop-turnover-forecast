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
df = pd.read_csv('combined_sales_weather.csv')

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Temperature range: {df['temperature_celsius'].min():.1f}°C to {df['temperature_celsius'].max():.1f}°C")
print(f"Rainfall range: {df['rain_1h'].min():.2f} to {df['rain_1h'].max():.2f} mm/h")

# 1. Basic statistics for sales and weather variables
print("\nSummary Statistics:")
print(df[['amount', 'temperature_celsius', 'rain_1h', 'clouds_all']].describe())

# Calculate average sales by hour and day type (weekend/weekday)
hourly_avg = df.groupby(['hour', 'is_weekend'])['amount'].mean().reset_index()
hourly_avg.columns = ['hour', 'is_weekend', 'avg_hourly_sales']

# Merge back to original dataframe
df = pd.merge(df, hourly_avg, on=['hour', 'is_weekend'], how='left')

# Create normalized sales (relative to typical sales for that hour and day type)
df['normalized_sales'] = df['amount'] / df['avg_hourly_sales']

# 2. Normalized Sales vs Temperature Analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(x='temperature_celsius', y='normalized_sales', data=df, alpha=0.6)
plt.title('Normalized Sales vs Temperature (Controlling for Hour and Weekend)', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Sales Relative to Hourly Average', fontsize=12)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)  # Reference line at y=1

# Add a trend line
z = np.polyfit(df['temperature_celsius'], df['normalized_sales'], 1)
p = np.poly1d(z)
plt.plot(df['temperature_celsius'], p(df['temperature_celsius']), "r--")

# Add correlation coefficient
corr = df['temperature_celsius'].corr(df['normalized_sales'])
plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)

plt.savefig('figures/normalized_sales_vs_temperature.png')
plt.close()

# 3. Normalized Sales vs Rainfall Analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(x='rain_1h', y='normalized_sales', data=df, alpha=0.6)
plt.title('Normalized Sales vs Rainfall (Controlling for Hour and Weekend)', fontsize=14)
plt.xlabel('Rainfall (mm/h)', fontsize=12)
plt.ylabel('Sales Relative to Hourly Average', fontsize=12)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)  # Reference line at y=1

# Add a trend line for non-zero rainfall
non_zero_rain = df[df['rain_1h'] > 0]
if len(non_zero_rain) > 1:
    z = np.polyfit(non_zero_rain['rain_1h'], non_zero_rain['normalized_sales'], 1)
    p = np.poly1d(z)
    plt.plot(non_zero_rain['rain_1h'], p(non_zero_rain['rain_1h']), "r--")

# Add correlation coefficient
corr = df['rain_1h'].corr(df['normalized_sales'])
plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)

plt.savefig('figures/normalized_sales_vs_rainfall.png')
plt.close()

# 4. Normalized Sales by Weather Condition
weather_conditions = ['weather_Fog', 'weather_Rain', 'weather_Clouds', 
                      'weather_Clear', 'weather_Snow', 'weather_Thunderstorm']

# Calculate average normalized sales for each weather condition
weather_sales = []
for condition in weather_conditions:
    condition_data = df[df[condition] == 1]
    if len(condition_data) > 0:  # Only include conditions that exist in the data
        avg_norm_sales = condition_data['normalized_sales'].mean()
        count = len(condition_data)
        weather_sales.append({
            'condition': condition.replace('weather_', ''),
            'avg_norm_sales': avg_norm_sales,
            'count': count
        })

weather_sales_df = pd.DataFrame(weather_sales)
weather_sales_df = weather_sales_df.sort_values('avg_norm_sales', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(weather_sales_df['condition'], weather_sales_df['avg_norm_sales'])
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)  # Reference line at y=1

# Add count labels on top of bars
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f"n={weather_sales_df['count'].iloc[i]}", 
             ha='center', va='bottom', fontsize=10)

plt.title('Normalized Sales by Weather Condition (Controlling for Hour and Weekend)', fontsize=14)
plt.xlabel('Weather Condition', fontsize=12)
plt.ylabel('Sales Relative to Hourly Average', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/normalized_sales_by_weather.png')
plt.close()

# 5. Temperature Categories Analysis
df['temp_category'] = pd.cut(
    df['temperature_celsius'],
    bins=[-5, 0, 5, 10, 15],
    labels=['Very Cold (< 0°C)', 'Cold (0-5°C)', 'Moderate (5-10°C)', 'Warm (> 10°C)']
)

plt.figure(figsize=(12, 6))
sns.boxplot(x='temp_category', y='normalized_sales', data=df)
plt.title('Normalized Sales by Temperature Range (Controlling for Hour and Weekend)', fontsize=14)
plt.xlabel('Temperature Range', fontsize=12)
plt.ylabel('Sales Relative to Hourly Average', fontsize=12)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)  # Reference line at y=1
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/normalized_sales_by_temp_category.png')
plt.close()

# 6. Cloud Cover Impact
plt.figure(figsize=(12, 6))
df['cloud_category'] = pd.cut(
    df['clouds_all'],
    bins=[0, 25, 50, 75, 100],
    labels=['Clear (0-25%)', 'Partly Cloudy (25-50%)', 'Mostly Cloudy (50-75%)', 'Overcast (75-100%)']
)
sns.boxplot(x='cloud_category', y='normalized_sales', data=df)
plt.title('Normalized Sales by Cloud Cover (Controlling for Hour and Weekend)', fontsize=14)
plt.xlabel('Cloud Cover', fontsize=12)
plt.ylabel('Sales Relative to Hourly Average', fontsize=12)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)  # Reference line at y=1
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/normalized_sales_by_cloud_cover.png')
plt.close()

# 7. Rain Categories Analysis
plt.figure(figsize=(12, 6))
df['rain_category'] = pd.cut(
    df['rain_1h'],
    bins=[-0.001, 0.001, 0.5, 1, 3],
    labels=['No Rain', 'Light Rain (0-0.5mm)', 'Moderate Rain (0.5-1mm)', 'Heavy Rain (>1mm)']
)
sns.boxplot(x='rain_category', y='normalized_sales', data=df)
plt.title('Normalized Sales by Rainfall Intensity (Controlling for Hour and Weekend)', fontsize=14)
plt.xlabel('Rainfall Intensity', fontsize=12)
plt.ylabel('Sales Relative to Hourly Average', fontsize=12)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)  # Reference line at y=1
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/normalized_sales_by_rain_category.png')
plt.close()

# 8. Multiple Regression Analysis on Normalized Sales
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Prepare data for regression
X = df[['temperature_celsius', 'rain_1h', 'clouds_all']]
y = df['normalized_sales']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the model
model = LinearRegression()
model.fit(X_scaled, y)

# Make predictions
y_pred = model.predict(X_scaled)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Create a summary of the regression results
coef_df = pd.DataFrame({
    'Feature': ['Temperature (°C)', 'Rainfall (mm/h)', 'Cloud Cover (%)'],
    'Coefficient': model.coef_
})

print("\nMultiple Linear Regression Results (Normalized Sales):")
print(f"R-squared: {r2:.4f}")
print("\nCoefficients:")
print(coef_df)

# Save regression results to a text file
with open('figures/normalized_regression_results.txt', 'w') as f:
    f.write("Multiple Linear Regression: Normalized Sales vs Weather Factors\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"R-squared: {r2:.4f}\n\n")
    f.write("Coefficients:\n")
    f.write(coef_df.to_string(index=False))
    f.write("\n\nNote: This analysis controls for hour of day and weekend effects by using normalized sales.")

print("\nAnalysis complete! Check the 'figures' directory for visualizations.")

# Analyze sales vs temperature by hour
plt.figure(figsize=(15, 10))
business_hours = range(11, 22)  # Assuming business hours are 11am-9pm

for i, hour in enumerate(business_hours):
    plt.subplot(3, 4, i+1)
    hour_data = df[df['hour'] == hour]
    
    if len(hour_data) > 1:  # Only plot if we have enough data points
        sns.scatterplot(x='temperature_celsius', y='amount', data=hour_data, alpha=0.6)
        
        # Add trend line if we have enough points
        if len(hour_data) > 2:
            z = np.polyfit(hour_data['temperature_celsius'], hour_data['amount'], 1)
            p = np.poly1d(z)
            plt.plot(hour_data['temperature_celsius'], p(hour_data['temperature_celsius']), "r--")
            
            # Add correlation coefficient
            corr = hour_data['temperature_celsius'].corr(hour_data['amount'])
            plt.annotate(f"Corr: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)
    
    plt.title(f'Hour {hour}:00', fontsize=12)
    plt.xlabel('Temperature (°C)', fontsize=10)
    plt.ylabel('Sales Amount', fontsize=10)

plt.tight_layout()
plt.savefig('figures/sales_vs_temperature_by_hour.png')
plt.close()

# Analyze sales vs rainfall by hour
plt.figure(figsize=(15, 10))

for i, hour in enumerate(business_hours):
    plt.subplot(3, 4, i+1)
    hour_data = df[df['hour'] == hour]
    
    if len(hour_data) > 1:  # Only plot if we have enough data points
        # Filter to only include data points with some rainfall
        rain_data = hour_data[hour_data['rain_1h'] > 0]
        
        if len(rain_data) > 2:  # Only plot if we have enough rainy data points
            sns.scatterplot(x='rain_1h', y='amount', data=hour_data, alpha=0.6)
            
            # Add trend line
            z = np.polyfit(rain_data['rain_1h'], rain_data['amount'], 1)
            p = np.poly1d(z)
            plt.plot(rain_data['rain_1h'], p(rain_data['rain_1h']), "r--")
            
            # Add correlation coefficient
            corr = hour_data['rain_1h'].corr(hour_data['amount'])
            plt.annotate(f"Corr: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)
            
            # Add count of rainy data points
            plt.annotate(f"n(rain)={len(rain_data)}", xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10)
        else:
            plt.text(0.5, 0.5, "Insufficient rain data", 
                     ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title(f'Hour {hour}:00', fontsize=12)
    plt.xlabel('Rainfall (mm/h)', fontsize=10)
    plt.ylabel('Sales Amount', fontsize=10)

plt.tight_layout()
plt.savefig('figures/sales_vs_rainfall_by_hour.png')
plt.close()

# Daily average sales with temperature and rainfall
# Aggregate data by date
daily_data = df.groupby('date').agg({
    'amount': 'sum',
    'temperature_celsius': 'mean',
    'rain_1h': 'sum'
}).reset_index()

# Convert date to datetime for proper plotting
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data = daily_data.sort_values('date')

# Create a figure with three subplots sharing the x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot daily sales
ax1.plot(daily_data['date'], daily_data['amount'], 'b-', linewidth=2)
ax1.set_ylabel('Daily Sales', fontsize=12)
ax1.set_title('Daily Sales, Temperature, and Rainfall', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot daily average temperature
ax2.plot(daily_data['date'], daily_data['temperature_celsius'], 'r-', linewidth=2)
ax2.set_ylabel('Avg. Temperature (°C)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Plot daily total rainfall
ax3.bar(daily_data['date'], daily_data['rain_1h'], color='skyblue', alpha=0.7)
ax3.set_ylabel('Total Rainfall (mm)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)

# Format x-axis dates
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('figures/daily_sales_temp_rain.png')
plt.close()

# Create a single graph with sales, temperature and rainfall
# Normalize the values to fit on the same scale
daily_data['normalized_sales'] = daily_data['amount'] / daily_data['amount'].max()
daily_data['normalized_temp'] = (daily_data['temperature_celsius'] - daily_data['temperature_celsius'].min()) / \
                               (daily_data['temperature_celsius'].max() - daily_data['temperature_celsius'].min())
daily_data['normalized_rain'] = daily_data['rain_1h'] / daily_data['rain_1h'].max() if daily_data['rain_1h'].max() > 0 else 0

# Create the plot
plt.figure(figsize=(15, 8))

# Plot the three normalized lines
plt.plot(daily_data['date'], daily_data['normalized_sales'], 'b-', linewidth=2, label='Sales')
plt.plot(daily_data['date'], daily_data['normalized_temp'], 'r-', linewidth=2, label='Temperature')
plt.plot(daily_data['date'], daily_data['normalized_rain'], 'g-', linewidth=2, label='Rainfall')

# Add labels and title
plt.xlabel('Date', fontsize=12)
plt.ylabel('Normalized Value (0-1)', fontsize=12)
plt.title('Daily Sales, Temperature, and Rainfall (Normalized)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Format x-axis dates
plt.gcf().autofmt_xdate()

# Add a second y-axis with the actual values
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.set_ylabel('Sales Amount', fontsize=12, color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Add a third y-axis for temperature
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward
ax3.set_ylabel('Temperature (°C)', fontsize=12, color='r')
ax3.tick_params(axis='y', labelcolor='r')

# Add a fourth y-axis for rainfall
ax4 = ax1.twinx()
ax4.spines['right'].set_position(('outward', 120))  # Move the fourth axis further outward
ax4.set_ylabel('Rainfall (mm)', fontsize=12, color='g')
ax4.tick_params(axis='y', labelcolor='g')

plt.tight_layout()
plt.savefig('figures/combined_daily_metrics.png')
plt.close()

# Create two separate graphs: Sales with Temperature and Sales with Rainfall

# 1. Sales with Temperature
plt.figure(figsize=(15, 7))

# Create primary y-axis for sales
ax1 = plt.gca()
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Daily Sales', fontsize=12, color='b')
ax1.plot(daily_data['date'], daily_data['amount'], 'b-', linewidth=2, label='Sales')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.7)

# Create secondary y-axis for temperature
ax2 = ax1.twinx()
ax2.set_ylabel('Temperature (°C)', fontsize=12, color='r')
ax2.plot(daily_data['date'], daily_data['temperature_celsius'], 'r-', linewidth=2, label='Temperature')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

plt.title('Daily Sales and Temperature', fontsize=16)
plt.gcf().autofmt_xdate()  # Format date labels
plt.tight_layout()
plt.savefig('figures/sales_with_temperature.png')
plt.close()

# 2. Sales with Rainfall
plt.figure(figsize=(15, 7))

# Create primary y-axis for sales
ax1 = plt.gca()
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Daily Sales', fontsize=12, color='b')
ax1.plot(daily_data['date'], daily_data['amount'], 'b-', linewidth=2, label='Sales')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.7)

# Create secondary y-axis for rainfall
ax2 = ax1.twinx()
ax2.set_ylabel('Rainfall (mm)', fontsize=12, color='g')
ax2.bar(daily_data['date'], daily_data['rain_1h'], color='g', alpha=0.5, label='Rainfall')
ax2.tick_params(axis='y', labelcolor='g')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

plt.title('Daily Sales and Rainfall', fontsize=16)
plt.gcf().autofmt_xdate()  # Format date labels
plt.tight_layout()
plt.savefig('figures/sales_with_rainfall.png')
plt.close()
