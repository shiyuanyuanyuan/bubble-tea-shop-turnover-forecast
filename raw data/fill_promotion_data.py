import pandas as pd

# Define file path
output_file = "raw data/promotion_data.csv"

# Generate date range
start_date = "2024-12-01"
end_date = "2025-02-28"
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

# Create data list
data = []

# Define promotion period
promo_start = "2025-01-11"
promo_end = "2025-01-19"

# Fill data
for date in date_range:
    date_str = date.strftime("%Y-%m-%d")
    for hour in range(24):
        # Determine if the current date is within the promotion period
        is_promo = 0
        if promo_start <= date_str <= promo_end:
            is_promo = 1

        data.append(
            {
                "date": date_str,
                "hour": f"{hour:02}",
                "timezone": "PST",
                "isPromotion": is_promo,
            }
        )

# Create DataFrame
try:
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} records.")

    # Save to CSV file, overwriting the header created earlier
    df.to_csv(output_file, index=False)
    print(f"Data successfully written to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
