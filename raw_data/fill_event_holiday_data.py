import pandas as pd
import holidays

# 生成日期范围
date_range = pd.date_range(start="2024-12-01", end="2025-02-28", freq="D")

# 创建空的 DataFrame
data = []

# 获取加拿大的节假日
ca_holidays = holidays.Canada(prov="BC")

# 填充数据
for date in date_range:
    for hour in range(24):
        is_holiday = 1 if date in ca_holidays else 0
        data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "hour": f"{hour:02}",
                "timezone": "PST",
                "isEvent": 0,
                "isHoliday": is_holiday,
                "isPromotion": 0,
            }
        )

# 创建 DataFrame
df = pd.DataFrame(data)

# 创建 DataFrame
print(f"Total records to write: {len(data)}")  # 打印总记录数
print(df.head())  # 打印前几行数据

# 保存到 CSV 文件
try:
    df.to_csv("raw data/event_holiday_promo_data.csv", index=False)
    print("Data successfully written to event_holiday_promo_data.csv")
except Exception as e:
    print(f"Error writing to CSV: {e}")

# Define file paths
source_file = "raw data/event_holiday_data.csv"
output_file = "raw data/event_holiday_promo_data.csv"

try:
    # Read the existing CSV file
    df = pd.read_csv(source_file)
    print(f"Successfully read {source_file}")

    # Add the isPromotion column with default value 0
    df["isPromotion"] = 0
    print("Added 'isPromotion' column with default value 0")

    # Define the promotion date range
    promo_start_date = "2025-01-11"
    promo_end_date = "2025-01-19"

    # Apply promotion logic
    try:
        # Create a boolean mask for the date range
        # Assuming 'date' is string 'YYYY-MM-DD'
        promo_mask = (df["date"] >= promo_start_date) & (df["date"] <= promo_end_date)
        # Use .loc to update 'isPromotion' based on the mask
        df.loc[promo_mask, "isPromotion"] = 1

        print(
            f"Set 'isPromotion' to 1 for dates between "
            f"{promo_start_date} and {promo_end_date}"
        )
        # Print how many rows were changed
        promo_count = promo_mask.sum()
        print(f"Number of promotion entries updated: {promo_count}")

    except TypeError as te:
        print(f"Error comparing dates: {te}. Check 'date' column format.")
    except KeyError:
        # Handle case where 'date' column might be missing
        print("Error: 'date' column not found. Cannot apply promotion.")

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Data successfully written to {output_file}")

except FileNotFoundError:
    print(f"Error: Source file '{source_file}' not found. Please ensure it exists.")
except KeyError as e:
    # Catches KeyError if columns are missing during read or access
    print(f"Error: Column '{e}' not found. Please check CSV column names.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
