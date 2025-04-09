import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SALES_FILE = "raw data/sales_data_extracted.csv"
# This file contains the isHoliday column
HOLIDAY_FILE = "raw data/event_holiday_data.csv"
SALES_COLUMN_NAME = "amount"
# --- End Configuration ---


def visualize_sales_vs_holiday(sales_file, holiday_file, sales_column):
    """
    Loads sales and holiday data, merges them, aggregates daily sales,
    and plots sales comparisons for holiday vs. non-holiday days.
    """
    try:
        # Load datasets
        print(f"Loading sales data from: {sales_file}")
        sales_df = pd.read_csv(sales_file)
        print(f"Loading holiday data from: {holiday_file}")
        holiday_df = pd.read_csv(holiday_file)
        print("Data loaded successfully.")

        # --- Data Preparation ---
        # Convert date columns to datetime
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        holiday_df["date"] = pd.to_datetime(holiday_df["date"])

        # Process 'hour' columns if present in sales data
        # We only need date-level holiday info, so simplify holiday_df
        # Get unique date and isHoliday status
        holiday_info = holiday_df[["date", "isHoliday"]].drop_duplicates(
            subset=["date"]
        )
        if "isHoliday" not in holiday_info.columns:
            print(f"Error: 'isHoliday' column not found in {holiday_file}.")
            return
        print("Processed holiday data to daily level.")

        # Merge sales data with daily holiday info
        merged_df = pd.merge(sales_df, holiday_info, on="date", how="left")
        print("Sales and daily holiday data merged.")

        # Fill missing isHoliday flags (for dates not in holiday file?) with 0
        merged_df["isHoliday"] = merged_df["isHoliday"].fillna(0)
        merged_df["isHoliday"] = merged_df["isHoliday"].astype(int)

        # Check if the sales column exists
        if sales_column not in merged_df.columns:
            print(f"Error: Sales column '{sales_column}' not found. Check config.")
            print(f"Available columns: {merged_df.columns.tolist()}")
            return

        # Aggregate daily sales and determine if it was a holiday
        print("Aggregating daily sales and identifying holiday days...")
        # Note: No need to check hourly holiday status, holiday is daily
        daily_data = (
            merged_df.groupby(["date", "isHoliday"])
            .agg(total_sales=(sales_column, "sum"))
            .reset_index()
        )
        print("Aggregation complete.")
        print(daily_data.head())

        # Calculate average daily sales for holiday vs non-holiday days
        avg_sales_by_holiday = (
            daily_data.groupby("isHoliday")["total_sales"].mean().reset_index()
        )
        # Map 0/1 to meaningful labels
        avg_sales_by_holiday["Day Type"] = avg_sales_by_holiday["isHoliday"].map(
            {0: "Non-Holiday", 1: "Holiday"}
        )
        print("\nAverage Daily Sales:")
        print(avg_sales_by_holiday[["Day Type", "total_sales"]])

        # --- Plotting ---
        if avg_sales_by_holiday.empty:
            print("\nNo data available to plot after aggregation.")
            return

        print("\nGenerating plots (Bar Chart and Box Plot in separate windows)...")

        # --- Figure 1: Bar Chart (Average Daily Sales) ---
        plt.figure(figsize=(8, 6))  # Create the first figure
        ax_bar = sns.barplot(
            x="Day Type",
            y="total_sales",
            data=avg_sales_by_holiday,
            palette=["lightblue", "salmon"],
            hue="Day Type",
            legend=False,
        )

        # Add labels on top of bars
        max_sales_val_avg = avg_sales_by_holiday["total_sales"].max()
        for index, row in avg_sales_by_holiday.iterrows():
            y_pos_bar = row.total_sales + 0.01 * max_sales_val_avg
            # Use ax_bar reference for text placement
            ax_bar.text(
                index, y_pos_bar, f"{row.total_sales:.2f}", color="black", ha="center"
            )

        plt.title("Average Daily Sales on Holiday vs. Non-Holiday Days", fontsize=14)
        plt.xlabel("Day Type", fontsize=12)
        plt.ylabel("Average Daily Sales", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        # --- Figure 2: Box Plot (Daily Sales Distribution) ---
        plt.figure(figsize=(8, 6))  # Create the second figure
        # Map isHoliday 0/1 to labels in the daily aggregated data
        daily_data["Day Type"] = daily_data["isHoliday"].map(
            {0: "Non-Holiday", 1: "Holiday"}
        )

        # No need to assign the return value if not used later for modifications
        sns.boxplot(
            x="Day Type",
            y="total_sales",
            data=daily_data,
            palette=["lightblue", "salmon"],
            hue="Day Type",
            legend=False,
        )

        plt.title(
            "Distribution of Daily Sales on Holiday vs. Non-Holiday Days", fontsize=14
        )
        plt.xlabel("Day Type", fontsize=12)
        plt.ylabel("Total Daily Sales", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        print("Plots generated. Displaying windows...")
        plt.show()  # Display both figures

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}. Check paths.")
    except KeyError as e:
        print(f"Error: Missing expected column - {e}. Verify CSV columns.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    print("Starting Holiday vs Sales analysis script...")
    visualize_sales_vs_holiday(SALES_FILE, HOLIDAY_FILE, SALES_COLUMN_NAME)
    print("Script finished.")
