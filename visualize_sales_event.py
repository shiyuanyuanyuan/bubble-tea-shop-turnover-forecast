import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Using seaborn for potentially nicer plots

# --- Configuration ---
SALES_FILE = "raw data/sales_data_extracted.csv"
# Assuming the event info is in this file, as it contains isEvent
EVENT_FILE = "raw data/event_holiday_data.csv"
SALES_COLUMN_NAME = "amount"
# --- End Configuration ---


def visualize_sales_vs_event(sales_file, event_file, sales_column):
    """
    Loads sales and event data, merges them, aggregates daily sales,
    and plots average daily sales for event vs. non-event days.
    """
    try:
        # Load datasets
        print(f"Loading sales data from: {sales_file}")
        sales_df = pd.read_csv(sales_file)
        print(f"Loading event data from: {event_file}")
        event_df = pd.read_csv(event_file)
        print("Data loaded successfully.")

        # --- Data Preparation ---
        # Convert date columns to datetime
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        event_df["date"] = pd.to_datetime(event_df["date"])

        # Process 'hour' columns for merging
        if "hour" not in sales_df.columns:
            print(
                "Warning: 'hour' column not found in sales data. Merging on date only."
            )
            # Determine if the day had *any* event hour
            daily_event_info = event_df.groupby("date")["isEvent"].max().reset_index()
            # Ensure 'isEvent' column exists before merging
            if "isEvent" not in daily_event_info.columns:
                print(
                    f"Error: 'isEvent' column not found in {event_file} after grouping."
                )
                return
            merged_df = pd.merge(sales_df, daily_event_info, on="date", how="left")
        else:
            try:
                # Extract starting hour from sales data (e.g., '00' from '00-01')
                sales_df["hour"] = sales_df["hour"].astype(str).str[:2].astype(int)
                # Convert event hour to integer
                event_df["hour"] = event_df["hour"].astype(int)
                print("Processed 'hour' columns for merging.")

                # Select necessary columns from event_df before merging
                event_cols_to_merge = ["date", "hour", "isEvent"]
                if "isEvent" not in event_df.columns:
                    print(f"Error: 'isEvent' column not found in {event_file}.")
                    return

                # Merge sales and event data
                merged_df = pd.merge(
                    sales_df,
                    event_df[event_cols_to_merge],
                    on=["date", "hour"],
                    how="left",
                )
                print("Sales and event data merged.")
            except KeyError as e:
                print(
                    f"Error processing/merging on ['date', 'hour']: Missing column {e}. Check CSVs."
                )
                return
            except ValueError as e:
                print(f"Error converting 'hour' column: {e}. Check format in CSVs.")
                return

        # Fill missing isEvent flags (if merge was 'left') with 0
        merged_df["isEvent"] = merged_df["isEvent"].fillna(0)
        merged_df["isEvent"] = merged_df["isEvent"].astype(int)

        # Check if the sales column exists
        if sales_column not in merged_df.columns:
            print(
                f"Error: Sales column '{sales_column}' not found. Check SALES_COLUMN_NAME."
            )
            print(f"Available columns: {merged_df.columns.tolist()}")
            return

        # --- Analysis at Hourly Level ---
        print("Calculating average hourly sales for event vs non-event hours...")
        # Group directly by isEvent status at the hourly level
        avg_hourly_sales_by_event = (
            merged_df.groupby("isEvent")[sales_column].mean().reset_index()
        )

        # Check if we have data for both event and non-event hours
        if avg_hourly_sales_by_event.empty:
            print("No data available after grouping by hour and event status.")
            return
        if len(avg_hourly_sales_by_event) < 2:
            print(
                "Warning: Data only found for one type (event or non-event hours). Comparison may not be meaningful."
            )

        # Map 0/1 to meaningful labels
        avg_hourly_sales_by_event["Hour Type"] = avg_hourly_sales_by_event[
            "isEvent"
        ].map({0: "Non-Event Hour", 1: "Event Hour"})
        print("\nAverage Hourly Sales:")
        print(avg_hourly_sales_by_event[["Hour Type", sales_column]])

        # --- Plotting ---
        print("\nGenerating combined plot (Bar Chart and Box Plot)...")
        # Create a figure with two subplots side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

        # --- Subplot 1: Bar Chart (Average Hourly Sales) ---
        sns.barplot(
            ax=axes[0],
            x="Hour Type",
            y=sales_column,
            data=avg_hourly_sales_by_event,
            palette=["skyblue", "lightcoral"],
            hue="Hour Type",
            legend=False,
        )

        # Add labels on top of bars for the bar chart
        max_sales_val_avg = avg_hourly_sales_by_event[sales_column].max()
        for index, row in avg_hourly_sales_by_event.iterrows():
            y_pos_bar = row[sales_column] + 0.01 * max_sales_val_avg
            axes[0].text(
                index, y_pos_bar, f"{row[sales_column]:.2f}", color="black", ha="center"
            )

        axes[0].set_title(f"Average Hourly {sales_column.capitalize()}", fontsize=14)
        axes[0].set_xlabel("Hour Type", fontsize=12)
        axes[0].set_ylabel(f"Average Hourly {sales_column.capitalize()}", fontsize=12)
        axes[0].tick_params(axis="x", labelsize=10)
        axes[0].tick_params(axis="y", labelsize=10)

        # --- Subplot 2: Box Plot (Hourly Sales Distribution) ---
        # We need the original hourly data for the box plot
        # Map isEvent 0/1 to labels directly in the merged_df for plotting
        merged_df["Hour Type"] = merged_df["isEvent"].map(
            {0: "Non-Event Hour", 1: "Event Hour"}
        )

        sns.boxplot(
            ax=axes[1],
            x="Hour Type",
            y=sales_column,
            data=merged_df,
            palette=["skyblue", "lightcoral"],
            hue="Hour Type",
            legend=False,
        )

        axes[1].set_title(
            f"Distribution of Hourly {sales_column.capitalize()}", fontsize=14
        )
        axes[1].set_xlabel("Hour Type", fontsize=12)
        axes[1].set_ylabel(f"Hourly {sales_column.capitalize()}", fontsize=12)
        axes[1].tick_params(axis="x", labelsize=10)
        axes[1].tick_params(axis="y", labelsize=10)

        # Improve overall layout
        plt.suptitle(
            f"Sales Analysis: Event vs. Non-Event Hours", fontsize=16, y=1.02
        )  # Add overall title
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to prevent title overlap
        print("Plot generated. Displaying window...")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}. Check paths configuration.")
    except KeyError as e:
        print(f"Error: Missing expected column - {e}. Verify CSV columns.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    print("Starting Event vs Sales analysis script...")
    visualize_sales_vs_event(SALES_FILE, EVENT_FILE, SALES_COLUMN_NAME)
    print("Script finished.")
