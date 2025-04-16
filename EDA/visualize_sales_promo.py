import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np

# --- Configuration ---
SALES_FILE = "raw data/sales_data_extracted.csv"
PROMO_FILE = "raw data/promotion_data.csv"
# !!! IMPORTANT: Please verify this is the correct column name for sales figures !!!
SALES_COLUMN_NAME = "amount"
# --- End Configuration ---


def visualize_sales_and_promotions(sales_file, promo_file, sales_column):
    """
    Loads sales and promotion data, merges them, aggregates sales daily,
    and plots daily sales highlighting the promotion period and its intensity.
    """
    try:
        # Load datasets
        print(f"Loading sales data from: {sales_file}")
        sales_df = pd.read_csv(sales_file)
        print(f"Loading promotion data from: {promo_file}")
        promo_df = pd.read_csv(promo_file)
        print("Data loaded successfully.")

        # --- Data Preparation ---
        # Ensure correct data types with specific date format
        sales_df["date"] = pd.to_datetime(sales_df["date"], format="%Y/%m/%d")
        promo_df["date"] = pd.to_datetime(promo_df["date"], format="%Y-%m-%d")

        # Handle 'hour' column processing
        if "hour" not in sales_df.columns:
            print(
                "Warning: 'hour' column not found in sales data. Merging on date only."
            )
            daily_promo_info = (
                promo_df.groupby("date")["isPromotion"].mean().reset_index()
            )
            merged_df = pd.merge(sales_df, daily_promo_info, on="date", how="left")
        else:
            try:
                # Extract starting hour from sales data
                sales_df["hour"] = sales_df["hour"].astype(str).str[:2].astype(int)
                promo_df["hour"] = promo_df["hour"].astype(int)
                print("Processed 'hour' columns for merging.")

                # Merge sales and promotion data
                merged_df = pd.merge(
                    sales_df, promo_df, on=["date", "hour"], how="left"
                )
                print("Sales and promotion data merged.")
            except KeyError as e:
                print(
                    f"Error processing/merging on ['date', 'hour']: Missing column {e}."
                )
                return
            except ValueError as e:
                print(f"Error converting 'hour' column: {e}. Check format in CSVs.")
                return

        # Fill any missing promotion values with 0
        merged_df["isPromotion"] = merged_df["isPromotion"].fillna(0)

        # Check if the sales column exists
        if sales_column not in merged_df.columns:
            print(f"Error: Sales column '{sales_column}' not found.")
            print(f"Available columns: {merged_df.columns.tolist()}")
            return

        # Aggregate sales by day and get promotion intensity
        print("Aggregating daily sales and promotion intensity...")
        daily_data = (
            merged_df.groupby("date")
            .agg(
                total_sales=(sales_column, "sum"),
                promo_intensity=("isPromotion", "mean"),
            )
            .reset_index()
        )

        # Create promotion type categories
        conditions = [
            (daily_data["promo_intensity"] == 0),
            (daily_data["promo_intensity"] == 0.2),
            (daily_data["promo_intensity"] == 0.5),
        ]
        choices = ["No Promotion", "20% Off", "50% Off"]
        daily_data["Promotion Type"] = np.select(
            conditions, choices, default="No Promotion"
        )

        # --- Plotting ---
        print("Generating plots...")
        print("Debug - Promotion Types:", daily_data["Promotion Type"].value_counts())
        print(
            "Debug - Unique promotion intensities:",
            daily_data["promo_intensity"].unique(),
        )

        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

        # Subplot 1: Sales Time Series (top left)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(
            daily_data["date"],
            daily_data["total_sales"],
            label="Daily Sales",
            color="royalblue",
            linewidth=2,
        )

        # Highlight promotion periods
        promo_colors = {"50% Off": "gold", "20% Off": "lightcoral"}
        legend_patches = []

        # Add sales line to legend patches
        legend_patches.append(
            plt.Line2D([0], [0], color="royalblue", linewidth=2, label="Daily Sales")
        )

        # Add promotion patches to legend
        for label, color in promo_colors.items():
            legend_patches.append(mpatches.Patch(color=color, alpha=0.3, label=label))
            intensity = 0.5 if label == "50% Off" else 0.2
            dates = daily_data[abs(daily_data["promo_intensity"] - intensity) < 0.01]
            if not dates.empty:
                for _, row in dates.iterrows():
                    ax1.axvspan(
                        row["date"],
                        row["date"] + pd.Timedelta(days=1),
                        color=color,
                        alpha=0.3,
                    )

        ax1.set_title("Daily Sales and Promotion Periods", fontsize=14)
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Total Daily Sales", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(handles=legend_patches, loc="upper right")

        # Subplot 2: Box Plot (top right)
        ax2 = fig.add_subplot(gs[1, :])
        sns.boxplot(
            data=daily_data,
            x="Promotion Type",
            y="total_sales",
            order=["No Promotion", "20% Off", "50% Off"],
            palette=["lightblue", "lightcoral", "gold"],
        )

        ax2.set_title("Sales Distribution by Promotion Type", fontsize=14)
        ax2.set_xlabel("Promotion Type", fontsize=12)
        ax2.set_ylabel("Total Daily Sales", fontsize=12)

        # Add mean sales values on top of boxes
        for i, ptype in enumerate(["No Promotion", "20% Off", "50% Off"]):
            mean_val = daily_data[daily_data["Promotion Type"] == ptype][
                "total_sales"
            ].mean()
            if not pd.isna(mean_val):  # Only add label if we have data
                ax2.text(
                    i,
                    mean_val,
                    f"Mean: {mean_val:.0f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        plt.tight_layout()
        print("Plots generated. Displaying windows...")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}. Check paths.")
    except KeyError as e:
        print(f"Error: Missing expected column - {e}. Verify CSV columns.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    print("Starting visualization script...")
    visualize_sales_and_promotions(SALES_FILE, PROMO_FILE, SALES_COLUMN_NAME)
    print("Script finished.")
