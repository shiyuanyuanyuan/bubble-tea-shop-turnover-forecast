import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Define operating hours (inclusive)
OPERATING_START_HOUR = 10
OPERATING_END_HOUR = 22 # Includes the 10 PM hour

# Define staffing rules (ADJUSTED BASED ON FLAWED ANALYSIS)
BASELINE_STAFF = 2 # Adjusted from 3 to 2 based on flawed average (1.50)
PEAK_SALES_PERCENTILE = 75 # Use top 25% of predicted sales as peak
PEAK_EXTRA_STAFF = 1 # Peak staff will be BASELINE + EXTRA = 3

# Define file paths (assuming script runs from project root)
PREDICTIONS_FILE = 'data/prediction_results_v2.csv'
OUTPUT_FILE = 'data/staffing_suggestions.csv'

# --- Helper Function for Paths ---
# (Optional, assumes script is in root. Adjust if needed)
script_dir = os.path.dirname(os.path.abspath(__file__))
def data_path(filename):
    # If running from root, construct path to data subdir
    # If script is moved to data/, this needs adjustment
    data_dir = os.path.join(script_dir, 'data')
    # Check if filename already contains 'data/' prefix to avoid doubling
    if filename.startswith('data/') or filename.startswith('data\\\\'):
         return os.path.join(script_dir, filename)
    return os.path.join(data_dir, os.path.basename(filename))

# --- Main Logic ---
def generate_suggestions():
    print(f"Reading prediction data from: {PREDICTIONS_FILE}")
    try:
        predictions_df = pd.read_csv(PREDICTIONS_FILE)
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {PREDICTIONS_FILE}")
        print("Please ensure 'data/predict_sales.py' has been run successfully.")
        return
    except Exception as e:
        print(f"Error reading predictions file: {e}")
        return

    # Prepare data: Ensure correct types
    try:
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        # Convert Hour to integer for comparison
        predictions_df['Hour'] = predictions_df['Hour'].astype(int)
    except KeyError as e:
        print(f"Error: Missing expected column in predictions file: {e}")
        return
    except Exception as e:
        print(f"Error processing prediction data types: {e}")
        return


    # Filter for operating hours to calculate threshold correctly
    operating_hours_df = predictions_df[
        (predictions_df['Hour'] >= OPERATING_START_HOUR) &
        (predictions_df['Hour'] <= OPERATING_END_HOUR)
    ].copy()

    if operating_hours_df.empty:
        print("Warning: No data found within operating hours. Cannot calculate peak threshold.")
        peak_threshold = np.inf # Set threshold high if no data
    else:
        # Calculate the peak sales threshold based on the defined percentile
        peak_threshold = np.percentile(operating_hours_df['Predicted'], PEAK_SALES_PERCENTILE)
        print(f"Calculated peak sales threshold ({PEAK_SALES_PERCENTILE}th percentile): {peak_threshold:.2f}")

    # Define function to apply staffing rules
    def apply_staffing_rules(row):
        hour = row['Hour']
        predicted_sales = row['Predicted']

        # Rule 1: Outside operating hours
        if hour < OPERATING_START_HOUR or hour > OPERATING_END_HOUR:
            return 0
        # Rule 2: Peak hours
        elif predicted_sales >= peak_threshold:
            return BASELINE_STAFF + PEAK_EXTRA_STAFF
        # Rule 3: Baseline operating hours
        else:
            return BASELINE_STAFF

    # Apply the rules to generate the suggestion column
    print("Applying staffing rules...")
    predictions_df['SuggestedStaff'] = predictions_df.apply(apply_staffing_rules, axis=1)

    # Prepare final output DataFrame
    output_df = predictions_df[['Date', 'Hour', 'Predicted', 'SuggestedStaff']].copy()
    output_df.rename(columns={'Predicted': 'PredictedSales'}, inplace=True)
    # Format hour back to string if needed for consistency with original CSVs
    output_df['Hour'] = output_df['Hour'].astype(str).str.zfill(2)
    # Format Date to string YYYY-MM-DD
    output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')


    # Save the suggestions to CSV
    try:
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully generated staffing suggestions to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    generate_suggestions() 