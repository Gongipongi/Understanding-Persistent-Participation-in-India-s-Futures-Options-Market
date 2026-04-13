import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ==================================================
# CONFIGURATION
# ==================================================
DATA_FOLDER = "data_folder"  # Relative path - folder should be in same directory as this script
OUTPUT_FILE = "compiled_analysis_report.xlsx"

# ==================================================
# ANALYZE ONE CSV (CORE LOGIC)
# ==================================================
def analyze_single_dataframe(df_raw, instrument_name, oi_floor=2000):
    """
    Analyzes a single futures CSV file.
    Returns cleaned, contract-separated dataframe with all signals.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # Validate required columns
    if "Date" not in df.columns:
        return pd.DataFrame()

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")

    # Contract identification
    if "Expiry" in df.columns:
        df["Contract_ID"] = df["Expiry"]
    elif "Expiry_Date" in df.columns:
        df["Contract_ID"] = df["Expiry_Date"]
    else:
        df["Contract_ID"] = df.index.astype(str)

    # Numeric conversion
    numeric_cols = ["Open", "Close", "No. of contracts"]
    if "Open Int" in df.columns:
        numeric_cols.append("Open Int")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    required_cols = ["Open", "Close", "No. of contracts", "Date"]
    df = df.dropna(subset=required_cols)
    df = df.sort_values(["Date", "Contract_ID"])

    # Process each contract separately
    all_contracts = []

    for cid in df["Contract_ID"].unique():
        cdf = df[df["Contract_ID"] == cid].copy()
        cdf = cdf.sort_values("Date").reset_index(drop=True)

        # Skip contracts with insufficient data
        if len(cdf) <= 6:
            continue

        # Remove rollover noise (first 3 and last 3 days)
        cdf = cdf.iloc[3:-3].reset_index(drop=True)

        # OI floor filter (if OI exists)
        if "Open Int" in cdf.columns:
            cdf = cdf[cdf["Open Int"].shift(1) >= oi_floor]

        if len(cdf) < 2:
            continue

        # === CORE SIGNALS ===
        # Price signals
        cdf["Daily_Change"] = cdf["Close"] - cdf["Open"]
        cdf["Is_Loss"] = cdf["Daily_Change"] < 0
        cdf["Is_Gain"] = cdf["Daily_Change"] > 0

        # Volume signals
        cdf["Volume_Pct_Change"] = cdf["No. of contracts"].pct_change() * 100

        # OI signals (if available)
        if "Open Int" in cdf.columns:
            cdf["OI_Change"] = cdf["Open Int"].diff()
            cdf["OI_20D_Avg"] = cdf["Open Int"].rolling(20, min_periods=5).mean()
            cdf["OI_Normalized_Change"] = cdf["OI_Change"] / cdf["OI_20D_Avg"]

            # Next-day OI response
            cdf["Next_Day_OI"] = cdf["Open Int"].shift(-1)
            cdf["Next_Day_OI_Change"] = cdf["Next_Day_OI"] - cdf["Open Int"]
            cdf["Next_Day_OI_Normalized_Change"] = (
                cdf["Next_Day_OI_Change"] / cdf["OI_20D_Avg"]
            )

        # Next-day volume response
        cdf["Next_Day_Volume"] = cdf["No. of contracts"].shift(-1)
        cdf["Next_Day_Volume_Pct_Change"] = (
            (cdf["Next_Day_Volume"] - cdf["No. of contracts"]) 
            / cdf["No. of contracts"] * 100
        )

        # Add instrument identifier
        cdf["Instrument"] = instrument_name
        cdf["Contract_ID"] = cid

        all_contracts.append(cdf)

    if not all_contracts:
        return pd.DataFrame()

    return pd.concat(all_contracts, ignore_index=True)


# ==================================================
# ANALYZE ENTIRE FOLDER
# ==================================================
def analyze_data_folder(data_folder_path=DATA_FOLDER, oi_floor=2000):
    """
    Analyzes all CSV files in the data folder.
    Returns aggregated dataframe with all instruments and contracts.
    """
    if not os.path.exists(data_folder_path):
        raise FileNotFoundError(f"Data folder not found: {data_folder_path}")

    all_data = []
    files_processed = 0
    files_failed = 0

    print(f"Starting analysis of folder: {data_folder_path}")
    print("=" * 60)

    for filename in os.listdir(data_folder_path):
        if not filename.lower().endswith(".csv"):
            continue

        file_path = os.path.join(data_folder_path, filename)
        instrument_name = filename.replace(".csv", "")

        try:
            df_raw = pd.read_csv(file_path)
            df_analyzed = analyze_single_dataframe(df_raw, instrument_name, oi_floor)

            if not df_analyzed.empty:
                all_data.append(df_analyzed)
                files_processed += 1
                print(f"✓ Processed: {filename} ({len(df_analyzed)} rows)")
            else:
                files_failed += 1
                print(f"✗ Skipped: {filename} (insufficient data)")

        except Exception as e:
            files_failed += 1
            print(f"✗ Failed: {filename} - {str(e)}")

    print("=" * 60)
    print(f"Summary: {files_processed} files processed, {files_failed} files skipped/failed")

    if not all_data:
        raise ValueError("No valid data found in any CSV files")

    # Combine all instruments and contracts
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


# ==================================================
# GENERATE COMPILED REPORT
# ==================================================
def generate_compiled_report(df_combined):
    """
    Generates aggregated metrics grouped by Instrument and Contract_ID.
    Returns a summary dataframe suitable for Excel export.
    """
    # Calculate percentiles for OI changes (global)
    if "Next_Day_OI_Normalized_Change" in df_combined.columns:
        df_combined["Next_Day_OI_Pctl"] = (
            df_combined["Next_Day_OI_Normalized_Change"].rank(pct=True) * 100
        )

    # Group by Instrument and Contract_ID
    grouped = df_combined.groupby(["Instrument", "Contract_ID"])

    # Calculate aggregated metrics
    report_data = []

    for (instrument, contract_id), group in grouped:
        loss_days = group[group["Is_Loss"]]
        gain_days = group[group["Is_Gain"]]

        metrics = {
            "Instrument": instrument,
            "Contract_ID": contract_id,
            "Total_Days": len(group),
            "Loss_Days": len(loss_days),
            "Gain_Days": len(gain_days),
            
            # After LOSS days
            "Avg_NextDay_Volume_Change_AfterLoss": loss_days["Next_Day_Volume_Pct_Change"].mean(),
            "Avg_NextDay_OI_Normalized_AfterLoss": loss_days["Next_Day_OI_Normalized_Change"].mean() if "Next_Day_OI_Normalized_Change" in loss_days.columns else np.nan,
            "Avg_OI_Percentile_AfterLoss": loss_days["Next_Day_OI_Pctl"].mean() if "Next_Day_OI_Pctl" in loss_days.columns else np.nan,
            "Pct_OI_Increase_AfterLoss": (loss_days["Next_Day_OI_Normalized_Change"] > 0).mean() * 100 if "Next_Day_OI_Normalized_Change" in loss_days.columns else np.nan,
            
            # After GAIN days
            "Avg_NextDay_Volume_Change_AfterGain": gain_days["Next_Day_Volume_Pct_Change"].mean(),
            "Avg_NextDay_OI_Normalized_AfterGain": gain_days["Next_Day_OI_Normalized_Change"].mean() if "Next_Day_OI_Normalized_Change" in gain_days.columns else np.nan,
            "Avg_OI_Percentile_AfterGain": gain_days["Next_Day_OI_Pctl"].mean() if "Next_Day_OI_Pctl" in gain_days.columns else np.nan,
            "Pct_OI_Increase_AfterGain": (gain_days["Next_Day_OI_Normalized_Change"] > 0).mean() * 100 if "Next_Day_OI_Normalized_Change" in gain_days.columns else np.nan,
        }

        report_data.append(metrics)

    df_report = pd.DataFrame(report_data)
    
    # Round numeric columns for readability
    numeric_cols = df_report.select_dtypes(include=[np.number]).columns
    df_report[numeric_cols] = df_report[numeric_cols].round(4)

    return df_report


# ==================================================
# MAIN EXECUTION FUNCTION
# ==================================================
def run_full_analysis(output_filename=OUTPUT_FILE):
    """
    Complete pipeline: analyze all files and export to Excel.
    """
    print("\n" + "=" * 60)
    print("FUTURES MARKET MICROSTRUCTURE ANALYSIS")
    print("=" * 60 + "\n")

    # Step 1: Analyze all files
    df_combined = analyze_data_folder()
    print(f"\nTotal rows analyzed: {len(df_combined)}")
    print(f"Unique instruments: {df_combined['Instrument'].nunique()}")
    print(f"Unique contracts: {len(df_combined.groupby(['Instrument', 'Contract_ID']))}")

    # Step 2: Generate compiled report
    print("\nGenerating compiled report...")
    df_report = generate_compiled_report(df_combined)

    # Step 3: Export to Excel
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_report.to_excel(writer, sheet_name='Compiled_Analysis', index=False)
        
    print(f"\n✓ Analysis complete!")
    print(f"✓ Report saved to: {output_filename}")
    print(f"✓ Contains {len(df_report)} contract-level summaries")
    print("=" * 60 + "\n")

    return df_report


# ==================================================
# RUN THE ANALYSIS
# ==================================================
if __name__ == "__main__":
    report = run_full_analysis()