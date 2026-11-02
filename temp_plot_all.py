import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs('./visualised data', exist_ok=True)

# Read CSV file

files=[42867,42871,42921,42925,42931,42933,42937,42939,42943,42946,43001,43003,43005,43009,43011,43012,43014,43017,43018,43021,43029,43057,43058,43060,43062,43063,43069,43071,43075,43076,43089,43109,43110,43111,43113,43117,43153,43157,43158,43193]
for i in files:
    df = pd.read_csv(f'./NDC DATA/1990-2000/{i}_Table_2_Daily_NDCQ-2025-03-081.csv')

    # Debug: print initial columns and a few rows
    print("Initial Columns:", df.columns.tolist())
    print("Initial Data (first 5 rows):")
    print(df.head())

    # If 'SSH' exists, drop it
    if 'SSH' in df.columns:
        df.drop('SSH', axis=1, inplace=True)

    # Debug: Check unique values of 'DT' before dropping NA
    if 'DT' in df.columns:
        print("\nUnique values in 'DT':", df['DT'].unique())

    # Drop rows with missing 'DT'
    df = df.dropna(subset=['DT'])
    print("\nAfter dropping NA in 'DT', number of rows:", len(df))

    # Convert 'DT' to numeric (this column represents day)
    df['DT'] = pd.to_numeric(df['DT'], errors='coerce')

    # Verify that required columns exist (adjust column names if necessary)
    expected_cols = ['INDEX', 'YEAR', 'MN', 'DT', 'MAX', 'MIN', 'AW', 'RF', 'EVP', 'DUR']
    missing_cols = [col for col in expected_cols if col not in df.columns] #
    if missing_cols:
        print("Warning: Missing expected columns:", missing_cols)

    # Debug: print sample of expected columns
    print("\nData sample before renaming:")
    print(df[expected_cols].head())

    # Rename columns to standard names
    df = df.rename(columns={'INDEX': 'year', 'YEAR': 'month', 'MN': 'day'})

    # Clean year, month, and day: strip spaces and convert to numeric
    df['year'] = pd.to_numeric(df['year'].astype(str).str.strip(), errors='coerce')
    df['month'] = pd.to_numeric(df['month'].astype(str).str.strip(), errors='coerce')
    df['day'] = pd.to_numeric(df['day'].astype(str).str.strip(), errors='coerce')

    # Debug: show a few rows of the date components
    print("\nDate components sample:")
    print(df[['year', 'month', 'day']].head())

    # Filter rows with month and day in valid ranges
    df = df[(df['month'].between(1, 12)) & (df['day'].between(1, 31))]
    # print("\nAfter filtering for valid month/day, rows remaining:", len(df))

    # Create a Date column
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    # print("\nNumber of invalid dates (NaT):", df['Date'].isnull().sum())

    # Debug: see a sample of the new Date column with components
    print("\nDate column sample:")
    print(df[['year', 'month', 'day', 'Date']].head(10))

    # Drop rows with missing Date or MAX temperature values
    df = df.dropna(subset=['Date', 'MAX'])
    print("\nRows remaining after dropping missing Date/MAX:", len(df))

    # Finally, if data remains, plot the data
    if len(df) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['DT'], label="Max Temperature", color="red")
        plt.xlabel("Date")
        plt.ylabel("Max Temperature")
        plt.title("Daily Max Temperature Over Time")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'./visualised data/{i}_max_temperature_1990_2000.png')
        
    else:
        print("\nNo data available after cleaning. Check the CSV file or cleaning steps.")
    # To plot the daily minimum temperature over time
    if len(df) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['MAX'], label="Max Temperature", color="red")
        plt.xlabel("Date")
        plt.ylabel("Min Temperature")
        plt.title("Daily Min Temperature Over Time")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'./visualised data/{i}_min_temperature_1990_2000.png')
    else:
        print("\nNo data available after cleaning. Check the CSV file or cleaning steps.")

plt.close()