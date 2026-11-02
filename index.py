import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

df=pd.read_csv("./NDC DATA/1990-2000/42851_Table_2_Daily_NDCQ-2025-03-081.csv")
df['MIN']=pd.to_numeric(df['MIN'],errors='coerce')
df['AW']=pd.to_numeric(df['AW'],errors='coerce')
df['RF']=pd.to_numeric(df['RF'],errors='coerce')
df['EVP']=pd.to_numeric(df['EVP'],errors='coerce')
df['DUR']=pd.to_numeric(df['DUR'],errors='coerce')

df = df.rename(columns={'YEAR': 'year', 'MN': 'month', 'DT': 'day'})
df['Date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')

# Check for any invalid date conversions
invalid_dates = df[df['Date'].isna()]
if not invalid_dates.empty:
    print("Invalid dates detected:")
    print(invalid_dates)

# Remove invalid dates
df = df.dropna(subset=['Date'])
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['MAX'], label='Max Temperature', color='red')
plt.plot(df['Date'], df['MIN'], label='Min Temperature', color='blue')
# plt.plot(df['Date'], df['RF'], label='Rainfall', color='green')

plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Temperature Trends')
plt.legend()
plt.grid(True)
plt.show()

df['MAX_rolling'] = df['MAX'].rolling(window=30).mean()
df['MIN_rolling'] = df['MIN'].rolling(window=30).mean()

plt.figure(figsize=(12,6))
plt.plot(df['MAX'], label='Max Temperature', alpha=0.5)
plt.plot(df['MAX_rolling'], label='30-Day Rolling Avg', color='red')
plt.legend()
plt.show()

# decompose time series data,interpret results
result = seasonal_decompose(df['MAX'], model='additive', period=30)
result.plot()
plt.show()

# correlation analysis
correlation_matrix = df[['MAX', 'MIN', 'RF', 'EVP', 'AW', 'DUR']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # scatter plot(s)
# sns.pairplot(df[['MAX', 'MIN', 'RF', 'EVP', 'AW']])
# plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()