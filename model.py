import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('./NDC DATA/1990-2000/43057_Table_2_Daily_NDCQ-2025-03-081.csv')

df.columns = df.columns.to_list()[1:] + ['Unnamed_Column']
df['MAX'] = pd.to_numeric(df['MAX'], errors='coerce')
df['MIN']=pd.to_numeric(df['MIN'],errors='coerce')
df['SSH']=pd.to_numeric(df['SSH'],errors='coerce')
# Drop unneccessarry rows

if 'DZ' in df.columns:
    df.drop('DZ',axis=1,inplace=True)
if 'SN' in df.columns:
    df.drop('SN',axis=1,inplace=True)
if 'TH' in df.columns:
    df.drop('TH',axis=1,inplace=True)
if 'FG' in df.columns:
    df.drop('FG',axis=1,inplace=True)
print(df.head())
# df.interpolate(method='polynomial', order=3, inplace=True)
# GPT stuff

# Assuming df contains your data

df['Linear_Interpolated_RF'] = df['RF'].interpolate(method='linear')
df['Spline_Interpolated_RF'] = df['RF'].interpolate(method='spline', order=3)

df['Linear_Interpolated_RA'] = df['RA'].interpolate(method='linear')
df['Spline_Interpolated_RA'] = df['RA'].interpolate(method='spline', order=3)

df['Linear_Interpolated_SL'] = df['SL'].interpolate(method='linear')
df['Spline_Interpolated_SL'] = df['SL'].interpolate(method='spline', order=3)

df['Linear_Interpolated_GA'] = df['GA'].interpolate(method='linear')
df['Spline_Interpolated_GA'] = df['GA'].interpolate(method='spline', order=3)

plt.figure(figsize=(12, 6))

# Plot original data
plt.plot(df.index, df['MAX'], 'o', label='Original Data (With Missing)', color='blue', alpha=0.4)

# Plot Linear Interpolation
plt.plot(df.index, df['Linear_Interpolated_GA'], label='Linear Interpolation', color='green')

# Plot Spline Interpolation
plt.plot(df.index, df['Spline_Interpolated_GA'], label='Spline Interpolation', color='red', linestyle='--')



plt.xlabel('Time (Days)')
plt.ylabel('Maximum Temperature')
plt.legend(loc='best')
plt.title('Interpolation Methods Comparison')
plt.grid(True)
plt.show()

# Residual plot
df['Residuals'] = df['MAX'] - df['MAX'].interpolate(method='linear')
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['Residuals'], color='purple', alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals After Linear Interpolation')
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.show()

# Correlation matrix after intrapolation


df['MAX'] = pd.to_numeric(df['MAX'], errors='coerce')
df['MIN']=pd.to_numeric(df['MIN'],errors='coerce')
# Ensure all columns are numeric (ignoring non-numeric columns)
numeric_df = df.select_dtypes(include=[np.number])

# Print to check if it only has numeric data
print(numeric_df.head())
correlation_matrix = numeric_df.corr()
print(correlation_matrix[['MAX', 'MIN']])
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
    

