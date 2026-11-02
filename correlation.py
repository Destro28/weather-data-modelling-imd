import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('./NDC DATA/1990-2000/43057_Table_2_Daily_NDCQ-2025-03-081.csv',delimiter=',',header=0,on_bad_lines="skip")
# If 'SSH' exists, drop it
# if 'SSH' in df.columns:
#     df.drop('SSH', axis=1, inplace=True)
if 'DZ' in df.columns:
    df.drop('DZ',axis=1,inplace=True)
if 'SN' in df.columns:
    df.drop('SN',axis=1,inplace=True)
if 'TH' in df.columns:
    df.drop('TH',axis=1,inplace=True)
if 'FG' in df.columns:
    df.drop('FG',axis=1,inplace=True)

df.columns = df.columns.to_list()[1:] + ['Unnamed_Column']
print(df.head())


# Debug: Check unique values of 'DT' before dropping NA
if 'DT' in df.columns:
    print("\nUnique values in 'DT':", df['DT'].unique())

# Drop rows with missing 'DT'
df = df.dropna(subset=['DT'])
print("\nAfter dropping NA in 'DT', number of rows:", len(df))

# Convert 'DT' to numeric (this column represents day)
df['MAX'] = pd.to_numeric(df['MAX'], errors='coerce')
df['MIN']=pd.to_numeric(df['MIN'],errors='coerce')
df['SSH']=pd.to_numeric(df['SSH'],errors='coerce')
# Ensure all columns are numeric (ignoring non-numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
# df=df['YEAR','MN','DT','MAX','MIN','RF','SSH','RA','SL','DS']
# Print to check if it only has numeric data
print(numeric_df.head())
correlation_matrix = numeric_df.corr()
print(correlation_matrix[['MAX', 'MIN']])

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
    
print(df['DT'].head(100))