import pandas as pd
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt


# === STEP 1: Load Both CSVs ===
df = pd.read_csv('./NDC DATA/1990-2000/43057_Table_2_Daily_NDCQ-2025-03-081.csv')
df_2 = pd.read_csv('./NDC DATA/2000-2024/43057_Table_2_Daily_NDCQ-2025-03-087.csv')

# === Convert Relevant Columns to Numeric ===
for data in [df, df_2]:
    data['MAX'] = pd.to_numeric(data['MAX'], errors='coerce')
    data['MIN'] = pd.to_numeric(data['MIN'], errors='coerce')
    data['SSH'] = pd.to_numeric(data['SSH'], errors='coerce')
    data.columns = data.columns.to_list()[1:] + ['Unnamed_Column']  # Drop unnecessary first column if needed

# === STEP 2: Interpolate Required Columns ===
cols_to_interpolate = ['MN', 'YEAR', 'DT', 'RF', 'SSH', 'RA', 'SL', 'DS', 'MAX', 'MIN']
for data in [df, df_2]:
    for col in cols_to_interpolate:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].interpolate(method='linear')

# === STEP 3: Additional Feature Engineering ===
for data in [df, df_2]:
    data.rename(columns=lambda x: x.strip().upper(), inplace=True)
    data['YEAR'] = pd.to_numeric(data['YEAR'], errors='coerce')
    data['MN'] = pd.to_numeric(data['MN'], errors='coerce')
    data['DT'] = pd.to_numeric(data['DT'], errors='coerce')
    data['DATE'] = pd.to_datetime(dict(year=data['YEAR'], month=data['MN'], day=data['DT']), errors='coerce')
    data['DOY'] = data['DATE'].dt.dayofyear
    data['MAX_lag1'] = data['MAX'].shift(1)
    data['MIN_lag1'] = data['MIN'].shift(1)
    data['MAX_roll3'] = data['MAX'].rolling(window=3).mean().shift(1)
    data['MIN_roll3'] = data['MIN'].rolling(window=3).mean().shift(1)

# === STEP 4: Drop NaNs Introduced by Lags ===
df = df.dropna(subset=cols_to_interpolate + ['MAX_lag1', 'MIN_lag1', 'MAX_roll3', 'MIN_roll3', 'DOY'])
df_2 = df_2.dropna(subset=cols_to_interpolate + ['MAX_lag1', 'MIN_lag1', 'MAX_roll3', 'MIN_roll3', 'DOY'])

# === STEP 5: Combine the Two Datasets ===
df_combined = pd.concat([df, df_2], ignore_index=True)
print("Combined dataset shape:", df_combined.shape)

# === STEP 6: Define Features and Targets ===
features = ['MN', 'YEAR', 'DT','RF','SSH', 'RA', 'SL', 'DS',
            'DOY', 'MAX_lag1', 'MIN_lag1', 'MAX_roll3', 'MIN_roll3']
target_max = 'MAX'
target_min = 'MIN'

X = df_combined[features]
y_max = df_combined[target_max]
y_min = df_combined[target_min]


# === STEP 5: Train-Test Split ===
X_train, X_test, y_max_train, y_max_test = train_test_split(X, y_max, test_size=0.2, random_state=42)
_, _, y_min_train, y_min_test = train_test_split(X, y_min, test_size=0.2, random_state=42)

# === STEP 6: Train XGBoost Regressors with tuned hyperparameters ===
xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'rmse',
    'early_stopping_rounds': 20
}

model_max = xgb.XGBRegressor(**xgb_params)
model_min = xgb.XGBRegressor(**xgb_params)

model_max.fit(X_train, y_max_train, eval_set=[(X_test, y_max_test)], verbose=True)
model_min.fit(X_train, y_min_train, eval_set=[(X_test, y_min_test)], verbose=True)

# === STEP 7: Predictions & Evaluation ===

# === STEP 7: Predictions & Evaluation ===

# MAX temp
y_max_train_pred = model_max.predict(X_train)
y_max_test_pred = model_max.predict(X_test)

print("=== MAX Temperature Prediction ===")
print("Train RMSE:", root_mean_squared_error(y_max_train, y_max_train_pred))
print("Train R² Score:", r2_score(y_max_train, y_max_train_pred))
print("Test RMSE:", root_mean_squared_error(y_max_test, y_max_test_pred))
print("Test R² Score:", r2_score(y_max_test, y_max_test_pred))

# MIN temp
y_min_train_pred = model_min.predict(X_train)
y_min_test_pred = model_min.predict(X_test)

print("\n=== MIN Temperature Prediction ===")
print("Train RMSE:", root_mean_squared_error(y_min_train, y_min_train_pred))
print("Train R² Score:", r2_score(y_min_train, y_min_train_pred))
print("Test RMSE:", root_mean_squared_error(y_min_test, y_min_test_pred))
print("Test R² Score:", r2_score(y_min_test, y_min_test_pred))

# === STEP 8: Feature Importance Visualization ===
xgb.plot_importance(model_max, importance_type='gain', title="MAX Temp - Feature Importance")
plt.show()

xgb.plot_importance(model_min, importance_type='gain', title="MIN Temp - Feature Importance")
plt.show()

# === STEP 9: Prediction Plotting ===
plt.figure(figsize=(12, 5))
plt.plot(y_max_test.values[:100], label='True MAX', marker='o')
plt.plot(y_max_test_pred[:100], label='Predicted MAX', marker='x')
plt.title('MAX Temp Prediction (First 100 Samples)')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_min_test.values[:100], label='True MIN', marker='o')
plt.plot(y_min_test_pred[:100], label='Predicted MIN', marker='x')
plt.title('MIN Temp Prediction (First 100 Samples)')
plt.legend()
plt.grid()
plt.show()