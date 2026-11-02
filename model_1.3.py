import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

# === STEP 1: Load Both CSVs ===
df = pd.read_csv('./NDC DATA/1990-2000/43057_Table_2_Daily_NDCQ-2025-03-081.csv')
df_2 = pd.read_csv('./NDC DATA/2000-2024/43057_Table_2_Daily_NDCQ-2025-03-087.csv')

# === STEP 2: Clean and Convert Columns ===
for data in [df, df_2]:
    data['MAX'] = pd.to_numeric(data['MAX'], errors='coerce')
    data['MIN'] = pd.to_numeric(data['MIN'], errors='coerce')
    data['SSH'] = pd.to_numeric(data['SSH'], errors='coerce')
    data.columns = data.columns.to_list()[1:] + ['Unnamed_Column']

# === STEP 3: Interpolate Required Columns ===
cols_to_interpolate = ['MN', 'YEAR', 'DT', 'RF', 'SSH', 'RA', 'SL', 'DS', 'MAX', 'MIN']
for data in [df, df_2]:
    for col in cols_to_interpolate:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].interpolate(method='linear')

# === STEP 4: Feature Engineering ===
for data in [df, df_2]:
    data.rename(columns=lambda x: x.strip().upper(), inplace=True)
    data['DATE'] = pd.to_datetime(dict(year=data['YEAR'], month=data['MN'], day=data['DT']), errors='coerce')
    data['DOY'] = data['DATE'].dt.dayofyear
    data['MAX_lag1'] = data['MAX'].shift(1)
    data['MIN_lag1'] = data['MIN'].shift(1)
    data['MAX_roll3'] = data['MAX'].rolling(window=3).mean().shift(1)
    data['MIN_roll3'] = data['MIN'].rolling(window=3).mean().shift(1)

# === STEP 5: Drop NaNs Introduced by Lags ===
df = df.dropna(subset=cols_to_interpolate + ['MAX_lag1', 'MIN_lag1', 'MAX_roll3', 'MIN_roll3', 'DOY'])
df_2 = df_2.dropna(subset=cols_to_interpolate + ['MAX_lag1', 'MIN_lag1', 'MAX_roll3', 'MIN_roll3', 'DOY'])

# === STEP 6: Combine the Two Datasets ===
df_combined = pd.concat([df, df_2], ignore_index=True)

# === STEP 7: Define Features and Targets ===
features = ['MN', 'YEAR', 'DT','RF','SSH', 'RA', 'SL', 'DS',
            'DOY', 'MAX_lag1', 'MIN_lag1', 'MAX_roll3', 'MIN_roll3']
target_max = 'MAX'
target_min = 'MIN'

# === STEP 8: Train-Test Split based on date ===
df_combined = df_combined.sort_values('DATE')
train_df = df_combined[df_combined['YEAR'] <= 2020]
test_df = df_combined[df_combined['YEAR'] > 2020]

X_train = train_df[features]
y_max_train = train_df[target_max]
y_min_train = train_df[target_min]

X_test = test_df[features]
y_max_test = test_df[target_max]
y_min_test = test_df[target_min]

# === STEP 9: Train XGBoost Regressors ===
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

model_max.fit(X_train, y_max_train, eval_set=[(X_test, y_max_test)], verbose=False)
model_min.fit(X_train, y_min_train, eval_set=[(X_test, y_min_test)], verbose=False)

# === STEP 10: Evaluation ===
def evaluate_model(y_true_train, y_pred_train, y_true_test, y_pred_test, label):
    print(f"=== {label} Temperature Prediction ===")
    print("Train RMSE:", root_mean_squared_error(y_true_train, y_pred_train))
    print("Train R² Score:", r2_score(y_true_train, y_pred_train))
    print("Test RMSE:", root_mean_squared_error(y_true_test, y_pred_test))
    print("Test R² Score:", r2_score(y_true_test, y_pred_test))

# MAX
evaluate_model(y_max_train, model_max.predict(X_train), y_max_test, model_max.predict(X_test), "MAX")
# MIN
evaluate_model(y_min_train, model_min.predict(X_train), y_min_test, model_min.predict(X_test), "MIN")

# === STEP 11: Feature Importance Visualization ===


# MAX temperature feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_max, importance_type='gain', title='MAX Temp - Feature Importance', show_values=False)
plt.grid()
plt.tight_layout()
plt.show()

# MIN temperature feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_min, importance_type='gain', title='MIN Temp - Feature Importance', show_values=False)
plt.grid()
plt.tight_layout()
plt.show()

# === STEP 12: Prediction vs True Plots ===
# First 100 samples comparison for better visibility
# === STEP 10.5: Make Predictions Again (Ensure these variables exist) ===

# MAX temperature predictions
y_max_train_pred = model_max.predict(X_train)
y_max_test_pred = model_max.predict(X_test)

# MIN temperature predictions
y_min_train_pred = model_min.predict(X_train)
y_min_test_pred = model_min.predict(X_test)

print(y_max_train_pred)
print(y_min_test_pred)

# MAX temperature
plt.figure(figsize=(12, 5))
plt.plot(y_max_test.values[:100], label='True MAX', marker='o')
plt.plot(y_max_test_pred[:100], label='Predicted MAX', marker='x')
plt.title('MAX Temperature Prediction (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# MIN temperature
plt.figure(figsize=(12, 5))
plt.plot(y_min_test.values[:100], label='True MIN', marker='o')
plt.plot(y_min_test_pred[:100], label='Predicted MIN', marker='x')
plt.title('MIN Temperature Prediction (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
