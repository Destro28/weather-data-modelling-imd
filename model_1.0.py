import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

# === STEP 1: Load CSV ===
df = pd.read_csv('./NDC DATA/1990-2000/43057_Table_2_Daily_NDCQ-2025-03-081.csv')
df['MAX'] = pd.to_numeric(df['MAX'], errors='coerce')
df.columns = df.columns.to_list()[1:] + ['Unnamed_Column']
df['MIN']=pd.to_numeric(df['MIN'],errors='coerce')
df['SSH']=pd.to_numeric(df['SSH'],errors='coerce')

# === STEP 2: Interpolate Required Columns ===
# YEAR,MN,DT,RF,SSH,RA,SL,DS
cols_to_interpolate = ['MN', 'YEAR', 'DT','RF', 'SSH', 'RA', 'SL', 'DS', 'MAX', 'MIN']

for col in cols_to_interpolate:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].interpolate(method='linear')

# === STEP 3: Drop rows where any selected features or targets are still NaN ===
df_model = df[cols_to_interpolate].dropna()

# === STEP 4: Define features and targets ===
features = ['MN', 'YEAR', 'DT','RF','SSH', 'RA', 'SL', 'DS']
target_max = 'MAX'
target_min = 'MIN'

X = df_model[features]
y_max = df_model[target_max]
y_min = df_model[target_min]

# === STEP 5: Train-Test Split ===
X_train, X_test, y_max_train, y_max_test = train_test_split(X, y_max, test_size=0.2, random_state=42)
_, _, y_min_train, y_min_test = train_test_split(X, y_min, test_size=0.2, random_state=42)

# === STEP 6: Train XGBoost Regressors ===
model_max = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='rmse', early_stopping_rounds=20,
    verbose=True)
model_min = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='rmse', early_stopping_rounds=20,
    verbose=True)

model_max.fit(
    X_train, y_max_train,
    eval_set=[(X_test, y_max_test)],
   
   
)

model_min.fit(
    X_train, y_min_train,
    eval_set=[(X_test, y_min_test)]
)


# === STEP 7: Predictions & Evaluation ===
y_max_pred = model_max.predict(X_test,)
y_min_pred = model_min.predict(X_test)

print("=== MAX Temperature Prediction ===")
print("RMSE:", root_mean_squared_error(y_max_test, y_max_pred))
print("R² Score:", r2_score(y_max_test, y_max_pred))

print("\n=== MIN Temperature Prediction ===")
print("RMSE:", root_mean_squared_error(y_min_test, y_min_pred))
print("R² Score:", r2_score(y_min_test, y_min_pred))

# === STEP 8: Feature Importance Visualization ===
xgb.plot_importance(model_max, importance_type='gain', title="MAX Temp - Feature Importance")
plt.show()

xgb.plot_importance(model_min, importance_type='gain', title="MIN Temp - Feature Importance")
plt.show()
