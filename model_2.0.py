import numpy as np
import tensorflow as tf
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras import losses, metrics

import tensorflow as tf

def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())




# === STEP 1: Load CSV ===
df = pd.read_csv('./NDC DATA/1990-2000/43057_Table_2_Daily_NDCQ-2025-03-081.csv')
df.columns = df.columns.to_list()[1:] + ['Unnamed_Column']
df['MAX'] = pd.to_numeric(df['MAX'], errors='coerce')
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




# === Step 1: Scaling Features ===
scaler_X = MinMaxScaler()
scaler_y_max = MinMaxScaler()
scaler_y_min = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df_model[features])
y_max_scaled = scaler_y_max.fit_transform(df_model[[target_max]])
y_min_scaled = scaler_y_min.fit_transform(df_model[[target_min]])

# === Step 2: Create Sequences ===
def create_sequences(X, y, sequence_length=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

sequence_length = 7  # 7-day window
X_seq, y_max_seq = create_sequences(X_scaled, y_max_scaled)
_, y_min_seq = create_sequences(X_scaled, y_min_scaled)

# === Step 3: Train-Test Split ===
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_max_train, y_max_test = y_max_seq[:split], y_max_seq[split:]
y_min_train, y_min_test = y_min_seq[:split], y_min_seq[split:]

# === Step 4: Define LSTM Model ===
def build_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse',metrics=[ 
        metrics.RootMeanSquaredError(), 
        r2_score])
    return model

input_shape = (sequence_length, len(features))

# === Step 5: Train Models ===
model_max_lstm = build_lstm_model(input_shape)
model_min_lstm = build_lstm_model(input_shape)

history_max = model_max_lstm.fit(X_train, y_max_train, validation_data=(X_test, y_max_test), epochs=50, batch_size=16, verbose=1)
history_min = model_min_lstm.fit(X_train, y_min_train, validation_data=(X_test, y_min_test), epochs=50, batch_size=16, verbose=1)

# === Step 6: Predictions and Evaluation ===
y_max_pred_scaled = model_max_lstm.predict(X_test)
y_min_pred_scaled = model_min_lstm.predict(X_test)

y_max_pred = scaler_y_max.inverse_transform(y_max_pred_scaled)
y_min_pred = scaler_y_min.inverse_transform(y_min_pred_scaled)

y_max_true = scaler_y_max.inverse_transform(y_max_test)
y_min_true = scaler_y_min.inverse_transform(y_min_test)

print("=== LSTM MAX Temperature ===")
print("RMSE:", root_mean_squared_error(y_max_true, y_max_pred))
print("R² Score:", r2_score(y_max_true, y_max_pred))

print("\n=== LSTM MIN Temperature ===")
print("RMSE:", root_mean_squared_error(y_min_true, y_min_pred))
print("R² Score:", r2_score(y_min_true, y_min_pred))
