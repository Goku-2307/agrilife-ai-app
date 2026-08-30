import os
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit(
        "XGBoost is not installed. Run:\n"
        "pip install xgboost"
    )

WINDOW_SIZE = 24
N_FEATURES = 5

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# XGBoost expects 2-D data.
# Flatten the previous 24 readings:
# 24 readings x 5 features = 120 inputs.
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Standardization is not strictly required for XGBoost, but keeping the
# preprocessing explicit makes the pipeline easy to reuse with other models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nTemperature/Humidity ML model trained.")
print(f"MAE  : {mae:.4f} days")
print(f"RMSE : {rmse:.4f} days")
print(f"R2   : {r2:.4f}")

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/temperature_humidity_xgboost.pkl")
joblib.dump(scaler, "models/temperature_humidity_scaler.pkl")

print("\nSaved:")
print("models/temperature_humidity_xgboost.pkl")
print("models/temperature_humidity_scaler.pkl")
