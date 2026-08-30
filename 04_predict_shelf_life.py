import pandas as pd
import numpy as np
import joblib

INPUT_FILE = "physics_output.csv"
OUTPUT_FILE = "final_shelf_life_predictions.csv"

WINDOW_SIZE = 24

FEATURES = [
    "temperature_C",
    "humidity_RH",
    "degradation_rate",
    "cumulative_degradation_D",
    "RSL_physics_days",
]

model = joblib.load(
    "models/temperature_humidity_xgboost.pkl"
)
scaler = joblib.load(
    "models/temperature_humidity_scaler.pkl"
)

df = pd.read_csv(INPUT_FILE)
df = df.sort_values(["lot_id", "timestamp_hour"]).reset_index(drop=True)

results = []

for lot_id, group in df.groupby("lot_id"):
    group = group.reset_index(drop=True)

    if len(group) < WINDOW_SIZE:
        continue

    # Latest 24 readings from this lot.
    latest = group[FEATURES].astype(float).iloc[-WINDOW_SIZE:].values

    flat = latest.reshape(1, -1)
    flat_scaled = scaler.transform(flat)

    # XGBoost predicts the correction:
    # delta_RSL_ML = RSL_actual - RSL_physics
    delta_rsl = float(model.predict(flat_scaled)[0])

    rsl_physics = float(group["RSL_physics_days"].iloc[-1])

    # Final formula from your screenshot:
    # RSL_final = RSL_physics + delta_RSL_ML
    rsl_final = max(0.0, rsl_physics + delta_rsl)

    results.append({
        "lot_id": lot_id,
        "timestamp_hour": group["timestamp_hour"].iloc[-1],
        "temperature_C": group["temperature_C"].iloc[-1],
        "humidity_RH": group["humidity_RH"].iloc[-1],
        "RSL_physics_days": rsl_physics,
        "delta_RSL_ML_days": delta_rsl,
        "RSL_final_days": rsl_final,
    })

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print("\nFINAL SHELF-LIFE RESULTS")
print(result_df.to_string(index=False))
print(f"\nSaved: {OUTPUT_FILE}")
