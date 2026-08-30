import pandas as pd
import numpy as np

INPUT_FILE = "synthetic_shelf_life_temperature_humidity.csv"
OUTPUT_FILE = "physics_output.csv"

# Initial shelf-life/degradation budget used by the synthetic dataset.
# Change these values when you have experimentally measured S0.
DEFAULT_S0_DAYS = {
    "Tomato": 10.0,
    "Banana": 8.0,
    "Mango": 12.0,
}

df = pd.read_csv(INPUT_FILE)

required = [
    "lot_id", "timestamp_hour", "temperature_C", "humidity_RH",
    "crop_type", "Q10", "T_ref_C", "r_ref_per_day"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.sort_values(["lot_id", "timestamp_hour"]).reset_index(drop=True)

# S0: initial shelf-life budget.
if "S0_days" in df.columns:
    df["S0_days"] = pd.to_numeric(df["S0_days"], errors="coerce")
else:
    df["S0_days"] = df["crop_type"].map(DEFAULT_S0_DAYS)

if df["S0_days"].isna().any():
    bad = df.loc[df["S0_days"].isna(), "crop_type"].unique()
    raise ValueError(f"No S0 defined for crop type(s): {bad}")

# Measurement interval in days.
if "delta_t_days" in df.columns:
    df["delta_t_days"] = pd.to_numeric(df["delta_t_days"], errors="coerce")
else:
    # Assumes hourly measurements.
    df["delta_t_days"] = 1.0 / 24.0

# Q10 model from your formula:
# r(T) = r_ref * Q10^((T - T_ref)/10)
df["degradation_rate"] = (
    df["r_ref_per_day"]
    * np.power(df["Q10"], (df["temperature_C"] - df["T_ref_C"]) / 10.0)
)

# Degradation accumulated during each measurement interval.
df["degradation_increment"] = (
    df["degradation_rate"] * df["delta_t_days"]
)

# D = sum(r(T_i) * delta_t_i)
df["cumulative_degradation_D"] = (
    df.groupby("lot_id")["degradation_increment"].cumsum()
)

# RSL_physics = S0 - D
df["RSL_physics_days"] = (
    df["S0_days"] - df["cumulative_degradation_D"]
).clip(lower=0.0)

# Keep the real/experimental target if it exists.
# For synthetic data this is supplied in the CSV.
if "RSL_actual_days" in df.columns:
    df["RSL_actual_days"] = pd.to_numeric(
        df["RSL_actual_days"], errors="coerce"
    )
    df["delta_RSL_ML_target_days"] = (
        df["RSL_actual_days"] - df["RSL_physics_days"]
    )

df.to_csv(OUTPUT_FILE, index=False)

print("Physics model completed.")
print(f"Rows: {len(df)}")
print(f"Saved: {OUTPUT_FILE}")
print("\nLatest result for each lot:")
latest = df.groupby("lot_id").tail(1)
print(
    latest[
        ["lot_id", "crop_type", "temperature_C", "humidity_RH",
         "cumulative_degradation_D", "RSL_physics_days"]
    ].head(10).to_string(index=False)
)
