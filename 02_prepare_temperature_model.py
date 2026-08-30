import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

INPUT_FILE = "physics_output.csv"

# We train on temperature/humidity history, not CNN features.
# 24 hourly readings = previous 24 hours.
WINDOW_SIZE = 24

# Features used by the temperature/humidity model.
FEATURES = [
    "temperature_C",
    "humidity_RH",
    "degradation_rate",
    "cumulative_degradation_D",
    "RSL_physics_days",
]

TARGET = "delta_RSL_ML_target_days"

df = pd.read_csv(INPUT_FILE)
df = df.sort_values(["lot_id", "timestamp_hour"]).reset_index(drop=True)

if TARGET not in df.columns:
    raise ValueError(
        f"{TARGET} is missing. You need RSL_actual_days in the training CSV "
        "to create the ML correction target."
    )

X, y, groups = [], [], []

for lot_id, group in df.groupby("lot_id"):
    group = group.reset_index(drop=True)

    if len(group) <= WINDOW_SIZE:
        continue

    feature_values = group[FEATURES].astype(float).values
    target_values = group[TARGET].astype(float).values

    # Every sample uses the previous 24 readings to predict correction at time i.
    for i in range(WINDOW_SIZE, len(group)):
        X.append(feature_values[i - WINDOW_SIZE:i])
        y.append(target_values[i])
        groups.append(lot_id)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
groups = np.asarray(groups)

if len(X) == 0:
    raise ValueError("No sequences were created. Check that each lot has > 24 rows.")

# Split by lot so readings from the same physical lot never leak into
# both train and test sets.
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Temperature/humidity training data prepared.")
print("Features:", FEATURES)
print("Window:", WINDOW_SIZE, "readings")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
