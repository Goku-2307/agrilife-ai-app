SHELF-LIFE TEMPERATURE/HUMIDITY MODEL
======================================

This version does NOT train a CNN.

Your CNN/ripeness model is separate.

This model uses:
1. ESP32 temperature readings
2. ESP32 humidity readings
3. Q10 physics model
4. XGBoost correction model

Formula:
    r(T) = r_ref * Q10^((T - T_ref)/10)

    D = sum(r(T_i) * delta_t_i)

    RSL_physics = S0 - D

    delta_RSL_ML = RSL_actual - RSL_physics

    RSL_final = RSL_physics + delta_RSL_ML

TRAINING
--------

1. Put this folder's files together with:
   synthetic_shelf_life_temperature_humidity.csv

2. Install:
   pip install pandas numpy scikit-learn xgboost joblib

3. Run:
   python 01_physics_temperature_model.py

4. Run:
   python 02_prepare_temperature_model.py

5. Run:
   python 03_train_temperature_xgboost.py

6. Run:
   python 04_predict_shelf_life.py

IMPORTANT
---------
The synthetic CSV is only for software testing.
For a real model, collect experimentally measured RSL_actual_days.

The current model uses the previous 24 readings.
If ESP32 sends one reading every 5 minutes, change WINDOW_SIZE to 288
(24 hours x 12 readings/hour).

CNN integration is intentionally NOT included here.
