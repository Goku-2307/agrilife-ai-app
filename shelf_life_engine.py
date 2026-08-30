import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any, Optional

# Predefined crop parameters database
CROP_DATABASE: Dict[str, Dict[str, Any]] = {
    "Tomato": {
        "display_name": "Tomato",
        "S0_days": 10.0,
        "Q10": 2.0,
        "T_ref_C": 25.0,
        "r_ref_per_day": 1.0,
        "RH_opt_min": 85.0,
        "RH_opt_max": 95.0,
        "T_opt_min": 12.0,
        "T_opt_max": 20.0,
        "icon": "🍅",
        "category": "Vegetable"
    },
    "Banana": {
        "display_name": "Banana",
        "S0_days": 8.0,
        "Q10": 2.2,
        "T_ref_C": 25.0,
        "r_ref_per_day": 1.0,
        "RH_opt_min": 90.0,
        "RH_opt_max": 95.0,
        "T_opt_min": 13.0,
        "T_opt_max": 15.0,
        "icon": "🍌",
        "category": "Fruit"
    },
    "Apple": {
        "display_name": "Apple",
        "S0_days": 25.0,
        "Q10": 1.8,
        "T_ref_C": 20.0,
        "r_ref_per_day": 0.8,
        "RH_opt_min": 90.0,
        "RH_opt_max": 95.0,
        "T_opt_min": 1.0,
        "T_opt_max": 4.0,
        "icon": "🍎",
        "category": "Fruit"
    },
    "Mango": {
        "display_name": "Mango",
        "S0_days": 12.0,
        "Q10": 2.1,
        "T_ref_C": 25.0,
        "r_ref_per_day": 1.0,
        "RH_opt_min": 85.0,
        "RH_opt_max": 90.0,
        "T_opt_min": 12.0,
        "T_opt_max": 14.0,
        "icon": "🥭",
        "category": "Fruit"
    }
}

WINDOW_SIZE = 24
FEATURES = [
    "temperature_C",
    "humidity_RH",
    "degradation_rate",
    "cumulative_degradation_D",
    "RSL_physics_days",
]


class ShelfLifeEngine:
    """
    Hybrid Physics (Q10) + Machine Learning (XGBoost) Shelf-Life Prediction Engine
    """
    def __init__(self, model_dir: str = "models"):
        self.model_path = os.path.join(model_dir, "temperature_humidity_xgboost.pkl")
        self.scaler_path = os.path.join(model_dir, "temperature_humidity_scaler.pkl")
        self.model = None
        self.scaler = None
        self.load_models()

    def load_models(self) -> bool:
        """Load trained XGBoost model and standard scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                return True
        except Exception as e:
            print(f"Warning: Could not load XGBoost model/scaler: {e}")
        return False

    def get_crop_params(self, crop_name: str) -> Dict[str, Any]:
        """Lookup crop parameters from database with case-insensitive fallback"""
        clean_name = crop_name.strip().capitalize()
        if clean_name in CROP_DATABASE:
            return CROP_DATABASE[clean_name]
        
        # Partial matching (e.g. 'fresh_apple' -> 'Apple')
        for key in CROP_DATABASE:
            if key.lower() in clean_name.lower():
                return CROP_DATABASE[key]
        
        # Default fallback
        return CROP_DATABASE["Tomato"]

    def calculate_physics_step(
        self,
        temp_c: float,
        crop_params: Dict[str, Any],
        delta_t_days: float = 1.0 / 24.0
    ) -> Tuple[float, float]:
        """
        Calculates degradation rate and degradation increment for a single time step
        Formula: r(T) = r_ref * Q10^((T - T_ref)/10)
        """
        q10 = crop_params.get("Q10", 2.0)
        t_ref = crop_params.get("T_ref_C", 25.0)
        r_ref = crop_params.get("r_ref_per_day", 1.0)

        # Q10 degradation rate per day
        deg_rate = r_ref * np.power(q10, (temp_c - t_ref) / 10.0)
        deg_increment = deg_rate * delta_t_days
        return float(deg_rate), float(deg_increment)

    def process_sensor_history(
        self,
        readings: List[Dict[str, float]],
        crop_name: str,
        initial_condition: str = "Fresh",
        cnn_confidence: float = 95.0
    ) -> Dict[str, Any]:
        """
        Processes a sequence of temperature and humidity readings:
        1. Computes Q10 physics degradation rate & cumulative degradation
        2. Computes Physics RSL = max(0, S0 - D)
        3. Prepares 24-step feature window and applies XGBoost ML correction
        4. Calculates Final Remaining Shelf Life (RSL) and risk assessment
        """
        crop_params = self.get_crop_params(crop_name)
        s0_days = crop_params.get("S0_days", 10.0)

        # If visual CNN condition is rotten, reduce base shelf-life budget
        if initial_condition.lower() == "rotten":
            s0_days = s0_days * 0.15  # Already deteriorated visually

        if not readings:
            # Default single reading placeholder
            readings = [{"temperature_C": 25.0, "humidity_RH": 70.0}]

        # Build step-by-step physics progression
        cumulative_d = 0.0
        processed_steps = []

        for r in readings:
            t = float(r.get("temperature_C", 25.0))
            h = float(r.get("humidity_RH", 70.0))
            
            # Step duration (default 1 hour = 1/24 day)
            dt = float(r.get("delta_t_days", 1.0 / 24.0))
            
            deg_rate, deg_inc = self.calculate_physics_step(t, crop_params, dt)
            cumulative_d += deg_inc
            rsl_phys = max(0.0, s0_days - cumulative_d)

            processed_steps.append({
                "temperature_C": t,
                "humidity_RH": h,
                "degradation_rate": deg_rate,
                "cumulative_degradation_D": cumulative_d,
                "RSL_physics_days": rsl_phys,
            })

        latest_step = processed_steps[-1]
        rsl_physics = latest_step["RSL_physics_days"]

        # XGBoost ML Correction
        ml_correction = 0.0
        ml_applied = False

        if self.model is not None and self.scaler is not None:
            try:
                # Need a 24-step window; if history is shorter, pad with earliest reading
                feature_matrix = [
                    [
                        step["temperature_C"],
                        step["humidity_RH"],
                        step["degradation_rate"],
                        step["cumulative_degradation_D"],
                        step["RSL_physics_days"],
                    ]
                    for step in processed_steps
                ]

                if len(feature_matrix) < WINDOW_SIZE:
                    # Pad start with first reading so model can run immediately
                    padding = [feature_matrix[0]] * (WINDOW_SIZE - len(feature_matrix))
                    window = padding + feature_matrix
                else:
                    window = feature_matrix[-WINDOW_SIZE:]

                flat_window = np.array(window, dtype=np.float32).reshape(1, -1)
                scaled_window = self.scaler.transform(flat_window)
                ml_correction = float(self.model.predict(scaled_window)[0])
                ml_applied = True
            except Exception as e:
                print(f"Error during XGBoost inference: {e}")
                ml_correction = 0.0
        else:
            # Fallback simple heuristic correction based on humidity stress
            rh_opt_min = crop_params.get("RH_opt_min", 85.0)
            rh_opt_max = crop_params.get("RH_opt_max", 95.0)
            current_rh = latest_step["humidity_RH"]
            if current_rh < rh_opt_min:
                # Dry air causes accelerated moisture loss
                ml_correction = -0.05 * (rh_opt_min - current_rh) / 10.0
            elif current_rh > rh_opt_max:
                # High humidity promotes fungal/mold growth
                ml_correction = -0.08 * (current_rh - rh_opt_max) / 10.0

        # Combine Physics RSL + ML Correction
        rsl_final = max(0.0, rsl_physics + ml_correction)
        
        # Risk assessment level
        if rsl_final <= 2.0 or initial_condition.lower() == "rotten":
            risk_level = "CRITICAL / HIGH"
            risk_color = "#ef4444"
            fefo_priority = 1
        elif rsl_final <= 4.5:
            risk_level = "MODERATE / EXPEDITE"
            risk_color = "#f59e0b"
            fefo_priority = 2
        else:
            risk_level = "OPTIMAL / SAFE"
            risk_color = "#10b981"
            fefo_priority = 3

        # Percentage shelf life remaining
        pct_remaining = min(100.0, max(0.0, (rsl_final / max(0.1, s0_days)) * 100.0))

        return {
            "crop": crop_params["display_name"],
            "category": crop_params["category"],
            "icon": crop_params["icon"],
            "initial_condition": initial_condition,
            "cnn_confidence": cnn_confidence,
            "S0_days": s0_days,
            "current_temp_C": latest_step["temperature_C"],
            "current_humidity_RH": latest_step["humidity_RH"],
            "degradation_rate": latest_step["degradation_rate"],
            "cumulative_degradation_D": latest_step["cumulative_degradation_D"],
            "RSL_physics_days": round(rsl_physics, 2),
            "delta_RSL_ML_days": round(ml_correction, 2),
            "RSL_final_days": round(rsl_final, 2),
            "RSL_final_hours": round(rsl_final * 24.0, 1),
            "pct_remaining": round(pct_remaining, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "fefo_priority": fefo_priority,
            "ml_applied": ml_applied,
            "total_readings_count": len(readings),
            "history_table": processed_steps
        }
