import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

print("Testing imports...")
from shelf_life_engine import ShelfLifeEngine
from vision_detector import VisionQualityDetector
from sensor_manager import ESP32SensorManager
from fefo_routing import FEFORoutingEngine

print("1. Testing Shelf Life Engine...")
engine = ShelfLifeEngine(model_dir="models")
test_readings = [
    {"temperature_C": 26.5 + i * 0.1, "humidity_RH": 74.0 - i * 0.2, "delta_t_days": 1.0/24.0}
    for i in range(24)
]
res = engine.process_sensor_history(test_readings, crop_name="Banana", initial_condition="Fresh", cnn_confidence=96.4)
print(f"   Crop: {res['crop']}, S0: {res['S0_days']}d")
print(f"   Physics RSL: {res['RSL_physics_days']}d, ML Delta: {res['delta_RSL_ML_days']}d -> Final RSL: {res['RSL_final_days']}d")
print(f"   Risk: {res['risk_level']}, FEFO Priority: {res['fefo_priority']}")

print("\n2. Testing Vision Detector...")
detector = VisionQualityDetector()
print(f"   Detector ready: {detector.is_ready}, classes: {detector.class_names}")
dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
v_pred = detector.predict(dummy_frame)
print(f"   Inference output: {v_pred}")
annotated = detector.annotate_frame(dummy_frame, v_pred, shipment_id="SH001")
print(f"   Annotated frame shape: {annotated.shape}")

print("\n3. Testing Sensor Manager...")
sensor_mgr = ESP32SensorManager()
reading = sensor_mgr.get_latest_reading("SH001")
print(f"   Simulated reading: {reading.temperature_C}°C, {reading.humidity_RH}% RH")
history = sensor_mgr.get_shipment_history_dicts("SH001")
print(f"   History count: {len(history)}")

print("\n4. Testing FEFO & Route Optimizer...")
fefo_engine = FEFORoutingEngine()
route_plan = fefo_engine.optimize_vendor_route(
    crop_name="Banana",
    shipment_quantity_kg=2500,
    rsl_final_days=res["RSL_final_days"],
    shipment_id="SH001"
)
print(f"   Recommended Vendor: {route_plan['recommended_vendor']['vendor_name']}")
print(f"   Route: {route_plan['recommended_route']}, ETA: {route_plan['eta_hours']}h, Net Profit: ₹{route_plan['recommended_vendor']['net_profit']}")
print(f"   Rationale: {route_plan['rationale']}")

print("\n✅ All core backend modules verified successfully!")
