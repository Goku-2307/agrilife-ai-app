import os
import sys
import time
import cv2
import numpy as np

# Force UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vision_detector import VisionQualityDetector
from sensor_manager import ESP32SensorManager
from shelf_life_engine import ShelfLifeEngine

def main():
    print("=" * 70)
    print("   FRESHROUTE - STANDALONE OPENCV LIVE VISION & HARDWARE HUD")
    print("=" * 70)
    
    # 1. Initialize PyTorch CNN & Shelf Life Engine
    print("[1/3] Loading PyTorch MobileNetV2 CNN Quality Detector...")
    detector = VisionQualityDetector()
    engine = ShelfLifeEngine(model_dir="models")
    
    if not detector.is_ready:
        print("[WARNING] CNN model weights not loaded. Using heuristic fallback.")

    # 2. Connect to Hardware ESP32 on COM port
    print("[2/3] Connecting to Physical ESP32 Hardware...")
    sensor_mgr = ESP32SensorManager()
    available_ports = sensor_mgr.list_available_ports()
    
    esp_connected = False
    if available_ports:
        print(f"Found {len(available_ports)} Serial Port(s):")
        for p in available_ports:
            print(f"  - {p['port']}: {p['description']}")
        
        # Auto-connect to first available COM port (e.g. COM14)
        target_port = available_ports[0]["port"]
        ok, msg = sensor_mgr.connect_serial(target_port, 115200)
        if ok:
            print(f"[SUCCESS] {msg}")
            esp_connected = True
        else:
            print(f"[WARNING] Could not connect to {target_port}: {msg}. Falling back to Simulator.")
    else:
        print("[INFO] No USB COM ports detected. Running in Sensor Simulation Mode.")

    # 3. Discover and Open Camera
    print("[3/3] Scanning and Opening Video Camera...")
    detected_cams = detector.scan_available_cameras(4)
    cam_index = 0
    if detected_cams:
        print(f"Found {len(detected_cams)} camera device(s):")
        for c in detected_cams:
            print(f"  - {c['label']}")
        active_cams = [c for c in detected_cams if not c.get("is_black", False)]
        if active_cams:
            cam_index = active_cams[0]["index"]
        else:
            cam_index = detected_cams[0]["index"]
    
    print(f"Opening Camera index {cam_index} with DirectShow / Multi-Backend Fallback...")
    cap, backend_used = detector.open_video_capture(cam_index, "AUTO")

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_index}. Check connections & privacy permissions.")
        return

    print(f"[SUCCESS] Connected to camera via {backend_used} backend.")

    # Warmup camera frames to let auto-exposure calibrate
    print("Calibrating camera sensor auto-exposure (12 warmup frames)...")
    for _ in range(12):
        cap.read()
        time.sleep(0.02)

    print("\n" + "=" * 70)
    print("   LIVE OPENCV WINDOW RUNNING (Separate Native Display)")
    print("   Controls:")
    print("     [S]     : Save inspection snapshot frame")
    print("     [C]     : Cycle to next Camera Device")
    print("     [Q/ESC] : Quit & Release Hardware")
    print("=" * 70 + "\n")

    frame_count = 0
    start_time = time.time()
    last_pred = detector.predict(None)
    
    # Track sensor reading
    current_temp = 24.5
    current_rh = 72.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame from camera.")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Read live hardware telemetry from ESP32
            if esp_connected:
                s_read = sensor_mgr.read_serial_line()
                if s_read:
                    current_temp = s_read.temperature_C
                    current_rh = s_read.humidity_RH
            else:
                sim_read = sensor_mgr.generate_simulated_reading()
                current_temp = sim_read.temperature_C
                current_rh = sim_read.humidity_RH

            # Check if frame is pitch black
            if frame.mean() < 2.0:
                cv2.putText(
                    frame, "WARNING: PITCH BLACK FRAME (OPEN PRIVACY SHUTTER)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

            # Run CNN inference every 2 frames for smooth 30+ FPS rendering
            if frame_count % 2 == 0:
                last_pred = detector.predict(frame)

            annotated = detector.annotate_frame(frame, last_pred, shipment_id="SH001")

            # Calculate Physics Shelf Life step for current crop
            crop_name = last_pred.get("crop", "Banana")
            crop_meta = engine.get_crop_params(crop_name)
            q10_rate, _ = engine.calculate_physics_step(current_temp, crop_meta)

            # Draw Hardware Telemetry HUD Box on Bottom-Left
            h, w = annotated.shape[:2]
            hud_bg = annotated.copy()
            cv2.rectangle(hud_bg, (15, h - 110), (320, h - 25), (15, 23, 42), -1)
            cv2.addWeighted(hud_bg, 0.8, annotated, 0.2, 0, annotated)
            cv2.rectangle(annotated, (15, h - 110), (320, h - 25), (56, 189, 248), 1)

            t_str = f"ESP32 Temp : {current_temp:.1f} C"
            h_str = f"ESP32 Hum  : {current_rh:.1f} % RH"
            q_str = f"Q10 Rate   : {q10_rate:.2f}x ({crop_name})"

            cv2.putText(annotated, t_str, (25, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 113, 113), 1)
            cv2.putText(annotated, h_str, (25, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1)
            cv2.putText(annotated, q_str, (25, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (167, 139, 250), 1)
            cv2.putText(annotated, f"Hardware: {'COM14 (ESP32 Live)' if esp_connected else 'Simulator'}", (25, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)

            # Calculate and display FPS
            fps = frame_count / max(0.1, time.time() - start_time)
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1
            )

            # Show Native OpenCV Window
            cv2.imshow("Freshroute - Standalone Live OpenCV Hardware Window", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                os.makedirs("sample_images", exist_ok=True)
                filename = f"sample_images/live_capture.jpg"
                cv2.imwrite(filename, annotated)
                print(f"[SAVED] Saved inspection snapshot frame: {filename}")
            elif key == ord('c'):
                # Cycle camera index
                cam_index = (cam_index + 1) % 3
                print(f"[SWITCH] Switching to Camera index {cam_index}...")
                cap.release()
                cap, backend_used = detector.open_video_capture(cam_index, "AUTO")
                if cap is None or not cap.isOpened():
                    print(f"[ERROR] Could not open camera {cam_index}, switching back to index 0.")
                    cam_index = 0
                    cap, backend_used = detector.open_video_capture(0, "AUTO")

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if esp_connected:
            sensor_mgr.disconnect_serial()
        print("\n[INFO] OpenCV window closed and hardware connections released safely.")

if __name__ == "__main__":
    main()
