import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

# Import custom core modules
from shelf_life_engine import ShelfLifeEngine, CROP_DATABASE
from vision_detector import VisionQualityDetector
from sensor_manager import ESP32SensorManager
from fefo_routing import FEFORoutingEngine, VENDOR_DATABASE
from readme_ui import render_readme_ui, render_algorithm_playground

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Freshroute | Smart Agro-Cold Chain & Shelf-Life Intelligence",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech CSS Theme with High Contrast & Responsive Layout
st.markdown("""
<style>
    /* Global dark theme overrides */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #020617 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45);
    }
    
    .hero-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #10b981 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 13px;
        margin-top: 5px;
        margin-bottom: 0;
        line-height: 1.4;
    }
    
    /* Tech Card containers */
    .tech-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 16px;
    }
    
    .card-title {
        font-size: 14.5px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric highlight cards */
    .metric-box {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #58a6ff;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 10.5px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 3px;
    }
    
    /* Consistent Status Badges (Green / Amber / Red / Sky) */
    .badge-optimal, .badge-fresh {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid #059669;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-critical, .badge-rotten {
        background-color: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-tech, .badge-opt {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid #0284c7;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-dest {
        background-color: rgba(16, 185, 129, 0.22);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.5px;
        display: inline-block;
    }

    /* Warning & info callout cards */
    .callout-box {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 12px;
        color: #cbd5e1;
        margin: 8px 0;
    }
    .callout-warning {
        background: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 12px;
        color: #fca5a5;
        margin: 8px 0;
    }

    /* Sidebar clean styling */
    .sidebar-section-title {
        font-size: 13px;
        font-weight: 700;
        color: #f1f5f9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 14px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION & SYSTEM SINGLETONS
# ==============================================================================
@st.cache_resource
def load_system_engines():
    """Initializes shelf-life engine, vision detector, sensor manager, and routing engine"""
    engine = ShelfLifeEngine(model_dir="models")
    detector = VisionQualityDetector()
    sensor_mgr = ESP32SensorManager()
    fefo = FEFORoutingEngine()
    return engine, detector, sensor_mgr, fefo

engine, detector, sensor_mgr, fefo_engine = load_system_engines()

# Initialize multi-shipment persistent state
if "shipments" not in st.session_state:
    st.session_state.shipments = {
        "SH001": {
            "id": "SH001",
            "farmer": "Ramanathan Agri Farms",
            "truck_id": "TN-23-AB-4412",
            "crop": "Banana",
            "condition": "Fresh",
            "confidence": 95.4,
            "quantity_kg": 3500,
            "origin": "Vellore Green Belt",
            "load_time": "2026-08-29 06:30",
            "truck_lat": 13.0400,
            "truck_lon": 80.1200,
            "readings": [
                {"temperature_C": 24.2 + (i % 3) * 0.4, "humidity_RH": 72.0 - (i % 4) * 0.5, "delta_t_days": 1.0/24.0}
                for i in range(24)
            ]
        },
        "SH002": {
            "id": "SH002",
            "farmer": "Krishna Agro Organics",
            "truck_id": "TN-09-CQ-8910",
            "crop": "Tomato",
            "condition": "Fresh",
            "confidence": 92.1,
            "quantity_kg": 5000,
            "origin": "Krishnagiri Farm",
            "load_time": "2026-08-28 14:00",
            "truck_lat": 12.9800,
            "truck_lon": 80.0500,
            "readings": [
                {"temperature_C": 28.5 + (i % 2) * 0.8, "humidity_RH": 65.0 - (i % 3) * 1.0, "delta_t_days": 1.0/24.0}
                for i in range(32)
            ]
        },
        "SH003": {
            "id": "SH003",
            "farmer": "Nilgiri Orchard Co.",
            "truck_id": "TN-43-XY-1209",
            "crop": "Apple",
            "condition": "Fresh",
            "confidence": 98.0,
            "quantity_kg": 2000,
            "origin": "Ooty High Range",
            "load_time": "2026-08-27 09:00",
            "truck_lat": 13.0800,
            "truck_lon": 80.2000,
            "readings": [
                {"temperature_C": 4.5 + (i % 2) * 0.3, "humidity_RH": 91.0 + (i % 3) * 0.5, "delta_t_days": 1.0/24.0}
                for i in range(48)
            ]
        }
    }

if "active_shipment_id" not in st.session_state:
    st.session_state.active_shipment_id = "SH001"

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "detected_cameras" not in st.session_state:
    st.session_state.detected_cameras = None

if "last_cnn_result" not in st.session_state:
    st.session_state.last_cnn_result = {
        "crop": "Banana",
        "condition": "Fresh",
        "confidence": 95.4,
        "class_name": "fresh_banana",
        "probabilities": {"fresh_banana": 95.4, "rotten_banana": 4.6, "fresh_apple": 0.0, "rotten_apple": 0.0}
    }

if "last_captured_frame" not in st.session_state:
    st.session_state.last_captured_frame = None


# ==============================================================================
# SIDEBAR CONTROLS & ESP32 HARDWARE LINK
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/temperature-condition.png", width=54)
    st.markdown('<div class="sidebar-section-title">📦 Cargo Registry & Fleet</div>', unsafe_allow_html=True)
    
    shipment_list = list(st.session_state.shipments.keys())
    selected_id = st.selectbox("Select Active Cargo Shipment", shipment_list, index=shipment_list.index(st.session_state.active_shipment_id))
    st.session_state.active_shipment_id = selected_id
    curr_shipment = st.session_state.shipments[selected_id]

    with st.expander("➕ Register New Cargo Shipment"):
        new_id = st.text_input("Shipment ID", f"SH{len(st.session_state.shipments)+1:03d}")
        new_farmer = st.text_input("Farmer / Origin", "GreenValley Farms")
        new_truck = st.text_input("Truck Reg", "TN-01-AB-1234")
        new_crop = st.selectbox("Cargo Crop", list(CROP_DATABASE.keys()))
        new_qty = st.number_input("Cargo Quantity (kg)", min_value=100, max_value=50000, value=2500, step=100)
        
        if st.button("Register Shipment", use_container_width=True):
            st.session_state.shipments[new_id] = {
                "id": new_id,
                "farmer": new_farmer,
                "truck_id": new_truck,
                "crop": new_crop,
                "condition": "Fresh",
                "confidence": 95.0,
                "quantity_kg": new_qty,
                "origin": new_farmer,
                "load_time": time.strftime("%Y-%m-%d %H:%M"),
                "truck_lat": 13.0400,
                "truck_lon": 80.1200,
                "readings": [
                    {"temperature_C": 24.5, "humidity_RH": 72.0, "delta_t_days": 1.0/24.0}
                    for _ in range(24)
                ]
            }
            st.session_state.active_shipment_id = new_id
            st.toast(f"✅ Cargo Shipment {new_id} Registered!", icon="📦")
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📡 ESP32 IoT Telemetry Link</div>', unsafe_allow_html=True)
    
    conn_mode = st.radio(
        "Telemetry Source",
        ["Simulator (Real-time Stream)", "Physical Serial (USB/COM)", "Wi-Fi HTTP / IP Stream", "Manual Injection"],
        index=0
    )

    if conn_mode == "Physical Serial (USB/COM)":
        com_ports_info = sensor_mgr.list_available_ports()
        if com_ports_info:
            port_options = [p["port"] for p in com_ports_info]
            port_labels = {p["port"]: p["description"] for p in com_ports_info}
            selected_port = st.selectbox("Select COM Port", port_options, format_func=lambda x: port_labels.get(x, x))
            baud = st.selectbox("Baud Rate", [115200, 9600, 57600], index=0)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("🔌 Connect", use_container_width=True):
                    ok, msg = sensor_mgr.connect_serial(selected_port, baud)
                    if ok:
                        st.toast(f"✅ {msg}", icon="🔌")
                    else:
                        st.toast(f"❌ {msg}", icon="⚠️")
            with col_c2:
                if st.button("❌ Disconnect", use_container_width=True):
                    sensor_mgr.disconnect_serial()
                    st.toast("Serial interface disconnected.", icon="ℹ️")
            
            if sensor_mgr.is_connected:
                st.markdown(f'<span class="badge-optimal">🟢 ESP32 Active: {sensor_mgr.serial_port} @ {sensor_mgr.baud_rate}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-tech">⚪ Status: Disconnected</span>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="callout-box" style="border-left-color: #f59e0b;">
                ⚠️ No COM ports detected. Plug ESP32 via USB or select <b>Simulator</b>.
            </div>
            """, unsafe_allow_html=True)
    
    elif conn_mode == "Wi-Fi HTTP / IP Stream":
        esp32_url = st.text_input("ESP32 JSON Endpoint URL", "http://192.168.1.100/data")
        if st.button("📡 Test Ping", use_container_width=True):
            r, msg = sensor_mgr.fetch_http_reading(esp32_url, timeout=1.5)
            if r:
                st.toast(f"📡 Ping OK: {r.temperature_C}°C, {r.humidity_RH}% RH", icon="✅")
                curr_shipment["readings"].append(r.to_dict())
            else:
                st.toast(f"❌ Ping Failed: {msg}", icon="⚠️")
        st.caption("ESP32 returns JSON: `{\"temperature\": 24.5, \"humidity\": 71.8}`")

    elif conn_mode == "Simulator (Real-time Stream)":
        sim_anomaly = st.selectbox(
            "Environmental Stress Scenario",
            ["NORMAL", "COOLING_FAILURE", "HEATWAVE", "HIGH_HUMIDITY"],
            format_func=lambda x: {
                "NORMAL": "🟢 Normal Cold-Chain (24°C / 72% RH)",
                "COOLING_FAILURE": "🔴 Cooling Unit Failure (Rapid Warming)",
                "HEATWAVE": "🟠 High Heatwave Exposure (Spike)",
                "HIGH_HUMIDITY": "🔵 Excessive Moisture / Condensation Risk"
            }[x]
        )
        sensor_mgr.sim_anomaly = sim_anomaly
        sensor_mgr.sim_base_temp = st.slider("Base Temp (°C)", 0.0, 45.0, 24.0, 0.5)
        sensor_mgr.sim_base_humidity = st.slider("Base RH (%)", 30.0, 98.0, 72.0, 1.0)
    
    elif conn_mode == "Manual Injection":
        man_t = st.slider("Manual Temp (°C)", -5.0, 50.0, 28.5, 0.1)
        man_h = st.slider("Manual RH (%)", 20.0, 100.0, 68.0, 0.5)
        if st.button("Inject Single Reading", use_container_width=True):
            r = sensor_mgr.get_latest_reading(curr_shipment["id"], manual_temp=man_t, manual_hum=man_h)
            curr_shipment["readings"].append(r.to_dict())
            st.toast(f"💉 Injected Reading: {man_t}°C, {man_h}% RH", icon="⚡")
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🔄 Live Stream Engine</div>', unsafe_allow_html=True)
    auto_refresh = st.checkbox("⚡ Auto-Stream Telemetry (2s interval)", value=False)
    
    st.markdown("---")
    st.caption("Freshroute Hybrid AI Intelligence System v2.4")


# ==============================================================================
# MAIN HERO COMMAND BANNER
# ==============================================================================
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="hero-title">🌱 Freshroute : Agro-Logistics & Shelf-Life Command Center</h1>
            <p class="hero-subtitle">
                Cargo <b>{curr_shipment['id']}</b> &nbsp;|&nbsp; 
                Origin: <b>{curr_shipment['farmer']}</b> &nbsp;|&nbsp; 
                Truck: <b>{curr_shipment['truck_id']}</b> &nbsp;|&nbsp; 
                Payload: <b>{curr_shipment['quantity_kg']} kg {curr_shipment['crop']}</b>
            </p>
        </div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap; align-items: center;">
            <span class="badge-tech">📡 ESP32: {conn_mode.split()[0]}</span>
            <span class="badge-tech">👁️ CNN: MobileNetV2</span>
            <span class="badge-tech">🧠 ML: XGBoost</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# TELEMETRY STEP EXECUTION (REAL HARDWARE SERIAL / HTTP / SIMULATOR)
# ==============================================================================
if conn_mode == "Simulator (Real-time Stream)":
    sim_reading = sensor_mgr.generate_simulated_reading()
    curr_shipment["readings"].append(sim_reading.to_dict())
    if len(curr_shipment["readings"]) > 120:
        curr_shipment["readings"] = curr_shipment["readings"][-120:]

elif conn_mode == "Physical Serial (USB/COM)" and sensor_mgr.is_connected:
    s_reading = sensor_mgr.read_serial_line()
    if s_reading:
        curr_shipment["readings"].append(s_reading.to_dict())
        if len(curr_shipment["readings"]) > 120:
            curr_shipment["readings"] = curr_shipment["readings"][-120:]

elif conn_mode == "Wi-Fi HTTP / IP Stream":
    if auto_refresh:
        h_reading, _ = sensor_mgr.fetch_http_reading(sensor_mgr.http_url, timeout=1.0)
        if h_reading:
            curr_shipment["readings"].append(h_reading.to_dict())
            if len(curr_shipment["readings"]) > 120:
                curr_shipment["readings"] = curr_shipment["readings"][-120:]


# ==============================================================================
# TOP-LEVEL NAVIGATION TABS
# ==============================================================================
tab_command, tab_readme, tab_lab = st.tabs([
    "🚀 Live Agro-Cold Chain Command Center",
    "📚 README UI : Complete Architecture & Blueprint",
    "🧪 Interactive Algorithm Lab & Math Playground"
])


# ==============================================================================
# TAB 1: LIVE AGRO-COLD CHAIN COMMAND CENTER
# ==============================================================================
with tab_command:
    # --------------------------------------------------------------------------
    # TOP SECTION: 2-COLUMN LAYOUT (OPENCV VISION + ESP32 SENSORS)
    # --------------------------------------------------------------------------
    col_vision, col_sensor = st.columns([1, 1])

    # SECTION 1: OPENCV & CNN REALTIME QUALITY / FRESHNESS VERIFICATION
    with col_vision:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📷 Visual Freshness Verification (OpenCV + MobileNetV2 CNN)</div>', unsafe_allow_html=True)

        vision_mode = st.radio(
            "Camera Input Source",
            ["📱 Phone / Browser Camera (Direct Snapshot)", "💻 Laptop OpenCV Webcam (USB / Hardware)", "📁 Upload Cargo Photo", "🍎 Sample Test Specimens"],
            index=0,
            horizontal=True
        )

        current_frame = None

        if vision_mode == "📱 Phone / Browser Camera (Direct Snapshot)":
            st.markdown("""
            <div class="callout-box" style="border-left-color: #10b981; margin-bottom: 10px;">
                📱 <b>Phone / Browser Camera Active:</b> Tap <b>"Take Photo"</b> below to activate your phone's rear or front camera for instant CNN quality verification.
            </div>
            """, unsafe_allow_html=True)
            cam_pic = st.camera_input("📸 Tap to capture fruit / vegetable cargo with your phone's camera:")
            if cam_pic is not None:
                bytes_data = cam_pic.getvalue()
                current_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                st.session_state.last_captured_frame = current_frame

        elif vision_mode == "💻 Laptop OpenCV Webcam (USB / Hardware)":
            # Auto-probe or scan devices
            if st.session_state.detected_cameras is None:
                st.session_state.detected_cameras = detector.scan_available_cameras(5)

            found_cams = st.session_state.detected_cameras or []
            if found_cams:
                cam_choices = [c["index"] for c in found_cams]
                cam_labels = {c["index"]: c["label"] for c in found_cams}
            else:
                cam_choices = [0, 1, 2, 3, 4]
                cam_labels = {i: f"Camera {i} (Standard Probe)" for i in range(5)}

            cam_c1, cam_c2, cam_c3 = st.columns([1.3, 1, 0.9])
            with cam_c1:
                selected_cam_idx = st.selectbox(
                    "Select Camera Device",
                    cam_choices,
                    format_func=lambda x: cam_labels.get(x, f"Camera {x}")
                )
            with cam_c2:
                cam_backend = st.selectbox(
                    "Capture Backend",
                    ["AUTO", "DSHOW", "MSMF", "DEFAULT"],
                    index=0,
                    help="AUTO tries DirectShow (Windows), then MSMF, then Default."
                )
            with cam_c3:
                warmup_f = st.slider("Warmup Frames", 5, 20, 12, help="Discards initial frames so exposure/gain settles.")

            v_btn1, v_btn2, v_btn3 = st.columns(3)
            with v_btn1:
                capture_single = st.button("📸 Snap Frame", use_container_width=True)
            with v_btn2:
                start_live = st.button("▶️ Start Live Feed", use_container_width=True)
            with v_btn3:
                if st.button("🔍 Scan Cameras", use_container_width=True):
                    with st.spinner("Scanning video devices (indices 0–4)..."):
                        st.session_state.detected_cameras = detector.scan_available_cameras(5)
                        if st.session_state.detected_cameras:
                            st.toast(f"Found {len(st.session_state.detected_cameras)} active camera(s)!", icon="📹")
                        else:
                            st.toast("No active camera found. Use Browser / Sample mode.", icon="⚠️")
                        st.rerun()

            # Handle Snap Frame
            if capture_single:
                ok, frame, msg, is_black = detector.capture_frame_with_warmup(
                    camera_index=selected_cam_idx,
                    backend_name=cam_backend,
                    warmup_frames=warmup_f
                )
                if ok and frame is not None:
                    current_frame = frame
                    st.session_state.last_captured_frame = frame
                    if is_black:
                        st.markdown(f'<div class="callout-warning">{msg}</div>', unsafe_allow_html=True)
                    else:
                        st.toast("📸 Frame captured successfully!", icon="✅")
                else:
                    st.markdown(f"""
                    <div class="callout-warning">
                        <b>⚠️ Camera Device Notice:</b> {msg}<br>
                        <span style="font-size: 11.5px; color: #cbd5e1;">Switching automatically to fallback specimen mode.</span>
                    </div>
                    """, unsafe_allow_html=True)
                    sample_path = "sample_images/fresh_banana.jpg" if curr_shipment["crop"] == "Banana" else "sample_images/fresh_apple.jpg"
                    if os.path.exists(sample_path):
                        current_frame = cv2.imread(sample_path)
                        st.session_state.last_captured_frame = current_frame

            # Handle Live Video Loop with controlled stop and guaranteed release
            if start_live:
                video_placeholder = st.empty()
                stop_btn_placeholder = st.empty()
                
                stop_clicked = stop_btn_placeholder.button("⏹️ Stop Live Feed", key="stop_live_btn", use_container_width=True)
                
                cap, b_used = detector.open_video_capture(selected_cam_idx, cam_backend)
                
                if cap is None or not cap.isOpened():
                    st.markdown(f"""
                    <div class="callout-warning">
                        <b>⚠️ Camera Failed to Open (Index {selected_cam_idx} via {b_used}):</b><br>
                        Unable to access video feed. Verify physical connection or select another camera device.<br>
                        <i>Tip: You can also use <b>Browser Camera Snapshot</b> or <b>Sample Test Specimens</b> above.</i>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    try:
                        # Auto-exposure warmup
                        for _ in range(warmup_f):
                            cap.read()
                            time.sleep(0.01)

                        st.toast("📹 Live OpenCV inspection stream active...", icon="🔴")
                        
                        for frame_idx in range(50):
                            ret, f = cap.read()
                            if not ret or f is None:
                                break
                            
                            if f.mean() < 2.0:
                                cv2.putText(f, "CHECK PRIVACY SHUTTER", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                            pred = detector.predict(f)
                            ann = detector.annotate_frame(f, pred, curr_shipment["id"])
                            disp = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                            
                            video_placeholder.image(
                                disp,
                                caption=f"Live CNN Stream: {pred['crop']} - {pred['condition']} ({pred['confidence']}%)",
                                use_container_width=True
                            )
                            
                            st.session_state.last_cnn_result = pred
                            st.session_state.last_captured_frame = f
                            curr_shipment["crop"] = pred["crop"]
                            curr_shipment["condition"] = pred["condition"]
                            curr_shipment["confidence"] = pred["confidence"]
                            
                            time.sleep(0.03)

                    except Exception as ex:
                        st.toast(f"Stream interrupted: {ex}", icon="⚠️")
                    finally:
                        if cap is not None and cap.isOpened():
                            cap.release()
                        stop_btn_placeholder.empty()
                        st.toast("✅ Live camera stream safely closed.", icon="📷")

        elif vision_mode == "Upload Cargo Photo":
            uploaded_file = st.file_uploader("Upload Cargo Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                bytes_data = uploaded_file.read()
                current_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                st.session_state.last_captured_frame = current_frame

        elif vision_mode == "Sample Test Specimens":
            sample_options = ["Fresh Banana", "Rotten Banana", "Fresh Apple", "Rotten Apple", "Fresh Tomato", "Fresh Mango"]
            if os.path.exists("sample_images/live_capture.jpg"):
                sample_options.insert(0, "📸 Latest Standalone OpenCV Live Capture")

            sample_choice = st.selectbox("Select Test Cargo Specimen", sample_options)
            fname_map = {
                "📸 Latest Standalone OpenCV Live Capture": "sample_images/live_capture.jpg",
                "Fresh Banana": "sample_images/fresh_banana.jpg",
                "Rotten Banana": "sample_images/rotten_banana.jpg",
                "Fresh Apple": "sample_images/fresh_apple.jpg",
                "Rotten Apple": "sample_images/rotten_apple.jpg",
                "Fresh Tomato": "sample_images/fresh_tomato.jpg",
                "Fresh Mango": "sample_images/fresh_mango.jpg",
            }
            fpath = fname_map.get(sample_choice)
            if fpath and os.path.exists(fpath):
                current_frame = cv2.imread(fpath)
                st.session_state.last_captured_frame = current_frame

        # Fallback to stored frame if available
        if current_frame is None and st.session_state.last_captured_frame is not None:
            current_frame = st.session_state.last_captured_frame

        # Run CNN inference and render annotated HUD
        if current_frame is not None:
            cnn_pred = detector.predict(current_frame)
            st.session_state.last_cnn_result = cnn_pred
            
            curr_shipment["crop"] = cnn_pred["crop"]
            curr_shipment["condition"] = cnn_pred["condition"]
            curr_shipment["confidence"] = cnn_pred["confidence"]

            annotated_frame = detector.annotate_frame(current_frame, cnn_pred, curr_shipment["id"])
            disp_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(disp_rgb, caption=f"Visual Verification HUD: {cnn_pred['crop']} - {cnn_pred['condition']} ({cnn_pred['confidence']}%)", use_container_width=True)
        else:
            cnn_pred = st.session_state.last_cnn_result

        # CNN Inference Status Row with high-contrast badge
        is_fresh = curr_shipment["condition"].lower() == "fresh"
        badge_class = "badge-optimal" if is_fresh else "badge-critical"
        status_label = "PASSED - FRESH" if is_fresh else "FAILED - ROTTEN"
        icon = CROP_DATABASE.get(curr_shipment["crop"], {}).get("icon", "📦")
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: #21262d; padding: 8px 12px; border-radius: 8px; margin-top: 8px; border: 1px solid #30363d;">
            <div>
                <span style="font-size: 15px; font-weight: 700; color: #f8fafc;">{icon} {curr_shipment['crop']}</span>
                <span style="font-size: 11px; color: #94a3b8; margin-left: 6px;">(MobileNetV2 Visual CNN)</span>
            </div>
            <div>
                <span class="{badge_class}">{status_label} ({curr_shipment['confidence']}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Restyled Probability Bar Chart matching Dark Theme
        if "probabilities" in cnn_pred and cnn_pred["probabilities"]:
            prob_df = pd.DataFrame([
                {"Class": k.replace("_", " ").title(), "Probability (%)": v}
                for k, v in cnn_pred["probabilities"].items()
            ])
            fig_prob = px.bar(
                prob_df, x="Probability (%)", y="Class", orientation="h",
                color="Probability (%)",
                color_continuous_scale=[[0, "#1e293b"], [0.5, "#0284c7"], [1.0, "#10b981"]],
                height=140
            )
            fig_prob.update_layout(
                margin=dict(l=6, r=6, t=6, b=6),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#cbd5e1", size=10),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#21262d", range=[0, 100]),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 2: ESP32 ENVIRONMENTAL TELEMETRY & REALTIME DUAL-AXIS CHARTS
    with col_sensor:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📡 ESP32 Environmental Telemetry (DHT22 / DHT11 Stream)</div>', unsafe_allow_html=True)

        history = curr_shipment.get("readings", [])
        latest_r = history[-1] if history else {"temperature_C": 24.5, "humidity_RH": 72.0}

        crop_meta = engine.get_crop_params(curr_shipment["crop"])
        t_opt_min = crop_meta.get("T_opt_min", 12.0)
        t_opt_max = crop_meta.get("T_opt_max", 20.0)
        rh_opt_min = crop_meta.get("RH_opt_min", 85.0)
        rh_opt_max = crop_meta.get("RH_opt_max", 95.0)

        m1, m2, m3 = st.columns(3)
        with m1:
            t_val = latest_r["temperature_C"]
            t_color = "#10b981" if t_opt_min <= t_val <= t_opt_max else ("#ef4444" if t_val > 30.0 else "#f59e0b")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {t_color};">{t_val:.1f} °C</div>
                <div class="metric-label">Temperature</div>
                <div style="font-size: 9.5px; color: #94a3b8; margin-top: 3px;">Opt: {t_opt_min}-{t_opt_max}°C</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            h_val = latest_r["humidity_RH"]
            h_color = "#10b981" if rh_opt_min <= h_val <= rh_opt_max else ("#ef4444" if h_val < 50.0 else "#f59e0b")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {h_color};">{h_val:.1f} %</div>
                <div class="metric-label">Relative Humidity</div>
                <div style="font-size: 9.5px; color: #94a3b8; margin-top: 3px;">Opt: {rh_opt_min}-{rh_opt_max}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            q10_rate, _ = engine.calculate_physics_step(t_val, crop_meta)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #a78bfa;">{q10_rate:.2f}x</div>
                <div class="metric-label">Degradation Rate r(T)</div>
                <div style="font-size: 9.5px; color: #94a3b8; margin-top: 3px;">Q10: {crop_meta['Q10']} @ {crop_meta['T_ref_C']}°C</div>
            </div>
            """, unsafe_allow_html=True)

        if len(history) > 0:
            hist_df = pd.DataFrame(history)
            hist_df["step"] = range(1, len(hist_df) + 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["step"], y=hist_df["temperature_C"],
                name="Temperature (°C)", mode="lines+markers",
                line=dict(color="#f87171", width=2.5, shape="spline"),
                marker=dict(size=4),
                hovertemplate="Step %{x}: %{y:.1f} °C<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=hist_df["step"], y=hist_df["humidity_RH"],
                name="Humidity (% RH)", mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5, shape="spline"),
                marker=dict(size=4),
                yaxis="y2",
                hovertemplate="Step %{x}: %{y:.1f} %% RH<extra></extra>"
            ))

            fig.add_hrect(
                y0=t_opt_min, y1=t_opt_max, fillcolor="rgba(16, 185, 129, 0.08)",
                line_width=0, annotation_text="Optimal Temp Band", annotation_position="top left",
                annotation_font=dict(size=9, color="#34d399")
            )

            fig.update_layout(
                height=210,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#cbd5e1", size=10.5),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                yaxis=dict(title="Temp (°C)", color="#f87171", showgrid=True, gridcolor="#21262d"),
                yaxis2=dict(title="RH (%)", color="#38bdf8", overlaying="y", side="right", showgrid=False),
                xaxis=dict(title="Telemetry Time Steps (Hourly / Sensor Intervals)", showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 3: HYBRID SHELF-LIFE ENGINE (PHYSICS Q10 + XGBOOST ML REGRESSOR)
    # --------------------------------------------------------------------------
    shelf_res = engine.process_sensor_history(
        readings=curr_shipment["readings"],
        crop_name=curr_shipment["crop"],
        initial_condition=curr_shipment["condition"],
        cnn_confidence=curr_shipment["confidence"]
    )

    curr_shipment["RSL_final_days"] = shelf_res["RSL_final_days"]
    curr_shipment["RSL_physics_days"] = shelf_res["RSL_physics_days"]
    curr_shipment["delta_RSL_ML_days"] = shelf_res["delta_RSL_ML_days"]
    curr_shipment["risk_level"] = shelf_res["risk_level"]
    curr_shipment["fefo_priority"] = shelf_res["fefo_priority"]

    st.markdown('<div class="tech-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧠 Real-Time Shelf-Life Engine: Q₁₀ Kinetics + XGBoost ML Correction</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #a5b4fc;">{shelf_res['S0_days']:.1f} Days</div>
            <div class="metric-label">Initial Shelf Budget (S₀)</div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 3px;">Baseline at Harvest / Visual</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #38bdf8;">{shelf_res['RSL_physics_days']:.2f} Days</div>
            <div class="metric-label">Physics RSL (S₀ - D)</div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 3px;">Cum. Loss D = {shelf_res['cumulative_degradation_D']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        delta_sign = "+" if shelf_res['delta_RSL_ML_days'] >= 0 else ""
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #f59e0b;">{delta_sign}{shelf_res['delta_RSL_ML_days']:.2f} Days</div>
            <div class="metric-label">XGBoost ML Correction (Δ)</div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 3px;">24-step Non-Linear Stress</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-box" style="border: 2px solid {shelf_res['risk_color']};">
            <div class="metric-value" style="color: {shelf_res['risk_color']}; font-size: 22px;">
                {shelf_res['RSL_final_days']:.2f} Days
            </div>
            <div class="metric-label" style="font-weight: 700;">Final Remaining Shelf Life</div>
            <div style="font-size: 10.5px; color: {shelf_res['risk_color']}; margin-top: 3px; font-weight: 700;">
                {shelf_res['RSL_final_hours']} Hours ({shelf_res['risk_level']})
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 12px;'>", unsafe_allow_html=True)
    pct_val = shelf_res['pct_remaining']
    st.progress(pct_val / 100.0, text=f"Remaining Shelf Life Budget: {pct_val:.1f}% ({shelf_res['RSL_final_days']:.2f} days / {shelf_res['RSL_final_hours']:.1f} hours remaining)")

    st.markdown(f"""
    <div style="background: #0d1117; padding: 10px 14px; border-radius: 8px; border: 1px solid #21262d; margin-top: 8px; font-size: 12.5px; color: #cbd5e1;">
        <b>📐 Formula Pipeline:</b> 
        <code>RSL_final = (S₀ - ∑ r(T)Δt) + XGBoost(T, RH, r(T), D, RSL_phys)</code> = 
        <b>{shelf_res['S0_days']:.1f}</b> - <b>{shelf_res['cumulative_degradation_D']:.2f}</b> + (<b>{shelf_res['delta_RSL_ML_days']:+.2f}</b>) = 
        <span style="color: {shelf_res['risk_color']}; font-weight: bold;">{shelf_res['RSL_final_days']:.2f} Days ({shelf_res['RSL_final_hours']:.1f} Hours remaining)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 4 & 5: FEFO DISPATCH QUEUE & GEOSPATIAL DEMAND MAP
    # --------------------------------------------------------------------------
    col_fefo, col_route = st.columns([1, 1.3])

    # SECTION 4: FEFO DISPATCH QUEUE
    with col_fefo:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚡ FEFO Multi-Shipment Dispatch Priority Queue</div>', unsafe_allow_html=True)

        all_shipment_list = list(st.session_state.shipments.values())
        fefo_ranked = fefo_engine.rank_fefo_queue(all_shipment_list)

        fefo_table_data = []
        for s in fefo_ranked:
            is_active = "👉 " if s["id"] == curr_shipment["id"] else ""
            fefo_table_data.append({
                "Rank": f"#{s['fefo_rank']}",
                "Shipment": f"{is_active}{s['id']}",
                "Crop": s["crop"],
                "Condition": s["condition"],
                "RSL (Days)": f"{s.get('RSL_final_days', 0.0):.2f}d",
                "Urgency / Action": s["dispatch_urgency"]
            })

        fefo_df = pd.DataFrame(fefo_table_data)
        st.dataframe(fefo_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="font-size: 11.5px; color: #94a3b8; margin-top: 6px;">
            💡 <b>FEFO Principle:</b> Batches with lowest Remaining Shelf Life are prioritized for immediate dispatch to eliminate in-transit spoilage.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<b>📍 Truck GPS Simulator (Current Position):</b>", unsafe_allow_html=True)
        loc_preset = st.selectbox(
            "Move Truck to Preset Corridor",
            [
                "Vellore Highway (13.0400, 80.1200)",
                "Outer Ring Road (13.0100, 80.0800)",
                "Guindy Central (13.0067, 80.2025)",
                "Koyambedu Bypass (13.0694, 80.1948)",
                "Sriperumbudur Industrial (12.9675, 79.9406)"
            ]
        )
        preset_coords = {
            "Vellore Highway (13.0400, 80.1200)": (13.0400, 80.1200),
            "Outer Ring Road (13.0100, 80.0800)": (13.0100, 80.0800),
            "Guindy Central (13.0067, 80.2025)": (13.0067, 80.2025),
            "Koyambedu Bypass (13.0694, 80.1948)": (13.0694, 80.1948),
            "Sriperumbudur Industrial (12.9675, 79.9406)": (12.9675, 79.9406)
        }
        t_lat, t_lon = preset_coords.get(loc_preset, (13.0400, 80.1200))
        curr_shipment["truck_lat"] = t_lat
        curr_shipment["truck_lon"] = t_lon

        st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 5: DEMAND MAP OF EXISTING SHOPS & NEXT DESTINATION ROUTE
    with col_route:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗺️ Nearby Shops Demand Map & Next Stop Route</div>', unsafe_allow_html=True)

        routing_result = fefo_engine.optimize_vendor_route(
            crop_name=curr_shipment["crop"],
            shipment_quantity_kg=curr_shipment["quantity_kg"],
            rsl_final_days=shelf_res["RSL_final_days"],
            truck_lat=curr_shipment.get("truck_lat", 13.0400),
            truck_lon=curr_shipment.get("truck_lon", 80.1200),
            shipment_id=curr_shipment["id"]
        )

        best_v = routing_result["recommended_vendor"]

        if best_v:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9)); border: 1.5px solid #10b981; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span class="badge-dest">🎯 RECOMMENDED NEXT STOP</span>
                        <h3 style="margin: 4px 0 2px 0; color: #f8fafc; font-size: 17px;">{best_v['vendor_name']}</h3>
                        <p style="margin: 0; font-size: 11.5px; color: #cbd5e1;">
                            🏬 {best_v['type']} &nbsp;|&nbsp; 📍 {best_v['city']} &nbsp;|&nbsp; 📞 {best_v['contact']} ({best_v['manager']})
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span class="badge-tech">₹{best_v['price_per_kg']}/kg</span>
                        <div style="font-size: 17px; font-weight: 800; color: #34d399; margin-top: 2px;">Net: ₹{best_v['net_profit']:,}</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px; margin-top: 10px; border-top: 1px solid #334155; padding-top: 8px;">
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">DISTANCE</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #f1f5f9;">{best_v['distance_km']} km</div>
                    </div>
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">TRANSIT ETA</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #38bdf8;">~{best_v['transit_minutes']} Mins</div>
                    </div>
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">SHOP DEMAND</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #f59e0b;">{best_v['demand_kg']:,} kg</div>
                    </div>
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">SHELF BUFFER</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #4ade80;">+{best_v['margin_hours']}h</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(routing_result["rationale"])

        active_crop = curr_shipment["crop"]
        map_points = []
        map_points.append({
            "name": f"🚚 Truck [{curr_shipment['truck_id']}]",
            "lat": curr_shipment.get("truck_lat", 13.0400),
            "lon": curr_shipment.get("truck_lon", 80.1200),
            "category": "Current Truck Location",
            "demand_kg": curr_shipment["quantity_kg"],
            "price_per_kg": 0.0,
            "marker_size": 20,
            "color": "#38bdf8",
            "details": f"Cargo: {curr_shipment['quantity_kg']}kg {active_crop} | Status: In Transit"
        })

        for v in VENDOR_DATABASE:
            is_best = (best_v and v["id"] == best_v["vendor_id"])
            d_crop = v["demands_kg"].get(active_crop, 0)
            p_crop = v["prices_per_kg"].get(active_crop, 0.0)
            
            size_scaled = int(max(10, min(28, np.sqrt(d_crop) * 0.35)))
            if is_best:
                size_scaled = max(size_scaled, 22)

            cat = "🎯 Recommended Next Stop" if is_best else ("🏪 Active Demand Shop" if d_crop > 0 else "🏬 Other Retailer")
            color = "#10b981" if is_best else ("#f59e0b" if d_crop > 0 else "#64748b")

            map_points.append({
                "name": f"{'⭐ DESTINATION: ' if is_best else '🏪 '}{v['name']}",
                "lat": v["lat"],
                "lon": v["lon"],
                "category": cat,
                "demand_kg": d_crop,
                "price_per_kg": p_crop,
                "marker_size": size_scaled,
                "color": color,
                "details": f"{v['city']} | Demand: {d_crop:,}kg | Price: ₹{p_crop}/kg | Cold Storage: {'❄️ Yes' if v['has_cold_storage'] else '❌ No'}"
            })

        map_df = pd.DataFrame(map_points)
        fig_map = go.Figure()
        ScatterMapTrace = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))

        if routing_result.get("route_waypoints"):
            wps = routing_result["route_waypoints"]
            w_lats = [pt["lat"] for pt in wps]
            w_lons = [pt["lon"] for pt in wps]
            
            fig_map.add_trace(ScatterMapTrace(
                lat=w_lats,
                lon=w_lons,
                mode="lines",
                line=dict(width=4, color="#10b981"),
                name="Recommended Route Polyline",
                hoverinfo="text",
                text=f"Navigation Corridor ➡️ {best_v['vendor_name']} ({best_v['distance_km']} km, ~{best_v['transit_minutes']} mins)"
            ))

        for cat_name, group in map_df.groupby("category"):
            fig_map.add_trace(ScatterMapTrace(
                lat=group["lat"],
                lon=group["lon"],
                mode="markers+text" if ("Truck" in cat_name or "Recommended" in cat_name) else "markers",
                marker=dict(
                    size=group["marker_size"],
                    color=group["color"],
                    opacity=0.9
                ),
                text=group["name"],
                textposition="top center",
                name=cat_name,
                hoverinfo="text",
                hovertext=[f"<b>{row['name']}</b><br>{row['details']}" for _, row in group.iterrows()]
            ))

        avg_lat = float(map_df["lat"].mean())
        avg_lon = float(map_df["lon"].mean())

        map_layout_kwargs = dict(
            style="carto-darkmatter",
            center=dict(lat=avg_lat, lon=avg_lon),
            zoom=10.2
        )

        if hasattr(go, "Scattermap"):
            fig_map.update_layout(map=map_layout_kwargs)
        else:
            fig_map.update_layout(mapbox=map_layout_kwargs)

        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=320,
            paper_bgcolor="#161b22",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(15, 23, 42, 0.85)",
                font=dict(color="#f1f5f9", size=10)
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        with st.expander(f"📊 Explore Nearby Shops Demand Matrix ({active_crop})", expanded=False):
            all_cands = routing_result.get("all_candidates", [])
            if all_cands:
                cands_df = pd.DataFrame([
                    {
                        "Shop Name": c["vendor_name"],
                        "Type": c["type"],
                        "City Area": c["city"],
                        "Demand (kg)": f"{c['demand_kg']:,} kg",
                        "Price (₹/kg)": f"₹{c['price_per_kg']}",
                        "Distance": f"{c['distance_km']} km",
                        "ETA": f"{c['transit_minutes']} mins",
                        "Net Profit": f"₹{c['net_profit']:,}",
                        "Cold Store": "❄️ Yes" if c["has_cold_storage"] else "❌ No",
                        "Buffer": f"+{c['margin_hours']}h"
                    }
                    for c in all_cands
                ])
                st.dataframe(cands_df, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Footer metrics
    st.markdown("---")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.caption(f"🕒 Engine Sync: {time.strftime('%H:%M:%S')}")
    with f_col2:
        st.caption(f"💾 Active Records: {len(curr_shipment.get('readings', []))} telemetry readings")
    with f_col3:
        st.caption("🔒 Checksums: XGBoost (Loaded), MobileNetV2 (Active)")
    with f_col4:
        if st.button("🔄 Force Real-Time Refresh", use_container_width=True):
            st.rerun()


# ==============================================================================
# TAB 2: SYSTEM ARCHITECTURE & README UI
# ==============================================================================
with tab_readme:
    render_readme_ui()


# ==============================================================================
# TAB 3: INTERACTIVE ALGORITHM LAB & MATH PLAYGROUND
# ==============================================================================
with tab_lab:
    render_algorithm_playground()


# ==============================================================================
# PERIODIC AUTO-REFRESH SCRIPT
# ==============================================================================
if auto_refresh:
    time.sleep(2)
    st.rerun()
