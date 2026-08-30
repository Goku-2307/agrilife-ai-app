import os
import sys
import time
import json
import subprocess
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
from fefo_routing import FEFORoutingEngine, VENDOR_DATABASE, haversine_distance_km

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Freshroute | Smart Agro-Cold Chain & Shelf-Life Command",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

if "current_page" not in st.session_state:
    st.session_state.current_page = "📷 Camera Verification"

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

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "⚡ Vibrant Cyber Neon"


# ==============================================================================
# SYSTEM SINGLETON ENGINES
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


# ==============================================================================
# THEME STYLES (VIBRANT CYBER NEON VS FORMAL SLATE VS FORMAL LIGHT)
# ==============================================================================
def inject_theme_css(theme_name: str):
    if theme_name == "⚡ Vibrant Cyber Neon":
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', sans-serif;
            }
            code, pre, .font-mono {
                font-family: 'JetBrains Mono', monospace;
            }

            .stApp {
                background-color: #060913;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.08) 0px, transparent 50%),
                    radial-gradient(at 100% 0%, rgba(121, 40, 202, 0.12) 0px, transparent 50%),
                    radial-gradient(at 50% 100%, rgba(0, 245, 160, 0.06) 0px, transparent 50%);
                color: #e2e8f0;
            }

            /* Neon Hero Header */
            .hero-neon-banner {
                background: linear-gradient(135deg, rgba(13, 18, 36, 0.85) 0%, rgba(20, 15, 45, 0.85) 50%, rgba(10, 15, 30, 0.9) 100%);
                border: 1px solid rgba(0, 242, 254, 0.35);
                box-shadow: 0 0 25px rgba(0, 242, 254, 0.12), inset 0 0 15px rgba(0, 242, 254, 0.04);
                border-radius: 14px;
                padding: 18px 24px;
                margin-bottom: 20px;
                backdrop-filter: blur(12px);
            }
            .hero-neon-title {
                font-size: 26px;
                font-weight: 800;
                letter-spacing: -0.5px;
                background: linear-gradient(90deg, #00f2fe 0%, #4facfe 35%, #00f5a0 70%, #ffd166 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                text-shadow: 0 0 25px rgba(0, 242, 254, 0.3);
            }
            .hero-neon-subtitle {
                color: #94a3b8;
                font-size: 13.5px;
                margin-top: 6px;
                margin-bottom: 0;
            }

            /* Cyber Tech Cards */
            .cyber-card {
                background: rgba(14, 20, 38, 0.75);
                border: 1px solid rgba(56, 189, 248, 0.22);
                border-radius: 12px;
                padding: 16px 18px;
                margin-bottom: 18px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
                backdrop-filter: blur(8px);
                transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            }
            .cyber-card:hover {
                border-color: rgba(0, 242, 254, 0.45);
                box-shadow: 0 8px 30px rgba(0, 242, 254, 0.15);
            }
            .cyber-card-title {
                font-size: 15px;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                padding-bottom: 8px;
            }

            /* Vibrant Metric Boxes */
            .neon-metric-box {
                background: linear-gradient(180deg, rgba(20, 28, 55, 0.8) 0%, rgba(13, 19, 38, 0.95) 100%);
                border: 1px solid rgba(0, 242, 254, 0.25);
                border-radius: 10px;
                padding: 12px 14px;
                text-align: center;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
                transition: transform 0.2s ease;
            }
            .neon-metric-box:hover {
                transform: translateY(-2px);
                border-color: rgba(0, 242, 254, 0.6);
            }
            .neon-metric-val {
                font-size: 22px;
                font-weight: 800;
                line-height: 1.2;
                letter-spacing: -0.5px;
            }
            .neon-metric-lbl {
                font-size: 11px;
                font-weight: 700;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                margin-top: 4px;
            }

            /* Cyber Status Badges */
            .badge-neon-green {
                background: rgba(0, 245, 160, 0.15);
                color: #00f5a0;
                border: 1px solid #00f5a0;
                box-shadow: 0 0 10px rgba(0, 245, 160, 0.25);
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 11.5px;
                font-weight: 700;
                display: inline-block;
            }
            .badge-neon-amber {
                background: rgba(255, 209, 102, 0.15);
                color: #ffd166;
                border: 1px solid #ffd166;
                box-shadow: 0 0 10px rgba(255, 209, 102, 0.25);
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 11.5px;
                font-weight: 700;
                display: inline-block;
            }
            .badge-neon-red {
                background: rgba(255, 51, 102, 0.15);
                color: #ff3366;
                border: 1px solid #ff3366;
                box-shadow: 0 0 10px rgba(255, 51, 102, 0.25);
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 11.5px;
                font-weight: 700;
                display: inline-block;
            }
            .badge-neon-cyan {
                background: rgba(0, 242, 254, 0.15);
                color: #00f2fe;
                border: 1px solid #00f2fe;
                box-shadow: 0 0 10px rgba(0, 242, 254, 0.25);
                padding: 3px 10px;
                border-radius: 6px;
                font-size: 11.5px;
                font-weight: 700;
                display: inline-block;
            }

            /* Custom callouts */
            .neon-callout {
                background: rgba(15, 23, 42, 0.85);
                border-left: 4px solid #00f2fe;
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 12.5px;
                color: #cbd5e1;
                margin: 8px 0;
            }
            .neon-callout-warn {
                background: rgba(45, 15, 25, 0.8);
                border-left: 4px solid #ff3366;
                border: 1px solid rgba(255, 51, 102, 0.3);
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 12.5px;
                color: #fecdd3;
                margin: 8px 0;
            }
        </style>
        """
    elif theme_name == "👔 Formal Slate Dark":
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            .stApp {
                background-color: #0b1120;
                color: #f1f5f9;
            }

            .hero-neon-banner {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 18px 22px;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            }
            .hero-neon-title {
                font-size: 24px;
                font-weight: 700;
                color: #ffffff;
                margin: 0;
            }
            .hero-neon-subtitle {
                color: #94a3b8;
                font-size: 13px;
                margin-top: 4px;
                margin-bottom: 0;
            }

            .cyber-card {
                background: #162032;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 16px 18px;
                margin-bottom: 18px;
            }
            .cyber-card-title {
                font-size: 14.5px;
                font-weight: 600;
                color: #f8fafc;
                margin-bottom: 12px;
                border-bottom: 1px solid #334155;
                padding-bottom: 6px;
            }

            .neon-metric-box {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 12px 14px;
                text-align: center;
            }
            .neon-metric-val {
                font-size: 20px;
                font-weight: 700;
                color: #38bdf8;
            }
            .neon-metric-lbl {
                font-size: 11px;
                font-weight: 600;
                color: #94a3b8;
                text-transform: uppercase;
                margin-top: 4px;
            }

            .badge-neon-green {
                background: rgba(16, 185, 129, 0.2);
                color: #34d399;
                border: 1px solid #10b981;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }
            .badge-neon-amber {
                background: rgba(245, 158, 11, 0.2);
                color: #fbbf24;
                border: 1px solid #f59e0b;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }
            .badge-neon-red {
                background: rgba(239, 68, 68, 0.2);
                color: #f87171;
                border: 1px solid #ef4444;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }
            .badge-neon-cyan {
                background: rgba(56, 189, 248, 0.2);
                color: #38bdf8;
                border: 1px solid #0284c7;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }

            .neon-callout {
                background: #1e293b;
                border-left: 4px solid #38bdf8;
                padding: 10px 14px;
                border-radius: 4px;
                font-size: 12.5px;
                color: #cbd5e1;
            }
            .neon-callout-warn {
                background: rgba(239, 68, 68, 0.1);
                border-left: 4px solid #ef4444;
                padding: 10px 14px;
                border-radius: 4px;
                font-size: 12.5px;
                color: #fca5a5;
            }
        </style>
        """
    else:  # 🏛️ Formal Enterprise Light
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            .stApp {
                background-color: #f8fafc;
                color: #0f172a;
            }

            .hero-neon-banner {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 18px 22px;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            }
            .hero-neon-title {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
                margin: 0;
            }
            .hero-neon-subtitle {
                color: #64748b;
                font-size: 13px;
                margin-top: 4px;
                margin-bottom: 0;
            }

            .cyber-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 16px 18px;
                margin-bottom: 18px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
            }
            .cyber-card-title {
                font-size: 14.5px;
                font-weight: 600;
                color: #0f172a;
                margin-bottom: 12px;
                border-bottom: 1px solid #f1f5f9;
                padding-bottom: 6px;
            }

            .neon-metric-box {
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px 14px;
                text-align: center;
            }
            .neon-metric-val {
                font-size: 20px;
                font-weight: 700;
                color: #0284c7;
            }
            .neon-metric-lbl {
                font-size: 11px;
                font-weight: 600;
                color: #64748b;
                text-transform: uppercase;
                margin-top: 4px;
            }

            .badge-neon-green {
                background: rgba(16, 185, 129, 0.15);
                color: #047857;
                border: 1px solid #10b981;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }
            .badge-neon-amber {
                background: rgba(245, 158, 11, 0.15);
                color: #b45309;
                border: 1px solid #f59e0b;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }
            .badge-neon-red {
                background: rgba(239, 68, 68, 0.15);
                color: #b91c1c;
                border: 1px solid #ef4444;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }
            .badge-neon-cyan {
                background: rgba(2, 132, 199, 0.15);
                color: #0369a1;
                border: 1px solid #0284c7;
                padding: 3px 9px;
                border-radius: 4px;
                font-size: 11.5px;
                font-weight: 600;
            }

            .neon-callout {
                background: #f8fafc;
                border-left: 4px solid #0284c7;
                border: 1px solid #e2e8f0;
                padding: 10px 14px;
                border-radius: 4px;
                font-size: 12.5px;
                color: #334155;
            }
            .neon-callout-warn {
                background: #fff1f2;
                border-left: 4px solid #ef4444;
                border: 1px solid #fecdd3;
                padding: 10px 14px;
                border-radius: 4px;
                font-size: 12.5px;
                color: #9f1239;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONTROLS, ACTIVE SHIPMENT & THEME PICKER
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 28px;">🌱</span>
        <div>
            <div style="font-size: 18px; font-weight: 800; background: linear-gradient(90deg, #00f2fe, #00f5a0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Freshroute</div>
            <div style="font-size: 10.5px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Cold Chain Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. NAVIGATION SELECTION
    st.markdown('<div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">📍 Navigation Pages</div>', unsafe_allow_html=True)
    
    pages = [
        "📷 Camera Verification",
        "📡 ESP32 IoT Detection",
        "🗺️ Route Optimization",
        "📊 Complete Report"
    ]
    
    current_page = st.radio(
        "Select Active Page",
        pages,
        index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
        label_visibility="collapsed"
    )
    st.session_state.current_page = current_page

    st.markdown("---")

    # 2. ACTIVE CARGO SHIPMENT SELECTOR
    st.markdown('<div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">📦 Active Cargo Shipment</div>', unsafe_allow_html=True)
    shipment_list = list(st.session_state.shipments.keys())
    selected_id = st.selectbox("Select Cargo", shipment_list, index=shipment_list.index(st.session_state.active_shipment_id), label_visibility="collapsed")
    st.session_state.active_shipment_id = selected_id
    curr_shipment = st.session_state.shipments[selected_id]

    with st.expander("➕ Register New Cargo Shipment", expanded=False):
        new_id = st.text_input("Shipment ID", f"SH{len(st.session_state.shipments)+1:03d}")
        new_farmer = st.text_input("Farmer / Origin", "GreenValley Agro")
        new_truck = st.text_input("Truck Reg", "TN-01-AB-1234")
        new_crop = st.selectbox("Cargo Crop", list(CROP_DATABASE.keys()))
        new_qty = st.number_input("Payload Quantity (kg)", min_value=100, max_value=50000, value=3000, step=100)
        
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

    # 3. DISPLAY THEME TOGGLE
    st.markdown('<div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">🎨 UI Theme & Style</div>', unsafe_allow_html=True)
    selected_theme = st.selectbox(
        "Theme Palette",
        ["⚡ Vibrant Cyber Neon", "👔 Formal Slate Dark", "🏛️ Formal Enterprise Light"],
        index=["⚡ Vibrant Cyber Neon", "👔 Formal Slate Dark", "🏛️ Formal Enterprise Light"].index(st.session_state.ui_theme)
    )
    st.session_state.ui_theme = selected_theme

    st.markdown("---")
    auto_refresh = st.checkbox("⚡ Live Sensor Polling (2s)", value=False)
    st.caption("Freshroute Agro-Intelligence v2.5")

# Inject chosen theme styles
inject_theme_css(st.session_state.ui_theme)


# ==============================================================================
# CALCULATE CURRENT SHELF LIFE & ROUTING STATE
# ==============================================================================
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

routing_result = fefo_engine.optimize_vendor_route(
    crop_name=curr_shipment["crop"],
    shipment_quantity_kg=curr_shipment["quantity_kg"],
    rsl_final_days=shelf_res["RSL_final_days"],
    truck_lat=curr_shipment.get("truck_lat", 13.0400),
    truck_lon=curr_shipment.get("truck_lon", 80.1200),
    shipment_id=curr_shipment["id"]
)
best_vendor = routing_result["recommended_vendor"]


# ==============================================================================
# MAIN HERO COMMAND BANNER
# ==============================================================================
crop_icon = CROP_DATABASE.get(curr_shipment["crop"], {}).get("icon", "📦")
status_badge_class = "badge-neon-green" if curr_shipment["condition"].lower() == "fresh" else "badge-neon-red"

st.markdown(f"""
<div class="hero-neon-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div>
            <h1 class="hero-neon-title">🌱 Freshroute Command Center</h1>
            <p class="hero-neon-subtitle">
                Cargo <b>{curr_shipment['id']}</b> &nbsp;|&nbsp; 
                Origin: <b>{curr_shipment['farmer']}</b> &nbsp;|&nbsp; 
                Truck: <b>{curr_shipment['truck_id']}</b> &nbsp;|&nbsp; 
                Payload: <b>{curr_shipment['quantity_kg']:,} kg {curr_shipment['crop']} {crop_icon}</b>
            </p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
            <span class="{status_badge_class}">Quality: {curr_shipment['condition'].upper()} ({curr_shipment['confidence']}%)</span>
            <span class="badge-neon-cyan">RSL: {shelf_res['RSL_final_days']:.2f} Days</span>
            <span class="badge-neon-amber">Next: {best_vendor['vendor_name'] if best_vendor else 'Hub'}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: 📷 CAMERA QUALITY VERIFICATION
# ==============================================================================
if st.session_state.current_page == "📷 Camera Verification":
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="cyber-card-title">
        <span>📷 Camera Quality Verification & MobileNetV2 Deep Learning HUD</span>
    </div>
    """, unsafe_allow_html=True)

    vision_mode = st.radio(
        "Select Camera Input Source",
        [
            "🎥 Integrated Laptop Camera (Live Stream & OpenCV)",
            "📱 Phone / Browser Camera (Direct Snapshot)",
            "📁 Upload Cargo Photo",
            "🍎 Sample Test Specimens"
        ],
        index=0,
        horizontal=True
    )

    current_frame = None
    v_col1, v_col2 = st.columns([1.3, 1.0])

    with v_col1:
        if vision_mode == "🎥 Integrated Laptop Camera (Live Stream & OpenCV)":
            if st.session_state.detected_cameras is None:
                st.session_state.detected_cameras = detector.scan_available_cameras(4)

            found_cams = st.session_state.detected_cameras or []
            if found_cams:
                cam_choices = [c["index"] for c in found_cams]
                cam_labels = {c["index"]: c["label"] for c in found_cams}
                default_idx = next((c["index"] for c in found_cams if not c.get("is_black", False)), cam_choices[0])
            else:
                cam_choices = [0, 1]
                cam_labels = {0: "Camera 0: Integrated Laptop Camera (DirectShow | 640x480)", 1: "Camera 1: Secondary Device"}
                default_idx = 0

            c_cfg1, c_cfg2, c_cfg3 = st.columns([1.5, 1.0, 0.8])
            with c_cfg1:
                selected_cam_idx = st.selectbox(
                    "Select Hardware Camera Device",
                    cam_choices,
                    index=cam_choices.index(default_idx) if default_idx in cam_choices else 0,
                    format_func=lambda x: cam_labels.get(x, f"Camera {x}")
                )
            with c_cfg2:
                cam_backend = st.selectbox(
                    "Capture Backend",
                    ["AUTO", "DSHOW", "MSMF", "DEFAULT"],
                    index=0,
                    help="AUTO tries DirectShow (fastest on Windows), then MediaFoundation, then Default."
                )
            with c_cfg3:
                warmup_f = st.slider("Warmup Frames", 5, 20, 10, help="Discards initial dark frames.")

            # Live Stream Toggle & Status
            v_t1, v_t2 = st.columns([1.2, 1.8])
            with v_t1:
                stream_toggle = st.toggle("🔴 Continuous Live Stream", value=False, help="Streams live webcam feed continuously inside the dashboard.")
            with v_t2:
                st.caption("🟢 **Camera Ready:** Use **'Snap & Inspect'** or **'Continuous Live Stream'**.")

            # Action Buttons Row
            b1, b2, b3, b4 = st.columns([1.1, 1.1, 1.4, 0.9])
            with b1:
                capture_single = st.button("📸 Snap & Inspect", use_container_width=True, help="Capture single frame and run CNN quality analysis.")
            with b2:
                start_live = st.button("▶️ Start Live Stream", use_container_width=True, help="Stream live video with real-time CNN HUD inside dashboard.")
            with b3:
                launch_native = st.button("🚀 Launch High-FPS Window", use_container_width=True, help="Launch standalone 60 FPS OpenCV hardware window.")
            with b4:
                if st.button("🔍 Re-scan", use_container_width=True):
                    with st.spinner("Scanning cameras..."):
                        st.session_state.detected_cameras = detector.scan_available_cameras(4)
                        st.toast(f"Found {len(st.session_state.detected_cameras or [])} devices!", icon="📹")
                        st.rerun()

            # Handle Standalone Native Window
            if launch_native:
                try:
                    subprocess.Popen([sys.executable, "live_webcam_vision.py"])
                    st.toast("🚀 Standalone 60 FPS OpenCV Live Window launched! Check your Windows taskbar.", icon="🎥")
                    st.markdown("""
                    <div class="neon-callout">
                        🎥 <b>Standalone Live OpenCV Window is Running!</b><br>
                        • Controls in OpenCV window: <b>[S]</b> Save inspection snapshot, <b>[C]</b> Cycle camera index, <b>[Q/ESC]</b> Close.<br>
                        • Telemetry from ESP32 is rendered directly on the bottom HUD.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as ex:
                    st.toast(f"Failed to launch standalone window: {ex}", icon="⚠️")

            # Handle Single Frame Snapshot
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
                        st.markdown(f'<div class="neon-callout-warn">{msg}</div>', unsafe_allow_html=True)
                    else:
                        st.toast("📸 Frame captured successfully from Live Camera!", icon="✅")
                else:
                    st.markdown(f"""
                    <div class="neon-callout-warn">
                        <b>⚠️ Camera Notice:</b> {msg}<br>
                        <i>Tip: Check camera permissions or choose another backend/camera device.</i>
                    </div>
                    """, unsafe_allow_html=True)
                    sample_path = "sample_images/fresh_banana.jpg" if curr_shipment["crop"] == "Banana" else "sample_images/fresh_apple.jpg"
                    if os.path.exists(sample_path):
                        current_frame = cv2.imread(sample_path)
                        st.session_state.last_captured_frame = current_frame

            # Handle Live Stream inside Streamlit
            if start_live or stream_toggle:
                video_placeholder = st.empty()
                status_placeholder = st.empty()
                cap, b_used = detector.open_video_capture(selected_cam_idx, cam_backend)
                
                if cap is None or not cap.isOpened():
                    st.markdown(f'<div class="neon-callout-warn">⚠️ Unable to open camera (Index {selected_cam_idx} via {b_used}).</div>', unsafe_allow_html=True)
                else:
                    try:
                        for _ in range(warmup_f):
                            cap.read()
                            time.sleep(0.01)

                        status_placeholder.markdown("""
                        <div style="background: rgba(255, 51, 102, 0.15); border: 1px solid #ff3366; border-radius: 6px; padding: 6px 12px; margin-bottom: 8px; font-size: 12px; color: #fecdd3; display: flex; align-items: center; justify-content: space-between;">
                            <span>🔴 <b>LIVE CAMERA STREAM ACTIVE</b> (Real-time MobileNetV2 Quality HUD)</span>
                            <span style="font-size: 11px; color: #cbd5e1;">Live Feed Running</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.toast("📹 Live OpenCV inspection stream active...", icon="🔴")
                        
                        max_frames = 300 if stream_toggle else 120
                        last_pred = detector.predict(None)
                        for frame_idx in range(max_frames):
                            ret, f = cap.read()
                            if not ret or f is None:
                                break
                            
                            if frame_idx % 2 == 0:
                                last_pred = detector.predict(f)

                            ann = detector.annotate_frame(f, last_pred, curr_shipment["id"])
                            disp = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                            
                            video_placeholder.image(
                                disp,
                                caption=f"Live CNN Stream: {last_pred['crop']} - {last_pred['condition']} ({last_pred['confidence']}%) | Frame {frame_idx+1}/{max_frames}",
                                use_container_width=True
                            )
                            
                            st.session_state.last_cnn_result = last_pred
                            st.session_state.last_captured_frame = f
                            curr_shipment["crop"] = last_pred["crop"]
                            curr_shipment["condition"] = last_pred["condition"]
                            curr_shipment["confidence"] = last_pred["confidence"]
                            time.sleep(0.02)
                    except Exception as ex:
                        st.toast(f"Stream error: {ex}", icon="⚠️")
                    finally:
                        if cap is not None and cap.isOpened():
                            cap.release()
                        status_placeholder.empty()

        elif vision_mode == "📱 Phone / Browser Camera (Direct Snapshot)":
            st.markdown("""
            <div class="neon-callout">
                📱 <b>Phone / Browser Camera Active:</b> Tap <b>"Take Photo"</b> below to capture fruit / vegetable cargo with your camera for instant CNN quality verification.
            </div>
            """, unsafe_allow_html=True)
            cam_pic = st.camera_input("📸 Tap to capture fruit / vegetable cargo:")
            if cam_pic is not None:
                bytes_data = cam_pic.getvalue()
                current_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                st.session_state.last_captured_frame = current_frame

        elif vision_mode == "📁 Upload Cargo Photo":
            uploaded_file = st.file_uploader("Upload Cargo Specimen Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                bytes_data = uploaded_file.read()
                current_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                st.session_state.last_captured_frame = current_frame

        elif vision_mode == "🍎 Sample Test Specimens":
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

        # Display annotated frame
        if current_frame is not None:
            cnn_pred = detector.predict(current_frame)
            st.session_state.last_cnn_result = cnn_pred
            annotated_frame = detector.annotate_frame(current_frame, cnn_pred, curr_shipment["id"])
            disp_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(disp_rgb, caption=f"Visual Verification HUD: {cnn_pred['crop']} - {cnn_pred['condition']} ({cnn_pred['confidence']}%)", use_container_width=True)
        else:
            cnn_pred = st.session_state.last_cnn_result
            st.info("💡 Select a camera input source or test specimen above to run real-time visual inspection.")

    # Quality Inspection Summary & Softmax Bar Chart
    with v_col2:
        is_fresh = cnn_pred["condition"].lower() == "fresh"
        v_badge = "badge-neon-green" if is_fresh else "badge-neon-red"
        v_status = "PASSED - FRESH" if is_fresh else "FAILED - ROTTEN"
        crop_name = cnn_pred.get("crop", curr_shipment["crop"])
        crop_ico = CROP_DATABASE.get(crop_name, {}).get("icon", "📦")

        st.markdown(f"""
        <div style="background: rgba(20, 28, 55, 0.9); border: 1.5px solid rgba(0, 242, 254, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span style="font-size: 20px; font-weight: 800; color: #f8fafc;">{crop_ico} {crop_name}</span>
                    <div style="font-size: 11.5px; color: #94a3b8;">MobileNetV2 Visual Classifier</div>
                </div>
                <div>
                    <span class="{v_badge}">{v_status}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                <div>
                    <div style="font-size: 10px; color: #94a3b8;">DETECTED CLASS</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f1f5f9;">{cnn_pred.get('class_name', 'fresh').replace('_', ' ').title()}</div>
                </div>
                <div>
                    <div style="font-size: 10px; color: #94a3b8;">CNN CONFIDENCE</div>
                    <div style="font-size: 14px; font-weight: 700; color: #00f2fe;">{cnn_pred.get('confidence', 95.0)}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Softmax Probability Distribution Bar Chart
        if "probabilities" in cnn_pred and cnn_pred["probabilities"]:
            st.markdown("<b>📊 Softmax Class Probability Distribution:</b>", unsafe_allow_html=True)
            prob_df = pd.DataFrame([
                {"Class": k.replace("_", " ").title(), "Probability (%)": v}
                for k, v in cnn_pred["probabilities"].items()
            ])
            fig_prob = px.bar(
                prob_df, x="Probability (%)", y="Class", orientation="h",
                color="Probability (%)",
                color_continuous_scale=[[0, "#1e293b"], [0.5, "#0284c7"], [1.0, "#00f5a0"]],
                height=170
            )
            fig_prob.update_layout(
                margin=dict(l=6, r=6, t=6, b=6),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", size=10.5),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", range=[0, 100]),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        if st.button("✅ Apply Inspection Result to Cargo Shipment", use_container_width=True):
            curr_shipment["crop"] = cnn_pred["crop"]
            curr_shipment["condition"] = cnn_pred["condition"]
            curr_shipment["confidence"] = cnn_pred["confidence"]
            st.toast(f"✅ Synced {cnn_pred['crop']} ({cnn_pred['condition']}) to Shipment {curr_shipment['id']}!", icon="🔄")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: 📡 ESP32 DETECTION & IOT TELEMETRY
# ==============================================================================
elif st.session_state.current_page == "📡 ESP32 IoT Detection":
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="cyber-card-title">
        <span>📡 ESP32 Edge Hardware Link & Environmental Telemetry Stream</span>
    </div>
    """, unsafe_allow_html=True)

    e_col1, e_col2 = st.columns([1, 1.4])

    with e_col1:
        st.markdown("<b>🔌 Hardware Telemetry Source:</b>", unsafe_allow_html=True)
        conn_mode = st.radio(
            "Connection Protocol",
            ["Simulator (Real-time Dynamic Stream)", "Physical Serial (USB/COM)", "Wi-Fi HTTP / IP Stream", "Manual Telemetry Injection"],
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
                    if st.button("🔌 Connect Serial", use_container_width=True):
                        ok, msg = sensor_mgr.connect_serial(selected_port, baud)
                        if ok:
                            st.toast(f"✅ {msg}", icon="🔌")
                        else:
                            st.toast(f"❌ {msg}", icon="⚠️")
                with col_c2:
                    if st.button("❌ Disconnect", use_container_width=True):
                        sensor_mgr.disconnect_serial()
                        st.toast("Serial disconnected.", icon="ℹ️")
                
                if sensor_mgr.is_connected:
                    st.markdown(f'<span class="badge-neon-green">🟢 ESP32 Active: {sensor_mgr.serial_port} @ {sensor_mgr.baud_rate}</span>', unsafe_allow_html=True)
                    s_reading = sensor_mgr.read_serial_line()
                    if s_reading:
                        curr_shipment["readings"].append(s_reading.to_dict())
                else:
                    st.markdown('<span class="badge-neon-cyan">⚪ Status: Disconnected</span>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="neon-callout">
                    ⚠️ No physical serial COM ports found. Connect ESP32 via USB or select <b>Simulator</b>.
                </div>
                """, unsafe_allow_html=True)

        elif conn_mode == "Wi-Fi HTTP / IP Stream":
            esp32_url = st.text_input("ESP32 JSON Endpoint URL", "http://192.168.1.100/data")
            if st.button("📡 Test Ping Endpoint", use_container_width=True):
                r, msg = sensor_mgr.fetch_http_reading(esp32_url, timeout=1.5)
                if r:
                    st.toast(f"📡 Ping OK: {r.temperature_C}°C, {r.humidity_RH}% RH", icon="✅")
                    curr_shipment["readings"].append(r.to_dict())
                else:
                    st.toast(f"❌ Ping Failed: {msg}", icon="⚠️")
            st.caption("Expected JSON: `{\"temperature\": 24.5, \"humidity\": 71.8}`")

        elif conn_mode == "Simulator (Real-time Dynamic Stream)":
            sim_anomaly = st.selectbox(
                "Environmental Stress Simulation Scenario",
                ["NORMAL", "COOLING_FAILURE", "HEATWAVE", "HIGH_HUMIDITY"],
                format_func=lambda x: {
                    "NORMAL": "🟢 Normal Cold-Chain Compliance (24°C / 72% RH)",
                    "COOLING_FAILURE": "🔴 Cooling Unit Failure (Rapid Warming)",
                    "HEATWAVE": "🟠 High Heatwave Exposure (Thermal Spike)",
                    "HIGH_HUMIDITY": "🔵 Excessive Moisture / Condensation Risk"
                }[x]
            )
            sensor_mgr.sim_anomaly = sim_anomaly
            sensor_mgr.sim_base_temp = st.slider("Base Temperature (°C)", 0.0, 45.0, 24.0, 0.5)
            sensor_mgr.sim_base_humidity = st.slider("Base Relative Humidity (%)", 30.0, 98.0, 72.0, 1.0)
            
            sim_reading = sensor_mgr.generate_simulated_reading()
            curr_shipment["readings"].append(sim_reading.to_dict())
            if len(curr_shipment["readings"]) > 120:
                curr_shipment["readings"] = curr_shipment["readings"][-120:]

        elif conn_mode == "Manual Telemetry Injection":
            man_t = st.slider("Manual Temp (°C)", -5.0, 50.0, 28.5, 0.1)
            man_h = st.slider("Manual RH (%)", 20.0, 100.0, 68.0, 0.5)
            if st.button("💉 Inject Single Reading", use_container_width=True):
                r = sensor_mgr.get_latest_reading(curr_shipment["id"], manual_temp=man_t, manual_hum=man_h)
                curr_shipment["readings"].append(r.to_dict())
                st.toast(f"💉 Injected Reading: {man_t}°C, {man_h}% RH", icon="⚡")
                st.rerun()

    with e_col2:
        history = curr_shipment.get("readings", [])
        latest_r = history[-1] if history else {"temperature_C": 24.5, "humidity_RH": 72.0}

        crop_meta = engine.get_crop_params(curr_shipment["crop"])
        t_opt_min = crop_meta.get("T_opt_min", 12.0)
        t_opt_max = crop_meta.get("T_opt_max", 20.0)
        rh_opt_min = crop_meta.get("RH_opt_min", 85.0)
        rh_opt_max = crop_meta.get("RH_opt_max", 95.0)

        # 4 Metric Cards Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            t_val = latest_r["temperature_C"]
            t_color = "#00f5a0" if t_opt_min <= t_val <= t_opt_max else ("#ff3366" if t_val > 30.0 else "#ffd166")
            st.markdown(f"""
            <div class="neon-metric-box">
                <div class="neon-metric-val" style="color: {t_color};">{t_val:.1f} °C</div>
                <div class="neon-metric-lbl">Temperature</div>
                <div style="font-size: 9.5px; color: #94a3b8; margin-top: 2px;">Opt: {t_opt_min}-{t_opt_max}°C</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            h_val = latest_r["humidity_RH"]
            h_color = "#00f5a0" if rh_opt_min <= h_val <= rh_opt_max else ("#ff3366" if h_val < 50.0 else "#ffd166")
            st.markdown(f"""
            <div class="neon-metric-box">
                <div class="neon-metric-val" style="color: {h_color};">{h_val:.1f} %</div>
                <div class="neon-metric-lbl">Humidity</div>
                <div style="font-size: 9.5px; color: #94a3b8; margin-top: 2px;">Opt: {rh_opt_min}-{rh_opt_max}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            q10_rate, _ = engine.calculate_physics_step(t_val, crop_meta)
            st.markdown(f"""
            <div class="neon-metric-box">
                <div class="neon-metric-val" style="color: #00f2fe;">{q10_rate:.2f}x</div>
                <div class="neon-metric-lbl">Degradation r(T)</div>
                <div style="font-size: 9.5px; color: #94a3b8; margin-top: 2px;">Q10: {crop_meta['Q10']}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            # Vapor Pressure Deficit (VPD in kPa)
            svp = 0.61078 * np.exp((17.27 * t_val) / (t_val + 237.3))
            avp = svp * (h_val / 100.0)
            vpd = max(0.0, svp - avp)
            st.markdown(f"""
            <div class="neon-metric-box">
                <div class="neon-metric-val" style="color: #a78bfa;">{vpd:.2f} kPa</div>
                <div class="neon-metric-lbl">VPD Deficit</div>
                <div style="font-size: 9.5px; color: #94a3b8; margin-top: 2px;">Transpiration</div>
            </div>
            """, unsafe_allow_html=True)

        # Dual-Axis Sensor History Plotly Chart
        if len(history) > 0:
            hist_df = pd.DataFrame(history)
            hist_df["step"] = range(1, len(hist_df) + 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["step"], y=hist_df["temperature_C"],
                name="Temperature (°C)", mode="lines+markers",
                line=dict(color="#ff3366", width=2.5, shape="spline"),
                marker=dict(size=4),
                hovertemplate="Step %{x}: %{y:.1f} °C<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=hist_df["step"], y=hist_df["humidity_RH"],
                name="Humidity (% RH)", mode="lines+markers",
                line=dict(color="#00f2fe", width=2.5, shape="spline"),
                marker=dict(size=4),
                yaxis="y2",
                hovertemplate="Step %{x}: %{y:.1f} %% RH<extra></extra>"
            ))

            fig.add_hrect(
                y0=t_opt_min, y1=t_opt_max, fillcolor="rgba(0, 245, 160, 0.12)",
                line_width=0, annotation_text="Optimal Temp Band", annotation_position="top left",
                annotation_font=dict(size=9, color="#00f5a0")
            )

            fig.update_layout(
                height=230,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", size=10.5),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                yaxis=dict(title="Temp (°C)", color="#ff3366", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                yaxis2=dict(title="RH (%)", color="#00f2fe", overlaying="y", side="right", showgrid=False),
                xaxis=dict(title="Telemetry Sensor Intervals (Hourly/Steps)", showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Telemetry Log Table & Actions
    with st.expander(f"📋 Sensor Telemetry Raw Data Stream Log ({len(history)} entries)", expanded=False):
        if history:
            log_df = pd.DataFrame(history)
            st.dataframe(log_df.tail(20), use_container_width=True)
            
            c_act1, c_act2 = st.columns(2)
            with c_act1:
                csv_bytes = log_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Telemetry to CSV",
                    data=csv_bytes,
                    file_name=f"telemetry_{curr_shipment['id']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c_act2:
                if st.button("🗑️ Reset Sensor Log Buffer", use_container_width=True):
                    curr_shipment["readings"] = [
                        {"temperature_C": 24.5, "humidity_RH": 72.0, "delta_t_days": 1.0/24.0}
                    ]
                    st.toast("Sensor log buffer cleared!", icon="🧹")
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: 🗺️ ROUTE OPTIMIZATION & FEFO LOGISTICS
# ==============================================================================
elif st.session_state.current_page == "🗺️ Route Optimization":
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="cyber-card-title">
        <span>🗺️ Geospatial Route Optimization & Multi-Shipment FEFO Dispatch Queue</span>
    </div>
    """, unsafe_allow_html=True)

    r_col1, r_col2 = st.columns([1, 1.4])

    with r_col1:
        # FEFO Dispatch Priority Queue
        st.markdown("<b>⚡ FEFO Multi-Shipment Priority Dispatch Queue:</b>", unsafe_allow_html=True)
        all_shipments = list(st.session_state.shipments.values())
        fefo_ranked = fefo_engine.rank_fefo_queue(all_shipments)

        fefo_table_data = []
        for s in fefo_ranked:
            is_active = "👉 " if s["id"] == curr_shipment["id"] else ""
            fefo_table_data.append({
                "Rank": f"#{s['fefo_rank']}",
                "Shipment": f"{is_active}{s['id']}",
                "Crop": s["crop"],
                "Condition": s["condition"],
                "RSL (Days)": f"{s.get('RSL_final_days', 0.0):.2f}d",
                "Action": s["dispatch_urgency"]
            })

        fefo_df = pd.DataFrame(fefo_table_data)
        st.dataframe(fefo_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="font-size: 11.5px; color: #94a3b8; margin: 6px 0 14px 0;">
            💡 <b>FEFO Principle:</b> Batches with lowest Remaining Shelf Life are prioritized for immediate dispatch to eliminate in-transit spoilage.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Truck GPS Simulator
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

    with r_col2:
        # Recommended Vendor Highlight Box
        if best_vendor:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(14, 20, 38, 0.95), rgba(20, 30, 60, 0.9)); border: 1.5px solid #00f5a0; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(0, 245, 160, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span class="badge-neon-green">🎯 RECOMMENDED NEXT DESTINATION</span>
                        <h3 style="margin: 6px 0 2px 0; color: #f8fafc; font-size: 18px;">{best_vendor['vendor_name']}</h3>
                        <p style="margin: 0; font-size: 11.5px; color: #cbd5e1;">
                            🏬 {best_vendor['type']} &nbsp;|&nbsp; 📍 {best_vendor['city']} &nbsp;|&nbsp; 📞 {best_vendor['contact']} ({best_vendor['manager']})
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span class="badge-neon-cyan">₹{best_vendor['price_per_kg']}/kg</span>
                        <div style="font-size: 18px; font-weight: 800; color: #00f5a0; margin-top: 2px;">Net: ₹{best_vendor['net_profit']:,}</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">DISTANCE</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #f1f5f9;">{best_vendor['distance_km']} km</div>
                    </div>
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">TRANSIT ETA</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #00f2fe;">~{best_vendor['transit_minutes']} Mins</div>
                    </div>
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">SHOP DEMAND</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #ffd166;">{best_vendor['demand_kg']:,} kg</div>
                    </div>
                    <div>
                        <div style="font-size: 9.5px; color: #94a3b8;">SHELF BUFFER</div>
                        <div style="font-size: 13.5px; font-weight: 700; color: #00f5a0;">+{best_vendor['margin_hours']}h</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Geospatial Map
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
            "color": "#00f2fe",
            "details": f"Cargo: {curr_shipment['quantity_kg']}kg {active_crop} | Status: In Transit"
        })

        for v in VENDOR_DATABASE:
            is_best = (best_vendor and v["id"] == best_vendor["vendor_id"])
            d_crop = v["demands_kg"].get(active_crop, 0)
            p_crop = v["prices_per_kg"].get(active_crop, 0.0)
            
            size_scaled = int(max(10, min(28, np.sqrt(d_crop) * 0.35)))
            if is_best:
                size_scaled = max(size_scaled, 22)

            cat = "🎯 Recommended Next Stop" if is_best else ("🏪 Active Demand Shop" if d_crop > 0 else "🏬 Other Retailer")
            color = "#00f5a0" if is_best else ("#ffd166" if d_crop > 0 else "#64748b")

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
                line=dict(width=4, color="#00f5a0"),
                name="Optimal Navigation Corridor",
                hoverinfo="text",
                text=f"Navigation Corridor ➡️ {best_vendor['vendor_name']} ({best_vendor['distance_km']} km, ~{best_vendor['transit_minutes']} mins)"
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
            style="carto-darkmatter" if "Dark" in st.session_state.ui_theme or "Neon" in st.session_state.ui_theme else "carto-positron",
            center=dict(lat=avg_lat, lon=avg_lon),
            zoom=10.2
        )

        if hasattr(go, "Scattermap"):
            fig_map.update_layout(map=map_layout_kwargs)
        else:
            fig_map.update_layout(mapbox=map_layout_kwargs)

        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
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

    # Nearby Shops Demand Matrix
    with st.expander(f"📊 Nearby Retail & Mandi Demand Matrix ({active_crop})", expanded=False):
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


# ==============================================================================
# PAGE 4: 📊 COMPLETE INTELLIGENCE REPORT & ANALYTICS
# ==============================================================================
elif st.session_state.current_page == "📊 Complete Report":
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="cyber-card-title">
        <span>📊 Comprehensive Cargo Intelligence & Shelf-Life Assessment Report</span>
    </div>
    """, unsafe_allow_html=True)

    # 1. EXECUTIVE SUMMARY DOSSIER
    rep_c1, rep_c2, rep_c3 = st.columns([1.2, 1.2, 1.0])
    
    with rep_c1:
        st.markdown(f"""
        <div style="background: rgba(20, 28, 55, 0.8); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 10px; padding: 14px; height: 100%;">
            <div style="font-size: 11px; font-weight: 700; color: #00f2fe; text-transform: uppercase;">📦 Cargo Logistics Dossier</div>
            <h3 style="margin: 4px 0 8px 0; color: #f8fafc;">{curr_shipment['id']} — {curr_shipment['crop']}</h3>
            <div style="font-size: 12px; color: #cbd5e1; line-height: 1.6;">
                • <b>Origin / Farm:</b> {curr_shipment['farmer']}<br>
                • <b>Truck Reg:</b> {curr_shipment['truck_id']}<br>
                • <b>Payload:</b> {curr_shipment['quantity_kg']:,} kg<br>
                • <b>Dispatch Date:</b> {curr_shipment.get('load_time', '2026-08-29 06:30')}<br>
                • <b>Current GPS:</b> ({curr_shipment.get('truck_lat', 13.04):.4f}, {curr_shipment.get('truck_lon', 80.12):.4f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rep_c2:
        is_fresh = curr_shipment["condition"].lower() == "fresh"
        c_badge = "badge-neon-green" if is_fresh else "badge-neon-red"
        st.markdown(f"""
        <div style="background: rgba(20, 28, 55, 0.8); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 10px; padding: 14px; height: 100%;">
            <div style="font-size: 11px; font-weight: 700; color: #00f5a0; text-transform: uppercase;">👁️ Visual Quality & Intake Verification</div>
            <div style="margin: 6px 0 8px 0;">
                <span class="{c_badge}">{curr_shipment['condition'].upper()} ({curr_shipment['confidence']}%)</span>
            </div>
            <div style="font-size: 12px; color: #cbd5e1; line-height: 1.6;">
                • <b>Classification Model:</b> MobileNetV2 CNN<br>
                • <b>Inference Class:</b> {st.session_state.last_cnn_result.get('class_name', 'fresh').replace('_', ' ').title()}<br>
                • <b>Visual Status:</b> {'Intact Epidermis & Normal Turgor' if is_fresh else 'Tissue Softening & Necrosis'}<br>
                • <b>Cold-Chain Risk:</b> {'Low Spoilage Propensity' if is_fresh else 'High Bacterial Acceleration'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rep_c3:
        st.markdown(f"""
        <div style="background: rgba(20, 28, 55, 0.8); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 10px; padding: 14px; height: 100%;">
            <div style="font-size: 11px; font-weight: 700; color: #ffd166; text-transform: uppercase;">🎯 Dispatch Target</div>
            <h4 style="margin: 4px 0 6px 0; color: #f8fafc;">{best_vendor['vendor_name'] if best_vendor else 'Central Mandi'}</h4>
            <div style="font-size: 12px; color: #cbd5e1; line-height: 1.6;">
                • <b>Distance:</b> {best_vendor['distance_km'] if best_vendor else 12} km<br>
                • <b>Transit ETA:</b> ~{best_vendor['transit_minutes'] if best_vendor else 30} mins<br>
                • <b>Net Projected Profit:</b> ₹{best_vendor['net_profit'] if best_vendor else 0:,}<br>
                • <b>Buffer Margin:</b> +{best_vendor['margin_hours'] if best_vendor else 48}h
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. SHELF-LIFE HYBRID MATH DECOMPOSITION
    st.markdown("<b>🧠 Hybrid Physics (Q₁₀) + XGBoost ML Shelf-Life Decomposition:</b>", unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="neon-metric-box">
            <div class="neon-metric-val" style="color: #a5b4fc;">{shelf_res['S0_days']:.1f} d</div>
            <div class="neon-metric-lbl">Baseline Budget (S₀)</div>
            <div style="font-size: 9.5px; color: #94a3b8;">Harvest Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="neon-metric-box">
            <div class="neon-metric-val" style="color: #00f2fe;">{shelf_res['RSL_physics_days']:.2f} d</div>
            <div class="neon-metric-lbl">Physics RSL (S₀ - D)</div>
            <div style="font-size: 9.5px; color: #94a3b8;">Loss D = {shelf_res['cumulative_degradation_D']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        d_sgn = "+" if shelf_res['delta_RSL_ML_days'] >= 0 else ""
        st.markdown(f"""
        <div class="neon-metric-box">
            <div class="neon-metric-val" style="color: #ffd166;">{d_sgn}{shelf_res['delta_RSL_ML_days']:.2f} d</div>
            <div class="neon-metric-lbl">XGBoost ML Correction (Δ)</div>
            <div style="font-size: 9.5px; color: #94a3b8;">Non-linear Stress</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="neon-metric-box" style="border-color: {shelf_res['risk_color']};">
            <div class="neon-metric-val" style="color: {shelf_res['risk_color']}; font-size: 22px;">{shelf_res['RSL_final_days']:.2f} Days</div>
            <div class="neon-metric-lbl" style="color: {shelf_res['risk_color']};">Final Shelf Life</div>
            <div style="font-size: 9.5px; color: {shelf_res['risk_color']}; font-weight: 700;">{shelf_res['RSL_final_hours']} Hours ({shelf_res['risk_level']})</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'>", unsafe_allow_html=True)
    pct_val = shelf_res['pct_remaining']
    st.progress(pct_val / 100.0, text=f"Remaining Shelf Life Budget: {pct_val:.1f}% ({shelf_res['RSL_final_days']:.2f} days / {shelf_res['RSL_final_hours']:.1f} hours remaining)")
    
    st.markdown(f"""
    <div class="neon-callout" style="font-size: 12.5px;">
        <b>📐 Mathematical Formula Pipeline:</b> 
        <code>RSL_final = (S₀ - ∑ r(T)Δt) + XGBoost(T, RH, r(T), D, RSL_phys)</code> = 
        <b>{shelf_res['S0_days']:.1f}</b> - <b>{shelf_res['cumulative_degradation_D']:.2f}</b> + (<b>{shelf_res['delta_RSL_ML_days']:+.2f}</b>) = 
        <b style="color: {shelf_res['risk_color']};">{shelf_res['RSL_final_days']:.2f} Days ({shelf_res['RSL_final_hours']:.1f} Hours remaining)</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. ENVIRONMENTAL SENSOR STATISTICS SUMMARY
    readings = curr_shipment.get("readings", [])
    if readings:
        rdf = pd.DataFrame(readings)
        t_min, t_mean, t_max = rdf["temperature_C"].min(), rdf["temperature_C"].mean(), rdf["temperature_C"].max()
        h_min, h_mean, h_max = rdf["humidity_RH"].min(), rdf["humidity_RH"].mean(), rdf["humidity_RH"].max()
        crop_p = engine.get_crop_params(curr_shipment["crop"])
        
        t_opt_l, t_opt_h = crop_p.get("T_opt_min", 12.0), crop_p.get("T_opt_max", 20.0)
        compliant_steps = ((rdf["temperature_C"] >= t_opt_l) & (rdf["temperature_C"] <= t_opt_h)).sum()
        compliance_pct = (compliant_steps / len(rdf)) * 100.0

        st.markdown("<b>🌡️ In-Transit Cold-Chain Telemetry Summary Statistics:</b>", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Mean Temperature", f"{t_mean:.1f} °C", f"Range: {t_min:.1f} - {t_max:.1f} °C")
        with s2:
            st.metric("Mean Humidity", f"{h_mean:.1f} %", f"Range: {h_min:.1f} - {h_max:.1f} %")
        with s3:
            st.metric("Cold-Chain Compliance", f"{compliance_pct:.1f} %", f"{compliant_steps}/{len(rdf)} safe steps")
        with s4:
            st.metric("Total Telemetry Logs", f"{len(rdf)} readings", f"~{len(rdf)} hours tracked")

    st.markdown("---")

    # 4. EXPORT AND DOWNLOAD ACTIONS
    st.markdown("<b>📥 Report Downloads & Fleet Export:</b>", unsafe_allow_html=True)
    d_col1, d_col2, d_col3 = st.columns(3)
    
    # Generate complete self-contained HTML printable report
    html_report = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Freshroute Cargo Dossier - {curr_shipment['id']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #ffffff; color: #1e293b; padding: 30px; line-height: 1.5; }}
        .header {{ border-bottom: 3px solid #00f2fe; padding-bottom: 12px; margin-bottom: 20px; }}
        .title {{ font-size: 24px; font-weight: bold; color: #0f172a; margin: 0; }}
        .badge {{ background: #00f5a0; color: #0f172a; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
        .box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; }}
        .metric-val {{ font-size: 20px; font-weight: bold; color: #0284c7; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 12px; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: left; }}
        th {{ background: #f1f5f9; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">🌱 Freshroute Intelligence Report : {curr_shipment['id']}</h1>
        <p>Cargo: <b>{curr_shipment['crop']}</b> | Farmer: <b>{curr_shipment['farmer']}</b> | Truck: <b>{curr_shipment['truck_id']}</b></p>
    </div>
    <div class="grid">
        <div class="box">
            <h3>Visual Inspection (MobileNetV2 CNN)</h3>
            <p>Condition: <span class="badge">{curr_shipment['condition']}</span> ({curr_shipment['confidence']}%)</p>
            <p>Payload: <b>{curr_shipment['quantity_kg']:,} kg</b></p>
        </div>
        <div class="box">
            <h3>Shelf-Life Forecast (Physics + XGBoost)</h3>
            <p>Final Remaining Shelf Life: <span class="metric-val">{shelf_res['RSL_final_days']:.2f} Days</span> ({shelf_res['RSL_final_hours']} hours)</p>
            <p>Degradation Loss D: <b>{shelf_res['cumulative_degradation_D']:.2f} days</b></p>
        </div>
    </div>
    <div class="box">
        <h3>Recommended Vendor Dispatch</h3>
        <p>Target: <b>{best_vendor['vendor_name'] if best_vendor else 'Central Hub'}</b> ({best_vendor['city'] if best_vendor else ''})</p>
        <p>Distance: <b>{best_vendor['distance_km'] if best_vendor else 0} km</b> | ETA: <b>~{best_vendor['transit_minutes'] if best_vendor else 0} mins</b> | Net Profit: <b>₹{best_vendor['net_profit'] if best_vendor else 0:,}</b></p>
    </div>
    <p style="font-size: 11px; color: #64748b; margin-top: 30px;">Generated by Freshroute Smart Cold-Chain Platform on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>"""

    with d_col1:
        st.download_button(
            label="📄 Download Printable HTML Dossier",
            data=html_report.encode("utf-8"),
            file_name=f"Freshroute_Report_{curr_shipment['id']}.html",
            mime="text/html",
            use_container_width=True
        )

    with d_col2:
        if readings:
            csv_data = pd.DataFrame(readings).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Telemetry CSV",
                data=csv_data,
                file_name=f"Telemetry_{curr_shipment['id']}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with d_col3:
        json_str = json.dumps({
            "shipment": curr_shipment,
            "shelf_life_assessment": shelf_res,
            "recommended_dispatch": best_vendor
        }, indent=2).encode('utf-8')
        st.download_button(
            label="📋 Download JSON Cargo State",
            data=json_str,
            file_name=f"Cargo_State_{curr_shipment['id']}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# FOOTER STATUS BAR
# ==============================================================================
st.markdown("---")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    st.caption(f"🕒 Sync Timestamp: {time.strftime('%H:%M:%S')}")
with f_col2:
    st.caption(f"💾 Active Records: {len(curr_shipment.get('readings', []))} telemetry entries")
with f_col3:
    st.caption(f"🎨 Active Theme: {st.session_state.ui_theme}")
with f_col4:
    if st.button("🔄 Force Sync & Refresh", use_container_width=True):
        st.rerun()

# Auto refresh if selected
if auto_refresh:
    time.sleep(2)
    st.rerun()
