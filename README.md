# 🥦 Freshroute : Smart Agro-Cold Chain & Shelf-Life Intelligence Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Residual_ML-2E8B57.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Freshroute** is an enterprise-grade perishable supply chain intelligence platform integrating **ESP32 Edge IoT Telemetry**, **PyTorch MobileNetV2 Computer Vision**, **Biochemical Arrhenius $Q_{10}$ Kinetic Physics**, **Residual XGBoost Machine Learning**, and **FEFO (First-Expired, First-Out) Multi-Objective Geospatial Routing**.

---

## 📖 Complete Documentation & Technical Manual
- **Interactive UI Documentation**: Launch the dashboard and navigate to the **`📚 README UI : Complete Architecture & Blueprint`** tab or the **`🧪 Interactive Algorithm Lab & Math Playground`** tab.
- **Markdown Architecture Manual**: Full technical documentation is available in **[`README_UI.md`](./README_UI.md)**.
- **Original Dashboard Summary**: See **[`README_DASHBOARD.md`](./README_DASHBOARD.md)**.

---

## 🚀 How to Run the Platform

### Option 1: Run with Batch Script (Windows)
Double-click `run_dashboard.bat` or run in terminal:
```cmd
run_dashboard.bat
```

### Option 2: Run directly via Streamlit
```bash
.\.venv\Scripts\streamlit run app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---

## 🏗️ Core Architecture Overview

```text
[DHT22 / ESP32 Sensor] ---> [USB Serial / BLE / Wi-Fi] ---> [ESP32SensorManager]
                                                                    |
[Camera Frame]          ---> [OpenCV DirectShow]        ---> [MobileNetV2 CNN]
                                                                    |
                                                                    v
                                                     [Q10 Kinetic Physics Model]
                                                                    | (S0 - D)
                                                                    v
                                                     [XGBoost Residual Regressor]
                                                                    | (Delta RSL)
                                                                    v
                                                    [Final Remaining Shelf Life]
                                                                    |
                                                                    v
                                                    [FEFO Multi-Shipment Queue]
                                                                    |
                                                                    v
                                                    [Plotly Geospatial Demand Map]
```
