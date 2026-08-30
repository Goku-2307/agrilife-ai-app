# 🥦 AgriLife AI : Complete UI & Backend Technical Architecture Manual

**AgriLife AI** is an enterprise-grade, end-to-end perishable supply chain intelligence, agro-cold chain telemetry, and dynamic logistics routing platform. It seamlessly integrates **physical ESP32 edge IoT telemetry**, **PyTorch MobileNetV2 computer vision**, **biochemical Arrhenius $Q_{10}$ kinetic physics**, **gradient boosted machine learning (XGBoost)**, and a **multi-objective FEFO (First-Expired, First-Out) geospatial routing engine** to eliminate transit spoilage and maximize agro-logistics profitability.

---

## 📑 Table of Contents
1. [🏛️ Executive System Architecture](#-executive-system-architecture)
2. [🖥️ Frontend UI Components Deep-Dive](#️-frontend-ui-components-deep-dive)
   - [1. Hero Command Banner & Global Status Badges](#1-hero-command-banner--global-status-badges)
   - [2. Multi-Shipment Registry & State Caching](#2-multi-shipment-registry--state-caching)
   - [3. ESP32 Telemetry Link Controls](#3-esp32-telemetry-link-controls)
   - [4. Visual Freshness Verification (OpenCV + MobileNetV2 CNN)](#4-visual-freshness-verification-opencv--mobilenetv2-cnn)
   - [5. ESP32 Environmental Telemetry & Dual-Axis Realtime Graphs](#5-esp32-environmental-telemetry--dual-axis-realtime-graphs)
   - [6. Hybrid Shelf-Life Engine UI](#6-hybrid-shelf-life-engine-ui)
   - [7. FEFO Multi-Shipment Priority Dispatch Queue](#7-fefo-multi-shipment-priority-dispatch-queue)
   - [8. Nearby Shops Demand Map & Next Stop Route](#8-nearby-shops-demand-map--next-stop-route)
   - [9. Interactive Algorithm Lab & Math Playground](#9-interactive-algorithm-lab--math-playground)
3. [⚙️ Backend Engineering Deep-Dive](#️-backend-engineering-deep-dive)
   - [1. Edge IoT Telemetry & ESP32 Protocol Layer (`sensor_manager.py`, `esp32_sensor_ble_serial.ino`)](#1-edge-iot-telemetry--esp32-protocol-layer)
   - [2. Deep Learning Computer Vision Pipeline (`vision_detector.py`)](#2-deep-learning-computer-vision-pipeline)
   - [3. Kinetic Physics $Q_{10}$ Degradation Engine (`shelf_life_engine.py`, `01_physics_temperature_model.py`)](#3-kinetic-physics-q10-degradation-engine)
   - [4. XGBoost Residual Machine Learning Regressor (`02_prepare_temperature_model.py`, `03_train_temperature_xgboost.py`)](#4-xgboost-residual-machine-learning-regressor)
   - [5. Dynamic FEFO (First-Expired, First-Out) Logistics Engine (`fefo_routing.py`)](#5-dynamic-fefo-first-expired-first-out-logistics-engine)
   - [6. Geospatial Multi-Objective Demand & Route Optimizer (`fefo_routing.py`)](#6-geospatial-multi-objective-demand--route-optimizer)
4. [📊 Crop Database & Parameter Reference Matrix](#-crop-database--parameter-reference-matrix)
5. [🏬 Commercial Buyer & Retail Network Database](#-commercial-buyer--retail-network-database)
6. [🚀 Installation & Quickstart Guide](#-installation--quickstart-guide)

---

## 🏛️ Executive System Architecture

```text
+---------------------------------------------------------------------------------------------------------+
|                                         PHYSICAL EDGE SENSING LAYER                                     |
|  [DHT22 / SHT31 Sensors] ---> [ESP32 DevKit V1 (FreeRTOS)] ---> [USB UART / BLE 5.0 / Wi-Fi HTTP]       |
|  [USB Webcam / Camera]   ---> [OpenCV DirectShow Grabber]  ---> [224x224 RGB Tensor]                    |
+---------------------------------------------------------------------------------------------------------+
                                                       |
                                                       v
+---------------------------------------------------------------------------------------------------------+
|                                          AI & ANALYTICS BACKEND LAYER                                   |
|  1. MobileNetV2 CNN: Classifies Crop Species & Freshness (Confidence %)                                 |
|  2. Crop Database: Retrieves S0 (Initial Budget), Q10, T_ref, r_ref, T_opt, RH_opt                       |
|  3. Kinetic Q10 Physics Engine: Calculates Instantaneous Rate r(T) & Cumulative Thermal Exposure (D)    |
|  4. Physics Baseline RSL: RSL_phys = max(0, S0 - D)                                                     |
|  5. 24-Step Sliding Window (120 Features): Ingests [T, RH, r(T), D, RSL_phys]                           |
|  6. XGBoost ML Regressor: Predicts Non-Linear Residual Correction (Delta RSL_ML)                        |
|  7. Final Remaining Shelf Life: RSL_final = max(0, RSL_phys + Delta RSL_ML)                             |
|  8. FEFO Prioritizer: Sorts Fleet Shipments by RSL_final Ascending (Immediate / Expedite / Scheduled)   |
|  9. Multi-Objective Route Optimizer: Evaluates Haversine Dist, ETA, Buffer Margin, Price & Profit      |
+---------------------------------------------------------------------------------------------------------+
                                                       |
                                                       v
+---------------------------------------------------------------------------------------------------------+
|                                      FRONTEND STREAMLIT COMMAND CENTER                                  |
|  • Tab 1: Live Command Center (Vision HUD + ESP32 Charts + Formula Cards + FEFO Queue + Plotly Map)    |
|  • Tab 2: Interactive README UI (Complete Visual Architecture & Blueprint)                              |
|  • Tab 3: Interactive Algorithm Lab & Math Playground (Real-Time Sandbox & Parameter Explorer)          |
+---------------------------------------------------------------------------------------------------------+
```

---

## 🖥️ Frontend UI Components Deep-Dive

### 1. Hero Command Banner & Global Status Badges
- **Purpose**: Provides instantaneous high-level situational awareness for fleet dispatchers, logistics managers, and truck drivers.
- **Active Metadata Displayed**:
  - **Cargo Tracking ID**: Unique consignment identifier (e.g., `SH001`).
  - **Farmer / Origin**: Source agricultural cluster (e.g., *Ramanathan Agri Farms, Vellore Green Belt*).
  - **Truck Registration**: Carrier vehicle license plate (e.g., `TN-23-AB-4412`).
  - **Payload Quantity & Crop**: Total cargo weight in kilograms (e.g., `3,500 kg Banana`).
- **Real-Time Badges**:
  - `📡 ESP32: Mode`: Displays active telemetry transport (`Simulator`, `Serial`, `HTTP`, or `Manual`).
  - `👁️ CNN: MobileNetV2`: Displays active deep learning vision state.
  - `🧠 ML: XGBoost Regressor`: Displays residual ML engine operational health.

### 2. Multi-Shipment Registry & State Caching
- **Persistent State**: Managed via Streamlit `st.session_state.shipments`. Supports concurrent monitoring of multiple distinct refrigerated trucks across different regional corridors.
- **Interactive Switcher**: Selecting any shipment dynamically recalculates all downstream physical models, ML predictions, FEFO queue positions, and recommended map routes in real-time.
- **"➕ Register New Cargo Shipment" Expander**:
  - Allows dispatchers to register new consignments on the fly.
  - Inputs: Shipment ID, Farmer Origin, Truck Reg, Crop Type, Quantity (kg).
  - Automatically initializes baseline telemetry streams and syncs with the fleet queue.

### 3. ESP32 Telemetry Link Controls
Supports 4 plug-and-play communication modalities:
1. **Simulator (Real-time Stream)**: Generates high-fidelity simulated telemetry with 4 stress scenarios:
   - `NORMAL`: Steady cold chain ($24^\circ\text{C} \pm 0.4^\circ\text{C}, 72\% \pm 0.8\%$ RH).
   - `COOLING_FAILURE`: Exponential chamber temperature rise ($\Delta T = +0.8^\circ\text{C/step}$, simulating compressor breakdown).
   - `HEATWAVE`: Severe thermal spike ($38^\circ\text{C}-42^\circ\text{C}$).
   - `HIGH_HUMIDITY`: Saturated moisture ($>94\%$ RH) triggering mold hazard warnings.
2. **Physical Serial (USB/COM)**: Auto-enumerates system COM ports using `pyserial`, presents device descriptions, and connects at `115200` baud.
3. **Wi-Fi HTTP / IP Stream**: Queries embedded FreeRTOS web servers on ESP32 boards via REST endpoint `http://<ip>/data` with ping latency validation.
4. **Manual Injection**: Fine-grained sliders for temperature ($-5^\circ\text{C}$ to $50^\circ\text{C}$) and humidity ($20\%$ to $100\%$ RH).
5. **Auto-Stream Toggle**: Executes a 2-second background refresh cycle for live streaming.

### 4. Visual Freshness Verification (OpenCV + MobileNetV2 CNN)
- **Input Modalities**:
  1. *Live OpenCV Webcam Feed*: Real-time camera streaming with DirectShow / MSMF backend selection and auto-exposure warmup frames.
  2. *Browser Camera Snapshot*: HTML5 WebRTC browser camera snapshot grabber.
  3. *Upload Cargo Photo*: Standard file uploader for field image testing (JPG/PNG).
  4. *Sample Test Specimens*: Built-in verified test specimens (Fresh Banana, Rotten Banana, Fresh Apple, Rotten Apple, Fresh Tomato, Fresh Mango).
- **HUD Visual Overlay**: Highlights detected cargo with a glowing green (Fresh) or red (Rotten) targeting bounding box, confidence score, and cargo ID overlay.
- **Class Probabilities Bar Chart**: Horizontal Plotly bar chart detailing the softmax probability distribution across all trained classes.

### 5. ESP32 Environmental Telemetry & Dual-Axis Realtime Graphs
- **3 Live Digital Gauges**:
  1. *Instantaneous Temperature (°C)*: Color-coded (Green: Optimal, Yellow: Sub-optimal, Red: Severe Thermal Breach).
  2. *Relative Humidity (% RH)*: Color-coded against crop-specific transpiration thresholds.
  3. *Instantaneous Degradation Multiplier $r(T)$*: Real-time $Q_{10}$ rate factor ($x$ times nominal degradation).
- **Plotly Dual-Axis Line Graph**:
  - Left Y-Axis (Red): Chamber Temperature (°C).
  - Right Y-Axis (Blue): Chamber Humidity (% RH).
  - Shaded Green Band: Highlights the crop-specific optimal storage temperature range ($T_{\text{opt, min}} - T_{\text{opt, max}}$).

### 6. Hybrid Shelf-Life Engine UI
- **4 Key Mathematical Calculation Cards**:
  1. **$S_0$ Harvest Shelf Budget**: Baseline lifespan at optimal harvest conditions.
  2. **Physics RSL ($S_0 - D$)**: Kinetic baseline after subtracting cumulative thermal degradation loss $D$.
  3. **XGBoost ML Correction ($\Delta RSL_{\text{ML}}$)**: Non-linear residual offset predicted from the 120-feature window.
  4. **Final Remaining Shelf Life ($RSL_{\text{final}}$)**: Composite shelf life in both **Days** and **Hours**, decorated with dynamic risk borders.
- **Progress Bar & Risk Assessment**:
  - `OPTIMAL / SAFE` ($RSL > 4.5\text{ days}$): Green status.
  - `MODERATE / EXPEDITE` ($2.0 < RSL \le 4.5\text{ days}$): Amber status.
  - `CRITICAL / HIGH` ($RSL \le 2.0\text{ days}$): Crimson red status.
- **Formula Pipeline Callout**: Transparently exposes the exact mathematical arithmetic step-by-step.

### 7. FEFO Multi-Shipment Priority Dispatch Queue
- **Dynamic Ascending Table**: Automatically ranks all active cargo batches in the fleet by lowest remaining shelf life first.
- **Urgency Badges**:
  - `🚨 IMMEDIATE DISPATCH (FEFO Critical)`
  - `⚠️ EXPEDITE SHIPMENT (FEFO Priority 2)`
  - `✅ SCHEDULED DISPATCH (FEFO Normal)`
- **Truck GPS Corridor Preset Selector**: Moves the truck to major transit corridors (Vellore Highway, Outer Ring Road, Guindy Central, Koyambedu Bypass, Sriperumbudur Industrial) to test routing behavior.

### 8. Nearby Shops Demand Map & Next Stop Route
- **Recommended Next Stop Highlight Card**: Displays chosen buyer name, store category, city area, manager contact, buying price (₹/kg), commercial net profit (₹), road distance (km), transit ETA (mins), and shelf safety buffer margin (+hours).
- **Plotly Carto-Darkmatter Map**:
  - 🚚 **Truck Marker (Blue)**: Current GPS coordinates of the refrigerated carrier.
  - 🎯 **Recommended Destination (Green)**: Optimal next stop highlighted with enlarged icon.
  - 🏪 **Commercial Demand Hubs (Gold/Amber)**: Other nearby retailers scaled in size by demand volume (kg).
  - 🛣️ **Route Polyline (Emerald Green)**: Smooth Bézier curvature navigation path connecting truck to target destination.
- **Explore Nearby Shops Demand Matrix**: Expandable data table ranking all evaluated candidate shops with price, distance, net profit, and cold storage status.

### 9. Interactive Algorithm Lab & Math Playground
- **Q₁₀ Kinetic Physics Simulator**: Live interactive sliders for $S_0, Q_{10}, T_{\text{ref}}$, chamber temperature, and exposure time with real-time degradation multiplier and shelf life decay curve visualization.
- **XGBoost 120-Feature Window Inspector**: Interactive tabular breakdown of the 24 past hourly readings across 5 features.
- **Multi-Objective Route Score Calculator**: Interactive sandbox testing custom buyer prices, demands, distances, and shelf life constraints.

---

## ⚙️ Backend Engineering Deep-Dive

### 1. Edge IoT Telemetry & ESP32 Protocol Layer
- **Source Files**: `sensor_manager.py`, `esp32_sensor_ble_serial.ino`
- **Microcontroller**: ESP32 DevKit V1 (32-bit dual-core Tensilica Xtensa LX6 @ 240MHz).
- **Sensor**: DHT22 (Aosong AM2302) capacitive humidity sensor and NTC thermistor (Accuracy: $\pm 0.5^\circ\text{C}, \pm 2\% \text{ RH}$, Sampling: $0.5\text{ Hz}$).
- **Firmware Logic (`esp32_sensor_ble_serial.ino`)**:
  - Uses `DHT.h` library on `GPIO 4`.
  - Serial UART baud: `115200` baud.
  - FreeRTOS timer task samples sensor data every $2,000\text{ ms}$.
  - Transmits standardized JSON payload:
    ```json
    {"temperature": 24.50, "humidity": 71.80, "uptime_ms": 120400}
    ```
- **Python Hardware Interface (`ESP32SensorManager`)**:
  - `list_available_ports()`: Uses `serial.tools.list_ports.comports()` to scan USB UART devices.
  - `connect_serial(port, baud_rate)`: Establishes low-latency serial connection with auto-clearing input buffers.
  - `read_serial_line()`: Non-blocking parser extracting validated JSON dictionaries.
  - `fetch_http_reading(url)`: REST HTTP client with 1.5s timeout.
  - `generate_simulated_reading()`: High-fidelity mathematical random walk generator with simulated thermal failures.

---

### 2. Deep Learning Computer Vision Pipeline
- **Source File**: `vision_detector.py`
- **Model Architecture**: PyTorch MobileNetV2 (`torchvision.models.mobilenet_v2`).
- **Transfer Learning Setup**:
  - Base feature extractor: 53 convolutional layers utilizing inverted residual bottlenecks.
  - Final classifier replaced with: `Linear(in_features=1280, out_features=num_classes)`.
  - Weights loaded from: `ShelfLife-CNN/best_fruit_quality_model.pth`.
- **Classes**: `fresh_apple`, `fresh_banana`, `rotten_apple`, `rotten_banana`, `fresh_tomato`, `fresh_mango`.
- **Inference Pipeline**:
  1. Frame capture via OpenCV (`cv2.VideoCapture` with DirectShow backend).
  2. Multi-frame auto-exposure settling (12 warmup frames).
  3. Color conversion `BGR -> RGB`.
  4. Resize to $224 \times 224$ pixels.
  5. Tensor conversion and ImageNet normalization:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  6. Softmax probability extraction:
     $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^K e^{z_j}}$$
  7. HUD rendering with target corner brackets and text overlay.

---

### 3. Kinetic Physics $Q_{10}$ Degradation Engine
- **Source Files**: `shelf_life_engine.py`, `01_physics_temperature_model.py`
- **Theoretical Basis**: The Arrhenius equation establishes that chemical reaction rates increase exponentially with absolute temperature:
  $$k = A e^{-\frac{E_a}{RT}}$$
  For biological produce, this is parameterized via the empirical $Q_{10}$ temperature quotient.
- **Mathematical Formulations**:
  1. **Instantaneous Degradation Multiplier $r(T)$**:
     $$r(T) = r_{\text{ref}} \cdot Q_{10}^{\frac{T - T_{\text{ref}}}{10}}$$
     Where $T$ is current temperature (°C), $T_{\text{ref}}$ is reference baseline temperature (typically $25.0^\circ\text{C}$), and $r_{\text{ref}} = 1.0\text{ day}^{-1}$.
  2. **Cumulative Thermal Degradation Loss ($D$)**:
     $$D = \sum_{i=1}^N r(T_i) \cdot \Delta t_i \quad \left(\text{with } \Delta t_i = \frac{1}{24}\text{ days}\right)$$
  3. **Physics Remaining Shelf Life ($RSL_{\text{phys}}$)**:
     $$RSL_{\text{phys}} = \max\left(0,\; S_0 - D\right)$$

---

### 4. XGBoost Residual Machine Learning Regressor
- **Source Files**: `02_prepare_temperature_model.py`, `03_train_temperature_xgboost.py`, `shelf_life_engine.py`
- **Rationale for Residual Learning**: Pure physics models capture general temperature kinetics but miss complex environmental interactions (humidity deficit transpiration, cyclic thermal fatigue, condensation-induced mold growth). Instead of replacing the physics model with a black box, XGBoost is trained strictly on the **residual error**:
  $$\text{Target} = \Delta RSL_{\text{ML}} = RSL_{\text{actual}} - RSL_{\text{phys}}$$
- **Sliding Window Feature Matrix**:
  - Window Size: $24\text{ hours}$ ($24\text{ time steps}$).
  - Features per step ($5$ features):
    1. $T$: Temperature (°C)
    2. $RH$: Relative Humidity (% RH)
    3. $r(T)$: Instantaneous $Q_{10}$ degradation rate
    4. $D$: Cumulative thermal loss
    5. $RSL_{\text{phys}}$: Current physics baseline shelf life
  - Total Flattened Vector: $24 \times 5 = 120\text{ inputs}$.
- **XGBoost Hyperparameters**:
  - `n_estimators`: 500
  - `max_depth`: 5
  - `learning_rate`: 0.03
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
  - `objective`: `reg:squarederror`
- **Composite Output**:
  $$RSL_{\text{final}} = \max\left(0,\; RSL_{\text{phys}} + \Delta RSL_{\text{ML}}\right)$$

---

### 5. Dynamic FEFO (First-Expired, First-Out) Logistics Engine
- **Source File**: `fefo_routing.py`
- **Sorting Logic**:
  $$\text{Fleet Priority Queue} = \operatorname{arg\,sort}_{s \in \text{Fleet}} \left(RSL_{\text{final}}(s)\right)$$
- **Urgency Decision Tiers**:
  - $RSL_{\text{final}} \le 2.0\text{ days}$: **CRITICAL / IMMEDIATE DISPATCH** (Risk Tier 1)
  - $2.0 < RSL_{\text{final}} \le 4.5\text{ days}$: **MODERATE / EXPEDITE** (Risk Tier 2)
  - $RSL_{\text{final}} > 4.5\text{ days}$: **OPTIMAL / SCHEDULED** (Risk Tier 3)

---

### 6. Geospatial Multi-Objective Demand & Route Optimizer
- **Source File**: `fefo_routing.py`
- **Formulas**:
  1. **Haversine Distance**:
     $$d = 2R \cdot \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
     Where $R = 6371.0\text{ km}$.
  2. **Transit Time & Arrival Buffer**:
     $$t_{\text{transit}} = \frac{d}{42.0\text{ km/h}}, \quad \text{Buffer (hours)} = \left(RSL_{\text{final}} \times 24\right) - t_{\text{transit}}$$
     *Feasibility condition:* $\text{Buffer} > 3.0\text{ hours}$.
  3. **Commercial Net Profit**:
     $$\text{Net Profit} = \left(\min(Q_{\text{cargo}}, Q_{\text{demand}}) \times P_{\text{kg}}\right) - (d \times ₹8.5/\text{km})$$
  4. **Multi-Objective Composite Score ($S_{\text{opt}}$)**:
     - *If $RSL \le 2.0\text{ days}$ (Spoilage Protection Mode):*
       $$S_{\text{opt}} = \text{Net Profit} - (d \times 60) + (\text{Buffer} \times 120) + \text{ColdStorageBonus}(2000)$$
     - *If $RSL > 2.0\text{ days}$ (Profit Maximization Mode):*
       $$S_{\text{opt}} = \text{Net Profit} - (d \times 12)$$
  5. **Quadratic Bézier Curve Waypoints**:
     $$\mathbf{B}(t) = (1-t)^2\mathbf{P}_{\text{start}} + 2(1-t)t\mathbf{P}_{\text{mid}} + t^2\mathbf{P}_{\text{end}} \quad (t \in [0, 1])$$

---

## 📊 Crop Database & Parameter Reference Matrix

| Crop | Icon | Category | $S_0$ Budget (Days) | $Q_{10}$ Multiplier | $T_{\text{ref}}$ (°C) | $r_{\text{ref}}$ (/day) | Optimal Temp ($T_{\text{opt}}$) | Optimal RH ($RH_{\text{opt}}$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tomato** | 🍅 | Vegetable | 10.0 d | 2.0x | 25.0 °C | 1.0 | 12.0°C – 20.0°C | 85% – 95% |
| **Banana** | 🍌 | Fruit | 8.0 d | 2.2x | 25.0 °C | 1.0 | 13.0°C – 15.0°C | 90% – 95% |
| **Apple** | 🍎 | Fruit | 25.0 d | 1.8x | 20.0 °C | 0.8 | 1.0°C – 4.0°C | 90% – 95% |
| **Mango** | 🥭 | Fruit | 12.0 d | 2.1x | 25.0 °C | 1.0 | 12.0°C – 14.0°C | 85% – 90% |

---

## 🏬 Commercial Buyer & Retail Network Database

| ID | Business Name | Category | City Area | Cold Storage | Sample Crop Buying Price |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `VEND-01` | Metro Fresh Supermarket Hub | Supermarket Chain | Chennai Central | ❄️ Yes | Banana: ₹38/kg, Tomato: ₹42/kg |
| `VEND-02` | Koyambedu Wholesale Agro Mandi | Wholesale Mandi | Koyambedu | ❌ No | Banana: ₹32/kg, Tomato: ₹36/kg |
| `VEND-03` | Apex Food Processing & Puree Plant | Food Processing | Sriperumbudur | ❄️ Yes | Banana: ₹26/kg, Tomato: ₹30/kg |
| `VEND-04` | GreenHarvest Organic Retail Chain | Premium Retail | Adyar | ❄️ Yes | Banana: ₹48/kg, Tomato: ₹55/kg |
| `VEND-05` | Tambaram Regional Distribution Center | Regional Depot | Tambaram | ❄️ Yes | Banana: ₹34/kg, Tomato: ₹38/kg |
| `VEND-06` | Anna Nagar Mega Hypermarket | Hypermarket | Anna Nagar | ❄️ Yes | Banana: ₹39/kg, Tomato: ₹44/kg |
| `VEND-07` | Velachery Fresh Agro Express | Urban Mart | Velachery | ❄️ Yes | Banana: ₹40/kg, Tomato: ₹46/kg |
| `VEND-08` | Ambattur Industrial Aggregator | B2B Depot | Ambattur | ❄️ Yes | Banana: ₹33/kg, Tomato: ₹37/kg |
| `VEND-09` | Porur Cold-Chain Logistics Hub | Cold Facility | Porur | ❄️ Yes | Banana: ₹36/kg, Tomato: ₹40/kg |
| `VEND-10` | OMR Tech Corridor Superstore | Urban Superstore | Sholinganallur | ❄️ Yes | Banana: ₹42/kg, Tomato: ₹48/kg |

---

## 🚀 Installation & Quickstart Guide

### Prerequisites
- Python 3.9+ installed
- Web camera (for live OpenCV computer vision testing)
- ESP32 microcontroller with DHT22 sensor (optional for physical serial/BLE/Wi-Fi testing)

### Step 1: Clone / Navigate to Workspace
```bash
cd "C:\Users\SV Gokul\Downloads\shelf_life_temperature_humidity_model"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
# or directly:
pip install streamlit opencv-python torch torchvision plotly pandas numpy joblib xgboost pyserial
```

### Step 3: Run the Dashboard
```bash
streamlit run app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---
*AgriLife AI Platform — Engineering resilient, data-driven, and zero-waste perishable supply chains.*
