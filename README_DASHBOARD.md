# 🥦 PULSE FRESH AI — AI-Powered Perishable Supply Chain Intelligence Platform

**PULSE FRESH AI** is an enterprise-grade, end-to-end perishable supply chain intelligence and cold-chain routing platform. It combines edge IoT telemetry, CNN computer vision, kinetic $Q_{10}$ physics models, residual XGBoost machine learning, and a multi-objective Intelligent Decision Engine with FEFO (First-Expired, First-Out) optimization and advance payment protection logic.

---

## 👥 Two Primary User Interfaces

### 1. 👨‍🌾 SUPPLIER / FARMER / TRUCK OWNER
1. **🏠 Command Center & Dashboard**: Executive fleet KPIs, dynamic recovery state, active cargo spotlight, and global actionable AI recommendation cards.
2. **📦 Register Shipment**: Multi-shipment support (register multiple shipments in the same truck), crop parameters, crates, loading timestamps, target mandis, and payment commitments (**ADVANCE PAID** vs **PAY AFTER DELIVERY**).
3. **📡 Live Truck & IoT Monitoring**: Live chamber temperature, relative humidity, smartphone GPS coordinates, transit duration, distance travelled/remaining, ETA, and historical temperature/humidity/thermal exposure plots.
4. **📷 AI Vision & 🧠 Shelf-Life Engine**:
   - **AI Vision Pipeline**: Camera $\rightarrow$ OpenCV $\rightarrow$ MobileNetV2 CNN $\rightarrow$ Crop species & Fresh/Rotten visual quality with confidence %.
   - **Remaining Shelf-Life Engine**: Crop Parameters + Exposure $\rightarrow$ $Q_{10}$ Physics Model $\rightarrow$ Physics Estimate $(S_0 - D)$ $\rightarrow$ XGBoost ML Correction $\rightarrow$ Final Remaining Shelf Life ($RSL_{final}$).
   - **Crop Parameter Database**: Stored parameters for Tomato, Banana, Apple, Mango, Strawberry, Grapes, Spinach, Potato, Onion, Orange ($S_0$, $Q_{10}$, $T_{ref}$, $r_{ref}$, $T_{opt}$, $RH_{opt}$, Critical Threshold, Price).
   - **Configurable Critical Shelf-Life Threshold**: Dynamic threshold slider (default 1.5 days) evaluating SAFE / WARNING / CRITICAL states.
5. **⚡ FEFO (First-Expired, First-Out)**: Prioritizes shipments strictly by **predicted remaining shelf life** (e.g. Tomato 1.2d $\rightarrow$ Banana 3.2d $\rightarrow$ Apple 5.8d), sorting by lowest shelf life, highest risk, or highest cargo value.
6. **🏪 Buyer Comparison & AI Match Scoring**: Evaluates alternative buyers with **AI Match Score %** (e.g. 94%), applying Advance Payment logic (balancing profit + payment security + shelf life buffer + ETA + distance + traffic).
7. **🗺️ Route Optimization & GPS Map**: Interactive multi-entity map (Truck, Buyers, Cold Storage, Traffic) comparing **Route A** (City Arterial, congested) vs **Route B** (AI Expressway Bypass, faster transit saving perishables).
8. **❄️ Cold Storage & Emergency Diversion**: Directory of nearby refrigerated facilities with distance, ETA, capacity %, daily storage cost, and automatic emergency cold-storage diversion recommendations.
9. **🚨 Real-Time Alerts Center**: Live alert hub for temperature breaches, humidity excursions, critical shelf-life approaches, heavy traffic, and buyer opportunities.
10. **📊 Analytics & Profit Optimization**: Demonstrates economic value preservation (Expected Revenue, Delivery Costs, Spoilage Loss, Value Preserved, Net Profit) and historical supplier KPI scorecards.

---

### 2. 🏬 BUYER / TRADER / VENDOR
1. **🏠 Buyer Dashboard**: Procurement metrics, incoming cold-chain shipments, freshness compliance score.
2. **📝 Post Produce Requirement**: Allows buyers to broadcast bids:
   - Crop, Variety, Required Quantity
   - **“I am willing to pay ₹___ / kg.”**
   - Maximum acceptable ceiling price
   - Preferred payment terms (**Advance Paid** vs **Pay After Delivery**)
   - Directly injects into the supplier rerouting matrix!
3. **🛒 Available Shipments Marketplace**: Browse active in-transit supply batches and place instant bids.
4. **🚚 Incoming Shipments Monitor**: Real-time telemetry, visual quality grade, live remaining shelf life, truck GPS, and condition badges (SAFE / WARNING / CRITICAL).
5. **💼 My Offers & Bids**: Status of all procurement bids and contract negotiations.
6. **🗺️ Live Delivery Map**: Buyer's perspective GPS tracking of incoming refrigerated carrier.
7. **📜 Digital QA Certificate**: Automated Certificate of Perishable Integrity detailing cold chain compliance and shelf life on delivery.

---

## ⚙️ Technical Architecture & Data Flow

```text
[DHT22 / SHT31 Sensors] ──► [ESP32 MCU] ──(BLE 5.0)──► [Driver Smartphone (GPS)] ──► [Backend Engine]
                                                                                             │
[Camera Frame] ──(OpenCV)──► [MobileNetV2 CNN] ──► [Crop Species + Quality Condition]        │
                                                                                             ▼
[Crop Parameters DB + Thermal Exposure] ──► [Q10 Kinetic Physics Model] ──► [Physics RSL Baseline]
                                                                                     │
                                                                                     ▼
                                                                        [XGBoost Residual ML Engine]
                                                                                     │
                                                                                     ▼
                                                                       [Final Remaining Shelf Life]
                                                                                     │
                                                                                     ▼
[Advance Payment Logic + Real-Time Traffic + Buyer Demands + Cold Storage] ──► [AI Decision Engine]
                                                                                     │
                                                                                     ▼
                                                         [FEFO Queue + Buyer Match Score + Optimal Route]
```

---

## 🚀 How to Run the Platform

### Option 1: Run with Batch Script (Windows)
Double-click `run_dashboard.bat` or run:
```cmd
run_dashboard.bat
```

### Option 2: Run directly from Terminal
```bash
.venv\Scripts\streamlit run app.py --server.port 8501
```

Open your browser at `http://localhost:8501` to access **PULSE FRESH AI**.
