@echo off
title Freshroute ESP32 Cloud Bridge
echo ======================================================================
echo    FRESHROUTE - ESP32 LOCAL TO CLOUD REAL-TIME TELEMETRY BRIDGE
echo ======================================================================
echo.
echo Connecting to local ESP32 (COM14) and streaming live data to:
echo https://agrilife-ai-app-nj283lphasyhcnbwac4q4n.streamlit.app/
echo.
.venv\Scripts\python.exe esp32_cloud_bridge.py
pause
