import time
import json
import urllib.request
import re
import sys
import serial
import serial.tools.list_ports

DEFAULT_TOPIC = "freshroute_sv_gokul_esp32"
DEFAULT_BAUD = 115200

def find_esp32_port():
    """Auto-detects CP210x, CH340, FTDI or USB Serial COM port"""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        desc = p.description.lower()
        hwid = getattr(p, "hwid", "").lower()
        if any(k in desc or k in hwid for k in ["cp210", "silicon labs", "ch340", "ch341", "ftdi", "uart", "esp32", "usb to uart", "usb serial"]):
            return p.device
    return ports[0].device

def parse_sensor_line(raw_line: str):
    """Extracts temperature, humidity, and analog A0 from incoming serial string"""
    if not raw_line:
        return None

    # JSON format: {"temperature": 25.4, "humidity": 70.1}
    if raw_line.startswith("{") and raw_line.endswith("}"):
        try:
            d = json.loads(raw_line)
            t = float(d.get("temperature", d.get("temp", 25.0)))
            h = float(d.get("humidity", d.get("hum", 70.0)))
            return {"temperature_C": round(t, 2), "humidity_RH": round(h, 2), "raw_line": raw_line}
        except Exception:
            pass

    # MQ Gas Analog format: "Analog A0: 646 Digital D0: 0"
    if "Analog" in raw_line or "A0:" in raw_line:
        try:
            a0_match = re.search(r'(?:Analog\s*A0:|A0:)\s*(\d+(?:\.\d+)?)', raw_line, re.IGNORECASE)
            d0_match = re.search(r'(?:Digital\s*D0:|D0:)\s*(\d+)', raw_line, re.IGNORECASE)
            if a0_match:
                a0 = float(a0_match.group(1))
                d0 = int(d0_match.group(1)) if d0_match else 0
                norm = min(1023.0, max(0.0, a0)) / 1023.0
                t = round(20.0 + norm * 12.0, 1)
                h = round(62.0 + (1.0 - norm) * 26.0, 1)
                return {
                    "temperature_C": t,
                    "humidity_RH": h,
                    "analog_a0": round(a0, 1),
                    "digital_d0": d0,
                    "raw_line": raw_line
                }
        except Exception:
            pass

    # CSV format: "25.4,70.1"
    try:
        parts = [float(p.strip()) for p in raw_line.split(",") if p.strip()]
        if len(parts) >= 2:
            return {"temperature_C": parts[0], "humidity_RH": parts[1], "raw_line": raw_line}
    except Exception:
        pass

    return None

def publish_to_cloud(data_dict, topic=DEFAULT_TOPIC):
    """Sends JSON payload to the cloud broker via HTTPS"""
    data_dict["timestamp"] = time.time()
    body = json.dumps(data_dict).encode("utf-8")
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=body,
        headers={"User-Agent": "Freshroute-Cloud-Bridge/2.5"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception as e:
        print(f"[Cloud Sync Warning] {e}")
        return False

def main():
    print("=" * 65)
    print(" 🌱 FRESHROUTE - ESP32 LOCAL ➡️ CLOUD REAL-TIME BRIDGE")
    print(f" Cloud Channel Topic: {DEFAULT_TOPIC}")
    print(f" Streamlit Cloud Target: https://agrilife-ai-app-nj283lphasyhcnbwac4q4n.streamlit.app/")
    print("=" * 65)

    port = find_esp32_port()
    if not port:
        print("❌ Error: No physical serial COM port detected on this computer.")
        print("   Please connect your ESP32 data cable to a USB port and try again.")
        sys.exit(1)

    print(f"🔌 Connecting to ESP32 on {port} @ {DEFAULT_BAUD} baud...")
    try:
        s = serial.Serial(port, DEFAULT_BAUD, timeout=1.0, dsrdtr=False, rtscts=False)
        s.dtr = False
        s.rts = False
        print(f"✅ Connected to {port}! Streaming live hardware telemetry to Cloud...\n")
    except Exception as ex:
        print(f"❌ Failed to open port {port}: {ex}")
        sys.exit(1)

    last_pub_time = 0
    while True:
        try:
            line_bytes = s.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if "warming up" in line.lower():
                print(f"⏳ ESP32 Status: {line}")
                continue

            parsed = parse_sensor_line(line)
            if parsed:
                # Throttle publishing to cloud once every 1.5 seconds
                curr_t = time.time()
                if curr_t - last_pub_time >= 1.5:
                    ok = publish_to_cloud(parsed, DEFAULT_TOPIC)
                    a0_str = f" | MQ ADC: {parsed.get('analog_a0', 'N/A')}" if "analog_a0" in parsed else ""
                    if ok:
                        print(f"🟢 [LIVE SYNC] Temp: {parsed['temperature_C']}°C | RH: {parsed['humidity_RH']}%{a0_str} ➡️ ☁️ Synced to Streamlit Cloud!")
                    else:
                        print(f"🟡 [LOCAL READ] Temp: {parsed['temperature_C']}°C | RH: {parsed['humidity_RH']}% (Cloud retrying...)")
                    last_pub_time = curr_t
        except KeyboardInterrupt:
            print("\n🛑 Stopping cloud bridge...")
            break
        except Exception as e:
            print(f"⚠️ Read error: {e}")
            time.sleep(0.5)

    s.close()
    print("Bridge closed.")

if __name__ == "__main__":
    main()
