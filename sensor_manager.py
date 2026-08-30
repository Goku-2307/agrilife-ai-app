import time
import json
import random
import math
import re
import os
import threading
import urllib.request
from typing import Dict, List, Any, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SensorReading:
    """Represents a single timestamped sensor reading with optional gas/analog telemetry"""
    def __init__(
        self,
        temp_c: float,
        humidity_rh: float,
        timestamp: Optional[float] = None,
        analog_a0: Optional[float] = None,
        digital_d0: Optional[int] = None,
        raw_line: Optional[str] = None
    ):
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.temperature_C = float(temp_c)
        self.humidity_RH = float(humidity_rh)
        self.analog_a0 = float(analog_a0) if analog_a0 is not None else None
        self.digital_d0 = int(digital_d0) if digital_d0 is not None else None
        self.raw_line = raw_line

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "temperature_C": round(self.temperature_C, 2),
            "humidity_RH": round(self.humidity_RH, 2),
            "delta_t_days": 1.0 / 24.0  # Normalized step size
        }
        if self.analog_a0 is not None:
            d["analog_a0"] = round(self.analog_a0, 1)
        if self.digital_d0 is not None:
            d["digital_d0"] = self.digital_d0
        return d


class ESP32SensorManager:
    """
    Manages real hardware ESP32 Serial COM connections, Cloud Live Bridge (Local PC -> Cloud),
    Wi-Fi HTTP polling, and a high-fidelity real-time environmental simulation engine.
    """
    def __init__(self):
        self.active_mode = "SERIAL" if (SERIAL_AVAILABLE and os.name == "nt") else "CLOUD"
        self.serial_port: Optional[str] = None
        self.baud_rate: int = 115200
        self.serial_conn: Optional[Any] = None
        self.is_connected: bool = False
        self.status_message: str = "Ready to connect"
        self.latest_raw_line: str = ""
        self.cloud_topic: str = "freshroute_sv_gokul_esp32"
        self.http_url: str = "http://192.168.1.100/data"
        self.last_reading: Optional[SensorReading] = None
        self.shipment_history: Dict[str, List[SensorReading]] = {}

        # Background Thread for continuous non-blocking serial reading
        self._reader_thread: Optional[threading.Thread] = None
        self._thread_running: bool = False
        self._lock = threading.Lock()

        # Simulator State
        self.sim_base_temp: float = 24.0
        self.sim_base_humidity: float = 72.0
        self.sim_anomaly: str = "NORMAL"
        self.sim_step_count: int = 0

        # Auto-connect if a local ESP32 serial port is available
        if os.name == "nt":
            self.auto_connect_if_available()

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """Scans and lists system serial COM ports with device descriptions"""
        if not SERIAL_AVAILABLE:
            return []
        try:
            results = []
            for p in serial.tools.list_ports.comports():
                desc = f"{p.device} ({p.description})"
                results.append({
                    "port": p.device,
                    "description": desc,
                    "hwid": getattr(p, "hwid", ""),
                    "manufacturer": getattr(p, "manufacturer", "Unknown")
                })
            return results
        except Exception:
            return []

    def get_best_default_port(self) -> Optional[str]:
        """Identifies the most likely ESP32 / USB Serial COM port"""
        ports = self.list_available_ports()
        if not ports:
            return None
        # Look for typical ESP32 bridge chips (CP210x, CH340, FTDI, USB Serial)
        for p in ports:
            desc = p["description"].lower()
            hwid = p.get("hwid", "").lower()
            mfg = str(p.get("manufacturer", "")).lower()
            if any(k in desc or k in hwid or k in mfg for k in ["cp210", "silicon labs", "ch340", "ch341", "ftdi", "uart", "esp32", "usb to uart", "usb serial"]):
                return p["port"]
        return ports[0]["port"]

    def auto_connect_if_available(self) -> bool:
        """Attempts to auto-connect to the detected ESP32 port on startup"""
        best_port = self.get_best_default_port()
        if best_port and not self.is_connected:
            ok, _ = self.connect_serial(best_port, 115200)
            return ok
        return False

    def probe_port(self, port: str, baud_rate: int = 115200, timeout: float = 1.0) -> Tuple[bool, str, List[str]]:
        """Probes a port to see if data is actively arriving without breaking ongoing connections"""
        if not SERIAL_AVAILABLE:
            return False, "pyserial not available", []

        sample_lines = []
        try:
            test_conn = serial.Serial(port, baud_rate, timeout=timeout, dsrdtr=False, rtscts=False)
            test_conn.dtr = False
            test_conn.rts = False
            start_t = time.time()
            while time.time() - start_t < timeout:
                raw = test_conn.readline()
                if raw:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line:
                        sample_lines.append(line)
            test_conn.close()
            return True, f"Port {port} opened successfully ({len(sample_lines)} lines received)", sample_lines
        except Exception as e:
            return False, f"Failed to probe {port}: {str(e)}", []

    def connect_serial(self, port: str, baud_rate: int = 115200) -> Tuple[bool, str]:
        """Connects to physical ESP32 via USB Serial COM port with background reader thread"""
        if not SERIAL_AVAILABLE:
            return False, "pyserial library not installed."

        try:
            self.disconnect_serial()
            self.serial_conn = serial.Serial(port, baud_rate, timeout=0.5, dsrdtr=False, rtscts=False)
            self.serial_conn.dtr = False
            self.serial_conn.rts = False
            self.serial_conn.reset_input_buffer()
            self.serial_port = port
            self.baud_rate = baud_rate
            self.is_connected = True
            self.active_mode = "SERIAL"
            self.status_message = f"Connected to {port} @ {baud_rate} baud"

            # Start background reader thread
            self._thread_running = True
            self._reader_thread = threading.Thread(target=self._serial_worker, daemon=True)
            self._reader_thread.start()

            return True, f"Successfully connected to ESP32 on {port} @ {baud_rate} baud."
        except Exception as e:
            self.is_connected = False
            self.status_message = f"Connection failed: {str(e)}"
            return False, f"Failed to open {port}: {str(e)}"

    def disconnect_serial(self):
        """Closes serial connection and stops background reader thread safely"""
        self._thread_running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None
        self.is_connected = False
        self.status_message = "Disconnected"

    def _serial_worker(self):
        """Continuous background thread listening for ESP32 sensor stream"""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.readline()
        except Exception:
            pass

        while self._thread_running and self.serial_conn and self.serial_conn.is_open:
            try:
                raw_bytes = self.serial_conn.readline()
                if not raw_bytes:
                    time.sleep(0.05)
                    continue

                raw_line = raw_bytes.decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                self.latest_raw_line = raw_line

                if "warming up" in raw_line.lower():
                    self.status_message = "ESP32 Warming Up Sensor (~15-20s)..."
                    continue

                parsed = self.parse_raw_sensor_line(raw_line)
                if parsed:
                    with self._lock:
                        self.last_reading = parsed
                        self.status_message = f"Streaming Live Data: {parsed.temperature_C}°C | {parsed.humidity_RH}% RH"
            except Exception as e:
                self.status_message = f"Serial read error: {e}"
                time.sleep(0.2)

    def parse_raw_sensor_line(self, raw_line: str) -> Optional[SensorReading]:
        """Parses any ESP32 telemetry string format (JSON, MQ Analog, DHT key-value)"""
        if not raw_line:
            return None

        # 1. Try JSON format: {"temperature": 24.5, "humidity": 70.2}
        if raw_line.startswith("{") and raw_line.endswith("}"):
            try:
                data = json.loads(raw_line)
                if "error" in data:
                    return None
                temp = float(data.get("temperature", data.get("temp", data.get("temperature_C", 25.0))))
                hum = float(data.get("humidity", data.get("hum", data.get("humidity_RH", 70.0))))
                a0 = float(data.get("analog_a0", data.get("a0", None))) if ("analog_a0" in data or "a0" in data) else None
                d0 = int(data.get("digital_d0", data.get("d0", None))) if ("digital_d0" in data or "d0" in data) else None
                return SensorReading(temp, hum, analog_a0=a0, digital_d0=d0, raw_line=raw_line)
            except Exception:
                pass

        # 2. Try MQ Gas / Analog Sensor Format: "Analog A0: 646    Digital D0: 0"
        if "Analog" in raw_line or "A0:" in raw_line or "D0:" in raw_line:
            try:
                a0_match = re.search(r'(?:Analog\s*A0:|A0:)\s*(\d+(?:\.\d+)?)', raw_line, re.IGNORECASE)
                d0_match = re.search(r'(?:Digital\s*D0:|D0:)\s*(\d+)', raw_line, re.IGNORECASE)
                if a0_match:
                    a0_val = float(a0_match.group(1))
                    d0_val = int(d0_match.group(1)) if d0_match else 0
                    norm_val = min(1023.0, max(0.0, a0_val)) / 1023.0
                    temp = round(20.0 + norm_val * 12.0, 1)
                    hum = round(62.0 + (1.0 - norm_val) * 26.0, 1)
                    return SensorReading(temp, hum, analog_a0=a0_val, digital_d0=d0_val, raw_line=raw_line)
            except Exception:
                pass

        # 3. Try CSV / Key-Value: "24.5,70.2"
        try:
            cleaned = raw_line.replace("T=", "").replace("H=", "").replace("TEMP:", "").replace("HUM:", "").replace("C", "").replace("%", "")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            if len(parts) >= 2:
                temp = float(parts[0])
                hum = float(parts[1])
                return SensorReading(temp, hum, raw_line=raw_line)
        except Exception:
            pass

        return None

    def fetch_cloud_bridge_reading(self, topic: str = "freshroute_sv_gokul_esp32", timeout: float = 2.0) -> Tuple[Optional[SensorReading], str]:
        """Fetches the latest reading published to the cloud topic by local ESP32 bridge or Wi-Fi"""
        try:
            url = f"https://ntfy.sh/{topic}/json?poll=1&since=10m"
            req = urllib.request.Request(url, headers={"User-Agent": "Freshroute-Dashboard/2.5"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode("utf-8")
                lines = [json.loads(l) for l in content.strip().split("\n") if l.strip()]
                if not lines:
                    return None, "No recent messages in cloud topic. Run 'run_cloud_bridge.bat' locally."

                # The latest message is the last element
                last_entry = lines[-1]
                msg_raw = last_entry.get("message", "")
                if msg_raw.startswith("{") and msg_raw.endswith("}"):
                    data = json.loads(msg_raw)
                else:
                    data = last_entry

                temp = float(data.get("temperature_C", data.get("temp", data.get("temperature", 25.0))))
                hum = float(data.get("humidity_RH", data.get("hum", data.get("humidity", 70.0))))
                a0 = float(data.get("analog_a0", data.get("a0", None))) if ("analog_a0" in data or "a0" in data) else None
                d0 = int(data.get("digital_d0", data.get("d0", None))) if ("digital_d0" in data or "d0" in data) else None
                ts = float(data.get("timestamp", data.get("ts", time.time())))

                reading = SensorReading(temp_c=temp, humidity_rh=hum, timestamp=ts, analog_a0=a0, digital_d0=d0, raw_line=msg_raw or str(data))
                with self._lock:
                    self.last_reading = reading
                    self.latest_raw_line = msg_raw or str(data)
                    self.status_message = f"Cloud Live Stream: {temp}°C | {hum}% RH"
                return reading, "Success"
        except Exception as e:
            return None, f"Cloud Fetch Error: {str(e)}"

    def read_serial_line(self) -> Optional[SensorReading]:
        """Returns the most recent reading received by the background reader"""
        with self._lock:
            return self.last_reading

    def fetch_http_reading(self, url: str, timeout: float = 2.0) -> Tuple[Optional[SensorReading], str]:
        """Fetches telemetry over Wi-Fi / HTTP from ESP32 web server endpoint"""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Freshroute-Dashboard/2.5", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode("utf-8")
                data = json.loads(content)
                temp = float(data.get("temperature", data.get("temp", data.get("temperature_C", 25.0))))
                hum = float(data.get("humidity", data.get("hum", data.get("humidity_RH", 70.0))))
                reading = SensorReading(temp, hum, raw_line=content)
                with self._lock:
                    self.last_reading = reading
                return reading, "Success"
        except Exception as e:
            return None, f"HTTP Fetch Error: {str(e)}"

    def generate_simulated_reading(self) -> SensorReading:
        """Generates realistic environmental fluctuations simulating transit conditions"""
        self.sim_step_count += 1
        t_phase = self.sim_step_count * 0.15

        temp_wave = math.sin(t_phase) * 1.5 + (random.random() - 0.5) * 0.4
        hum_wave = math.cos(t_phase) * 2.5 + (random.random() - 0.5) * 0.8

        if self.sim_anomaly == "COOLING_FAILURE":
            temp = self.sim_base_temp + min(15.0, self.sim_step_count * 0.5) + temp_wave
            hum = max(40.0, self.sim_base_humidity - self.sim_step_count * 0.3) + hum_wave
        elif self.sim_anomaly == "HEATWAVE":
            temp = self.sim_base_temp + 8.5 + temp_wave
            hum = max(45.0, self.sim_base_humidity - 15.0) + hum_wave
        elif self.sim_anomaly == "HIGH_HUMIDITY":
            temp = self.sim_base_temp + temp_wave
            hum = min(98.0, self.sim_base_humidity + 18.0) + hum_wave
        else:  # NORMAL
            temp = self.sim_base_temp + temp_wave
            hum = min(98.0, max(30.0, self.sim_base_humidity + hum_wave))

        reading = SensorReading(temp_c=temp, humidity_rh=hum)
        with self._lock:
            self.last_reading = reading
        return reading

    def get_latest_reading(
        self,
        shipment_id: str,
        manual_temp: Optional[float] = None,
        manual_hum: Optional[float] = None
    ) -> SensorReading:
        """Fetch latest reading based on active telemetry mode"""
        if manual_temp is not None and manual_hum is not None:
            reading = SensorReading(temp_c=manual_temp, humidity_rh=manual_hum)
            with self._lock:
                self.last_reading = reading
            return reading

        if self.active_mode == "SERIAL" and self.is_connected:
            serial_reading = self.read_serial_line()
            if serial_reading:
                return serial_reading
            elif self.last_reading:
                return self.last_reading

        return self.generate_simulated_reading()
