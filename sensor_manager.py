import time
import json
import random
import math
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
    """Represents a single timestamped sensor reading"""
    def __init__(self, temp_c: float, humidity_rh: float, timestamp: Optional[float] = None):
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.temperature_C = float(temp_c)
        self.humidity_RH = float(humidity_rh)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "temperature_C": round(self.temperature_C, 2),
            "humidity_RH": round(self.humidity_RH, 2),
            "delta_t_days": 1.0 / 24.0  # Normalized step size
        }


class ESP32SensorManager:
    """
    Manages real hardware ESP32 Serial COM connections, Wi-Fi HTTP polling,
    and a high-fidelity real-time environmental simulation engine.
    """
    def __init__(self):
        self.active_mode = "SIMULATOR"  # "SERIAL", "HTTP", "SIMULATOR", "MANUAL"
        self.serial_port: Optional[str] = None
        self.baud_rate: int = 115200
        self.serial_conn: Optional[Any] = None
        self.is_connected: bool = False
        self.http_url: str = "http://192.168.1.100/data"
        self.last_reading: Optional[SensorReading] = None
        self.shipment_history: Dict[str, List[SensorReading]] = {}

        # Simulator State
        self.sim_base_temp: float = 24.0
        self.sim_base_humidity: float = 72.0
        self.sim_anomaly: str = "NORMAL"  # "NORMAL", "COOLING_FAILURE", "HEATWAVE", "HIGH_HUMIDITY"
        self.sim_step_count: int = 0

        # Lock for thread safety
        self._lock = threading.Lock()

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """Scans and lists system serial COM ports with device descriptions"""
        if not SERIAL_AVAILABLE:
            return []
        try:
            results = []
            for p in serial.tools.list_ports.comports():
                desc = f"{p.device} ({p.description})"
                results.append({"port": p.device, "description": desc, "hwid": getattr(p, "hwid", "")})
            return results
        except Exception:
            return []

    def connect_serial(self, port: str, baud_rate: int = 115200) -> Tuple[bool, str]:
        """Connects to physical ESP32 via USB Serial COM port"""
        if not SERIAL_AVAILABLE:
            return False, "pyserial library not installed."

        try:
            self.disconnect_serial()
            self.serial_conn = serial.Serial(port, baud_rate, timeout=1.0)
            self.serial_conn.reset_input_buffer()
            self.serial_port = port
            self.baud_rate = baud_rate
            self.is_connected = True
            self.active_mode = "SERIAL"
            return True, f"Successfully connected to ESP32 on {port} @ {baud_rate} baud."
        except Exception as e:
            self.is_connected = False
            return False, f"Failed to open {port}: {str(e)}"

    def disconnect_serial(self):
        """Closes serial connection safely"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None
        self.is_connected = False

    def read_serial_line(self) -> Optional[SensorReading]:
        """
        Drains all available lines in the serial buffer to return the freshest reading
        from the ESP32 DHT sensor stream.
        """
        if not self.is_connected or not self.serial_conn:
            return None

        latest_parsed: Optional[SensorReading] = None

        try:
            while self.serial_conn.in_waiting > 0:
                raw_bytes = self.serial_conn.readline()
                raw_line = raw_bytes.decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                # 1. Try JSON format: {"temperature": 24.5, "humidity": 70.2}
                if raw_line.startswith("{") and raw_line.endswith("}"):
                    try:
                        data = json.loads(raw_line)
                        temp = float(data.get("temperature", data.get("temp", data.get("temperature_C", 25.0))))
                        hum = float(data.get("humidity", data.get("hum", data.get("humidity_RH", 70.0))))
                        latest_parsed = SensorReading(temp, hum)
                        continue
                    except Exception:
                        pass

                # 2. Try CSV / Key-Value: "24.5,70.2" or "T=24.5,H=70.2" or "TEMP:24.5,HUM:70.2"
                cleaned = raw_line.replace("T=", "").replace("H=", "").replace("TEMP:", "").replace("HUM:", "").replace("C", "").replace("%", "")
                parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                if len(parts) >= 2 and not "Analog" in raw_line:
                    try:
                        temp = float(parts[0])
                        hum = float(parts[1])
                        latest_parsed = SensorReading(temp, hum)
                        continue
                    except ValueError:
                        pass

                # 3. Try Analog / Digital Sensor Format: "Analog A0: 511    Digital D0: 0"
                if "Analog" in raw_line or "A0:" in raw_line:
                    try:
                        import re
                        a0_match = re.search(r'(?:Analog\s*A0:|A0:)\s*(\d+(?:\.\d+)?)', raw_line, re.IGNORECASE)
                        if a0_match:
                            a0_val = float(a0_match.group(1))
                            # Normalize analog 0-1023/4095 range to realistic ambient chamber temp & RH
                            norm_val = min(1023.0, a0_val) / 1023.0
                            temp = round(18.0 + norm_val * 14.0, 1)  # 18°C to 32°C range
                            hum = round(65.0 + (1.0 - norm_val) * 25.0, 1) # 65% to 90% RH
                            latest_parsed = SensorReading(temp, hum)
                            continue
                    except Exception:
                        pass
        except Exception as e:
            print(f"[SensorManager] Serial read error: {e}")

        if latest_parsed is not None:
            self.last_reading = latest_parsed
            return latest_parsed
        return self.last_reading

    def fetch_http_reading(self, url: str, timeout: float = 2.0) -> Tuple[Optional[SensorReading], str]:
        """Fetches telemetry over Wi-Fi / HTTP from ESP32 web server endpoint"""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AgriLife-Dashboard/2.4", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode("utf-8")
                data = json.loads(content)
                temp = float(data.get("temperature", data.get("temp", data.get("temperature_C", 25.0))))
                hum = float(data.get("humidity", data.get("hum", data.get("humidity_RH", 70.0))))
                reading = SensorReading(temp, hum)
                self.last_reading = reading
                return reading, "Success"
        except Exception as e:
            return None, f"HTTP Fetch Error: {str(e)}"

    def generate_simulated_reading(self) -> SensorReading:
        """
        Generates realistic environmental fluctuations simulating transit conditions
        """
        self.sim_step_count += 1
        t_phase = self.sim_step_count * 0.15

        # Base sinusoidal ambient wave
        temp_wave = math.sin(t_phase) * 1.5 + (random.random() - 0.5) * 0.4
        hum_wave = math.cos(t_phase) * 2.5 + (random.random() - 0.5) * 0.8

        if self.sim_anomaly == "COOLING_FAILURE":
            # Rising temperature rapidly, drying out or condensation
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
            self.last_reading = reading
            return reading

        if self.active_mode == "SERIAL" and self.is_connected:
            serial_reading = self.read_serial_line()
            if serial_reading:
                self.last_reading = serial_reading
                return serial_reading
            elif self.last_reading:
                return self.last_reading

        # Fallback / Default Simulator mode
        return self.generate_simulated_reading()

    def record_reading_for_shipment(self, shipment_id: str, reading: SensorReading):
        """Appends reading to shipment time-series buffer"""
        with self._lock:
            if shipment_id not in self.shipment_history:
                self.shipment_history[shipment_id] = []
            self.shipment_history[shipment_id].append(reading)
            if len(self.shipment_history[shipment_id]) > 120:
                self.shipment_history[shipment_id] = self.shipment_history[shipment_id][-120:]

    def get_shipment_history_dicts(self, shipment_id: str) -> List[Dict[str, Any]]:
        """Returns list of dict readings for ML/physics engine processing"""
        with self._lock:
            history = self.shipment_history.get(shipment_id, [])
            return [r.to_dict() for r in history]
