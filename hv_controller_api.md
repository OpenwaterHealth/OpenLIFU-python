# HVController API Documentation

The `HVController` class provides an interface to control and monitor a
high-voltage console device over UART using the `LIFUUart` interface.

---

## Initialization

```python
from openlifu.io.LIFUHVController import HVController
from openlifu.io.LIFUUart import LIFUUart

interface = LIFUInterface(TX_test_mode=False)
tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f"LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}")

if not hv_connected:
    print("HV Controller not connected.")
    sys.exit()
```

---

## Methods

### Device Info & Connectivity

| Method              | Description                                     |
| ------------------- | ----------------------------------------------- |
| `is_connected()`    | Check if UART is connected                      |
| `ping()`            | Sends a ping to check device responsiveness     |
| `get_version()`     | Returns firmware version as `vX.Y.Z`            |
| `get_hardware_id()` | Returns the 16-byte hardware ID as a hex string |
| `echo(data: bytes)` | Echoes back sent data, useful for testing       |
| `soft_reset()`      | Sends a soft reset to the device                |
| `enter_dfu()`       | Put the device into DFU mode                    |

---

### Voltage Control

| Method                           | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `set_voltage(voltage: float)`    | Sets output voltage (5V–100V range)          |
| `get_voltage()`                  | Reads and returns the current output voltage |
| `turn_hv_on()` / `turn_hv_off()` | Turns high voltage supply ON or OFF          |
| `get_hv_status()`                | Returns current HV ON/OFF status             |

---

### 12V Control

| Method                             | Description                               |
| ---------------------------------- | ----------------------------------------- |
| `turn_12v_on()` / `turn_12v_off()` | Controls 12V auxiliary power              |
| `get_12v_status()`                 | Reads the ON/OFF status of the 12V supply |

---

### Temperature Monitoring

| Method               | Description                            |
| -------------------- | -------------------------------------- |
| `get_temperature1()` | Returns temperature from sensor 1 (°C) |
| `get_temperature2()` | Returns temperature from sensor 2 (°C) |

---

### Fan Control

| Method                             | Description                                              |
| ---------------------------------- | -------------------------------------------------------- |
| `set_fan_speed(fan_id, fan_speed)` | Sets fan speed (0–100%) for fan ID 0 (bottom) or 1 (top) |
| `get_fan_speed(fan_id)`            | Reads current fan speed percentage                       |

---

### RGB LED Control

| Method               | Description                                 |
| -------------------- | ------------------------------------------- |
| `set_rgb_led(state)` | Sets RGB LED: 0=OFF, 1=RED, 2=BLUE, 3=GREEN |
| `get_rgb_led()`      | Gets current RGB LED state                  |

---

### Advanced Control

| Method                         | Description                               |
| ------------------------------ | ----------------------------------------- |
| `set_dacs(hvp, hvm, hrp, hrm)` | Sets DAC outputs for high voltage control |

---

## Notes

- Most methods raise `ValueError` if UART is not connected.
- All operations clear the UART buffer after use.
- Demo mode behavior returns mocked values.

---

## Example

```python
interface.hvcontroller.turn_12v_on()
interface.hvcontroller.set_voltage(60.0)
print(f"Output Voltage: {interface.hvcontroller.get_voltage()} V")
interface.hvcontroller.turn_hv_on()
```
