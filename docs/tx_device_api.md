# TxDevice API Documentation

The `TxDevice` class provides a high-level interface for communicating with and
controlling ultrasound transmitter modules using the `LIFUUart` communication
backend.

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

if not tx_connected:
    print("TX Device not connected.")
    sys.exit()
```

---

## Core Methods

### Device Communication

| Method              | Description                                        |
| ------------------- | -------------------------------------------------- |
| `is_connected()`    | Check if the TX device is connected                |
| `ping()`            | Send a ping command to check connectivity          |
| `get_version()`     | Get the firmware version (e.g., `v0.1.1`)          |
| `echo(data: bytes)` | Send and receive echo data to verify communication |
| `toggle_led()`      | Toggle the device onboard LED                      |
| `get_hardware_id()` | Return the 16-byte hardware ID in hex format       |
| `soft_reset()`      | Perform a software reset on the device             |
| `enter_dfu()`       | Put the device into DFU mode                       |

---

### Trigger Configuration

| Method                               | Description                                 |
| ------------------------------------ | ------------------------------------------- |
| `set_trigger(...)`                   | Configure triggering with manual parameters |
| `set_trigger_json(data: dict)`       | Set trigger via JSON dictionary             |
| `get_trigger()`                      | Return current trigger config as a dict     |
| `get_trigger_json()`                 | Retrieve raw JSON trigger data              |
| `start_trigger()` / `stop_trigger()` | Begin or halt triggering                    |

---

### Register Operations

| Method                                        | Description                      |
| --------------------------------------------- | -------------------------------- |
| `write_register(identifier, addr, value)`     | Write to a single register       |
| `write_register_verify(addr, value)`          | Write and verify a register      |
| `read_register(addr)`                         | Read a register value            |
| `write_block(identifier, start_addr, values)` | Write a block of register values |
| `write_block_verify(start_addr, values)`      | Verified block write             |

---

### Device Setup

| Method                                   | Description                               |
| ---------------------------------------- | ----------------------------------------- |
| `enum_tx7332_devices(num)`               | Scan for TX7332 devices                   |
| `demo_tx7332(identifier)`                | Set test waveform to TX7332               |
| `apply_all_registers()`                  | Apply all configured profiles to hardware |
| `write_ti_config_to_tx_device(path, id)` | Load and apply config from TI text file   |
| `print`                                  | Print TX interface info                   |

---

### Pulse/Delay Profiles

| Method                                         | Description                         |
| ---------------------------------------------- | ----------------------------------- |
| `set_solution(pulse, delays, apods, seq, ...)` | Apply full beamforming config       |
| `add_pulse_profile(profile)`                   | Add pulse shape settings            |
| `add_delay_profile(profile)`                   | Add delay+apodization configuration |

---

## Data Classes

- `Tx7332PulseProfile`: Defines frequency, cycles, duty cycle, inversion, etc.
- `Tx7332DelayProfile`: Defines delays and apodization for each channel
- `TxDeviceRegisters`: Holds and manages TX chip register blocks per transmitter

---

## Example

```python
pulse = {"frequency": 3e6, "duration": 2e-6}
delays = [0] * 32
apods = [1] * 32
sequence = {
    "pulse_interval": 0.01,
    "pulse_count": 1,
    "pulse_train_interval": 0.0,
    "pulse_train_count": 1,
}

# Apply configuration
if tx.is_connected():
    tx.set_solution(pulse, delays, apods, sequence)
    tx.start_trigger()
```

---

## Notes

- Profile management assumes 32 transmit channels per chip.
- Delay units default to seconds unless specified.
- Device must be enumerated before applying register values.
