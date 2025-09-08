**API Documentation** for the High-Level LIFU interface `LIFUInterface` based on
the contents of `LIFUInterface.py`, `LIFUHVController.py`, and
`LIFUTXDevice.py`.

---

# `LIFUInterface` API Documentation

The `LIFUInterface` class provides a high-level interface for managing the LIFU
system, including the **TX transmitter module** and the **high voltage (HV)
controller**. It encapsulates all communication and configuration for loading
ultrasound solutions, performing hardware checks, and controlling sonication
operations.

---

## Initialization

```python
interface = LIFUInterface()
tx_connected, hv_connected = interface.is_device_connected()

if not tx_connected:
    print("TX device not connected. Attempting to turn on 12V...")
    interface.hvcontroller.turn_12v_on()

    # Give time for the TX device to power up and enumerate over USB
    time.sleep(2)

    # Cleanup and recreate interface to reinitialize USB devices
    interface.stop_monitoring()
    del interface
    time.sleep(1)  # Short delay before recreating

    print("Reinitializing LIFU interface after powering 12V...")
    interface = LIFUInterface()

    # Re-check connection
    tx_connected, hv_connected = interface.is_device_connected()

if tx_connected and hv_connected:
    print("✅ LIFU Device fully connected.")
else:
    print("❌ LIFU Device NOT fully connected.")
    print(f"  TX Connected: {tx_connected}")
    print(f"  HV Connected: {hv_connected}")
    sys.exit(1)
```

---

## Core Components

- `txdevice`: Instance of `TxDevice` that controls the transmitter module.
- `hvcontroller`: Instance of `HVController` to control high voltage behavior.
- `signal_connect`, `signal_disconnect`, `signal_data_received`: Signals emitted
  by the UART layer (if async mode is enabled).

---

## Key Methods

### Device Monitoring

| Method                                  | Description                                             |
| --------------------------------------- | ------------------------------------------------------- |
| `start_monitoring(interval: int = 1)`   | Start asynchronous monitoring of USB connection status. |
| `stop_monitoring()`                     | Stop USB device monitoring.                             |
| `is_device_connected() -> (bool, bool)` | Returns a tuple `(tx_connected, hv_connected)`.         |

---

### Solution Management

| Method                                                                                                      | Description                                                                         |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `set_solution(solution, profile_index=1, profile_increment=True, trigger_mode="sequence", turn_hv_on=True)` | Load and apply a full ultrasound solution to both TX and HV.                        |
| `check_solution(solution)`                                                                                  | Validate the provided solution against voltage and safety limits.                   |
| `get_max_voltage(solution)`                                                                                 | Return the safe maximum voltage for a solution.                                     |
| `get_max_voltage_table()`                                                                                   | Return a Pandas DataFrame showing max voltage per duty cycle and sequence duration. |
| `get_sequence_duty_cycle(solution)`                                                                         | Compute the duty cycle from the sequence config.                                    |
| `get_sequence_duration(solution)`                                                                           | Compute the total sequence duration from the solution config.                       |

---

### Sonication Control

| Method               | Description                                       |
| -------------------- | ------------------------------------------------- |
| `start_sonication()` | Powers on HV and starts the pulse trigger for TX. |
| `stop_sonication()`  | Stops TX trigger and powers down HV.              |
| `set_status(status)` | Set the current system status enum.               |
| `get_status()`       | Retrieve current internal status enum.            |

---

### Context Manager Support

```python
with LIFUInterface() as interface:
    tx_connected, hv_connected = interface.is_device_connected()
    ...
```

---

## Status Enum

The `LIFUInterfaceStatus` enum provides internal state tracking:

| Status Constant      | Meaning                 |
| -------------------- | ----------------------- |
| `STATUS_COMMS_ERROR` | Communication failure   |
| `STATUS_SYS_OFF`     | System is off           |
| `STATUS_SYS_POWERUP` | System is powering on   |
| `STATUS_SYS_ON`      | System is on            |
| `STATUS_PROGRAMMING` | Programming in progress |
| `STATUS_READY`       | System ready            |
| `STATUS_NOT_READY`   | System not ready        |
| `STATUS_RUNNING`     | Sonication running      |
| `STATUS_FINISHED`    | Sonication finished     |
| `STATUS_ERROR`       | System in error state   |

---

## Example: Load and Trigger

```python
from openlifu.plan.solution import Solution

solution = Solution(
    name="Test",
    voltage=40,
    pulse={"frequency": 2e6, "duration": 2e-6},
    delays=[0] * 32,
    apodizations=[1] * 32,
    sequence={
        "pulse_interval": 0.01,
        "pulse_count": 1,
        "pulse_train_interval": 0.0,
        "pulse_train_count": 1,
    },
)

interface.set_solution(solution)
interface.start_sonication()
time.sleep(1)
interface.stop_sonication()
```

---

## Notes

- The TX and HV devices are initialized independently using their respective USB
  VID/PIDs.
- If devices are disconnected, `turn_12v_on()` can be used to power the TX
  device before retrying connection.
- Use `check_solution()` to verify safety constraints before applying any
  configuration.
- `Solution` objects can be converted to `dict` automatically by
  `LIFUInterface`.
