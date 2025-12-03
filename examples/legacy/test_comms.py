from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from contextlib import suppress
from typing import Any, Dict

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_comms.py
# Import LIFU modules
from openlifu.io.LIFUInterface import LIFUInterface

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---- Globals ----
CURRENT_IFC: dict[str, LIFUInterface | None] = {"iface": None}

# Signal handling for graceful exit
def signal_handler(_signum: int, _frame: Any) -> None:
    print("\n\n🛑 Stopping test gracefully...")
    try:
        if CURRENT_IFC["iface"] is not None:
            CURRENT_IFC["iface"].close()
            logging.info("Interface safely closed.")
    except Exception as e:
        logging.warning(f"Error closing interface: {e}")
    finally:
        CURRENT_IFC["iface"] = None
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Constants for test validation
EXPECTED_VERSION = ["v2.0.1", "v1.0.14"]
EXPECTED_TX7332_COUNT = 2
TEMP_MIN = 20.0
TEMP_MAX = 50.0
FREQ_MIN = 10
FREQ_MAX = 40
PULSE_WIDTH_MIN = 500
PULSE_WIDTH_MAX = 5000
PULSE_INTERVAL_MIN = 125000
PULSE_INTERVAL_MAX = 500000
PULSE_COUNT_MIN = 1
PULSE_COUNT_MAX = 10
PULSE_TRAIN_COUNT_MIN = 1
PULSE_TRAIN_COUNT_MAX = 10

# Trigger modes (use actual values from your code)
TRIGGER_MODE_SEQUENCE = 1
TRIGGER_MODE_CONTINUOUS = 2
TRIGGER_MODE_SINGLE = 3

# Test parameters
TEST_ITERATIONS = int(os.getenv("TEST_ITERATIONS", "5"))
TIMEOUT = int(os.getenv("TEST_TIMEOUT", "3"))

# Helper: Safe execution (no concurrency). Logs duration and errors, returns None on failure.
def safe_execute(func, description: str, *args, **kwargs) -> Any:
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed > TIMEOUT:
            logging.warning(f"⚠ '{description}' took {elapsed:.2f}s (> {TIMEOUT}s)")
        else:
            logging.info(f"✅ '{description}' completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        logging.error(f"❌ Failed to {description}: {e}")
        return None

# Helper: Check if value is in valid range
def is_in_range(value: float, min_val: float, max_val: float) -> bool:
    return min_val <= value <= max_val

# Helper: Generate random valid trigger parameters
def generate_random_trigger_params() -> Dict[str, Any]:
    return {
        "TriggerFrequencyHz": random.randint(FREQ_MIN, FREQ_MAX),
        "TriggerPulseCount": random.randint(PULSE_COUNT_MIN, PULSE_COUNT_MAX),
        "TriggerPulseWidthUsec": random.randint(PULSE_WIDTH_MIN, PULSE_WIDTH_MAX),
        "TriggerPulseTrainInterval": random.randint(PULSE_INTERVAL_MIN, PULSE_INTERVAL_MAX),
        "TriggerPulseTrainCount": random.randint(PULSE_TRAIN_COUNT_MIN, PULSE_TRAIN_COUNT_MAX),
        "TriggerMode": random.choice([TRIGGER_MODE_SEQUENCE, TRIGGER_MODE_CONTINUOUS, TRIGGER_MODE_SINGLE]),
        "ProfileIndex": 0,
        "ProfileIncrement": 0
    }

# Helper: Compare two dicts (allowing small float differences)
def dicts_equal_with_tolerance(d1: Dict, d2: Dict, tolerance: float = 1e-6) -> bool:
    if set(d1.keys()) != set(d2.keys()):
        return False
    for k, v1 in d1.items():
        v2 = d2.get(k)
        if isinstance(v1, float) and isinstance(v2, float):
            if abs(v1 - v2) > tolerance:
                return False
        elif v1 != v2:
            return False
    return True
def _as_int(x):
    try:
        return int(x)
    except Exception:
        return x

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return x

# Compare only the fields we set, allow device-side coercions and small float diffs
def trigger_dict_matches_set(set_params: Dict[str, Any],
                             reply: Dict[str, Any],
                             tol: float = 1e-6) -> bool:
    if not isinstance(reply, dict):
        return False
    keys_to_check = [
        "TriggerFrequencyHz",
        "TriggerPulseCount",
        "TriggerPulseWidthUsec",
        "TriggerPulseTrainInterval",
        "TriggerPulseTrainCount",
        "TriggerMode",
        "ProfileIndex",
        "ProfileIncrement",
    ]
    for k in keys_to_check:
        v_set = set_params.get(k)
        v_rep = reply.get(k)
        # allow numeric type coercions
        if isinstance(v_set, float) or isinstance(v_rep, float):
            if abs(_as_float(v_set) - _as_float(v_rep)) > tol:
                return False
        elif _as_int(v_set) != _as_int(v_rep):
            return False
    return True

# Main test function
def run_tx_test() -> int:
    print("🚀 Starting LIFU Transmitter Module Test Script...")
    test_results = {"total": 0, "passed": 0, "failed": 0}

    try:
        # Initialize interface
        CURRENT_IFC["iface"] = LIFUInterface()
        tx_connected, hv_connected = CURRENT_IFC["iface"].is_device_connected()
        if not tx_connected and hv_connected:
            print("❌ LIFU Device not fully connected. Attempting to power 12V...")
            with suppress(Exception):
                CURRENT_IFC["iface"].hvcontroller.turn_12v_on()
            time.sleep(2)

            # If there's monitoring that interferes with reconnection, stop it (ignore if missing)
            with suppress(Exception):
                CURRENT_IFC["iface"].stop_monitoring()

            # Re-init
            with suppress(Exception):
                CURRENT_IFC["iface"].close()
            CURRENT_IFC["iface"] = None
            time.sleep(3)
            print("🔄 Reinitializing after 12V power-on...")
            CURRENT_IFC["iface"] = LIFUInterface()
            tx_connected, hv_connected = CURRENT_IFC["iface"].is_device_connected()
            if not tx_connected or not hv_connected:
                print(f"❌ Still not connected. TX: {tx_connected}, HV: {hv_connected}")
                return 1

        if not tx_connected:
            print("❌ TX device not connected. Cannot proceed with tests.")
            return 1

        print("✅ LIFU Transmitter Device connected.")

        # Basic connectivity tests
        print(f"\n🔍 Testing: Transmitter (iterating {TEST_ITERATIONS} times)...")
        for i in range(TEST_ITERATIONS):
            print(f"  ➤ Iteration {i+1}/{TEST_ITERATIONS}...")
            params = generate_random_trigger_params()

            # Test 1: Ping
            print("\n🔍 Testing: Ping...")
            ping_result = safe_execute(CURRENT_IFC["iface"].txdevice.ping, "ping")
            if ping_result is True:
                print("✅ Ping successful (expected: always True)")
                test_results["passed"] += 1
            else:
                print("❌ Ping failed (should always return True)")
                test_results["failed"] += 1
            test_results["total"] += 1

            # Test 2: Get Version
            print("\n🔍 Testing: Get Version...")
            version = safe_execute(CURRENT_IFC["iface"].txdevice.get_version, "get version")
            if version in EXPECTED_VERSION:
                print(f"✅ Version correct: {version}")
                test_results["passed"] += 1
            else:
                print(f"❌ Version mismatch: expected '{EXPECTED_VERSION}', got '{version}'")
                test_results["failed"] += 1
            test_results["total"] += 1

            # Test 3: Enumerate TX7332 Devices
            print("\n🔍 Testing: Enumerate TX7332 Devices...")
            num_devices = safe_execute(CURRENT_IFC["iface"].txdevice.enum_tx7332_devices, "enum devices")
            if num_devices == EXPECTED_TX7332_COUNT:
                print(f"✅ Found exactly {num_devices} TX7332 devices (expected: {EXPECTED_TX7332_COUNT})")
                test_results["passed"] += 1
            else:
                print(f"❌ Wrong number of devices: expected {EXPECTED_TX7332_COUNT}, got {num_devices}")
                test_results["failed"] += 1
            test_results["total"] += 1

            # Test 4: Temperature
            print("\n🔍 Testing: Temperature...")
            temp = safe_execute(CURRENT_IFC["iface"].txdevice.get_temperature, "get temperature")
            if temp is not None and is_in_range(temp, TEMP_MIN, TEMP_MAX):
                print(f"✅ Temperature: {temp:.1f}°C (in valid range: {TEMP_MIN}-{TEMP_MAX}°C)")
                test_results["passed"] += 1
            else:
                print(f"❌ Temperature out of range: {temp}°C (expected: {TEMP_MIN}-{TEMP_MAX}°C)")
                test_results["failed"] += 1
            test_results["total"] += 1

            # Test 5: Set & Get Trigger (loop over iterations)
            print(f"    Setting trigger: {json.dumps(params, indent=2)}")
            try:
                set_reply = safe_execute(
                    CURRENT_IFC["iface"].txdevice.set_trigger_json,
                    "set trigger",
                    data=params
                )
            except TypeError:
                # If API expects a positional dict
                set_reply = safe_execute(lambda p=params: CURRENT_IFC["iface"].txdevice.set_trigger_json(p), "set trigger")

            # Interpret success: dict reply means success; True also okay (legacy)
            if isinstance(set_reply, dict) or set_reply is True:
                if isinstance(set_reply, dict):
                    if trigger_dict_matches_set(params, set_reply):
                        print("    ✅ Trigger set successfully (echo validated).")
                    else:
                        print("    ⚠ Trigger set returned but values differ after device coercion:")
                        print("       ", json.dumps(set_reply, indent=2))
                else:
                    print("    ✅ Trigger set successfully.")
            else:
                print(f"    ❌ Failed to set trigger: {set_reply}")
                test_results["failed"] += 1
                test_results["total"] += 1
                continue  # skip read-back on failure


            # Read back trigger
            get_result = safe_execute(CURRENT_IFC["iface"].txdevice.get_trigger_json, "get trigger")
            if get_result is None:
                print("    ❌ Failed to read trigger after set.")
                test_results["failed"] += 1
            else:
                print(f"    Read back: {json.dumps(get_result, indent=2)}")

                # Validate all fields
                passed = True
                for key in params:
                    if key == "TriggerMode":
                        if get_result.get(key) != params[key]:
                            print(f"    ❌ TriggerMode mismatch: expected {params[key]}, got {get_result.get(key)}")
                            passed = False
                    elif key in [
                        "TriggerFrequencyHz", "TriggerPulseCount", "TriggerPulseWidthUsec",
                        "TriggerPulseTrainInterval", "TriggerPulseTrainCount"
                    ]:
                        val1 = params[key]
                        val2 = get_result.get(key)
                        if val2 is None or abs(float(val1) - float(val2)) > 1e-6:
                            print(f"    ❌ {key} mismatch: expected {val1}, got {val2}")
                            passed = False
                    else:
                        # Skip profile fields for now
                        continue

                if passed:
                    print("    ✅ All trigger values match!")
                    test_results["passed"] += 1
                else:
                    test_results["failed"] += 1
            test_results["total"] += 1

        # Final Summary
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print(f"Total Tests: {test_results['total']}")
        print(f"Passed:      {test_results['passed']}")
        print(f"Failed:      {test_results['failed']}")
        print("="*70)

        if test_results["failed"] == 0:
            print("🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("❌ Some tests failed.")
            return 1

    except Exception as e:
        logging.critical(f"Unexpected error during test: {e}")
        return 1

    finally:
        if CURRENT_IFC["iface"] is not None:
            with suppress(Exception):
                CURRENT_IFC["iface"].close()
                logging.info("Interface safely closed.")
            CURRENT_IFC["iface"] = None

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Test LIFU Transmitter Module")
    parser.add_argument('--iterations', type=int, default=5, help='Number of trigger test iterations')
    parser.add_argument('--timeout', type=int, default=3, help='Soft timeout (seconds) for warning logs')
    args = parser.parse_args()

    # Override defaults from environment or args
    TEST_ITERATIONS = args.iterations
    TIMEOUT = args.timeout

    # Run test
    sys.exit(run_tx_test())
