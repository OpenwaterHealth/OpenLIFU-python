from __future__ import annotations

import random

from openlifu.io.LIFUInterface import LIFUInterface


def run_test(interface, iterations):
    """
    Run the LIFU test loop with random trigger settings.

    Args:
        interface (LIFUInterface): The LIFUInterface instance.
        iterations (int): Number of iterations to run.
    """
    for i in range(iterations):
        print(f"Starting Test Iteration {i + 1}/{iterations}...")

        try:
            tx_connected, hv_connected = interface.is_device_connected()
            if not tx_connected: # or not hv_connected:
                raise ConnectionError(f"LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}")

            print("Ping the device")
            interface.txdevice.ping()

            print("Toggle LED")
            interface.txdevice.toggle_led()

            print("Get Version")
            version = interface.txdevice.get_version()
            print(f"Version: {version}")

            print("Echo Data")
            echo, length = interface.txdevice.echo(echo_data=b'Hello LIFU!')
            if length > 0:
                print(f"Echo: {echo.decode('utf-8')}")
            else:
                raise ValueError("Echo failed.")

            print("Get HW ID")
            hw_id = interface.txdevice.get_hardware_id()
            print(f"HWID: {hw_id}")

            print("Get Temperature")
            temperature = interface.txdevice.get_temperature()
            print(f"Temperature: {temperature} °C")

            print("Get Trigger")
            trigger_setting = interface.txdevice.get_trigger()
            if trigger_setting:
                print(f"Trigger Setting: {trigger_setting}")
            else:
                raise ValueError("Failed to get trigger setting.")

            print("Set Trigger with Random Parameters")
            # Generate random trigger frequency and pulse width
            trigger_frequency = random.randint(5, 25)  # Random frequency between 5 and 25 Hz
            trigger_pulse_width = random.randint(10, 30) * 1000  # Random pulse width between 10 and 30 ms (convert to µs)

            json_trigger_data = {
                "TriggerFrequencyHz": trigger_frequency,
                "TriggerPulseCount": 0,
                "TriggerPulseWidthUsec": trigger_pulse_width,
                "TriggerPulseTrainInterval": 0,
                "TriggerPulseTrainCount": 0,
                "TriggerMode": 1,
                "ProfileIndex": 0,
                "ProfileIncrement": 0
            }
            trigger_setting = interface.txdevice.set_trigger_json(data=json_trigger_data)

            trigger_setting = interface.txdevice.get_trigger_json()
            if trigger_setting:
                print(f"Trigger Setting Applied: Frequency = {trigger_frequency} Hz, Pulse Width = {trigger_pulse_width // 1000} ms")
                if trigger_setting["TriggerFrequencyHz"] != trigger_frequency or trigger_setting["TriggerPulseWidthUsec"] != trigger_pulse_width:
                    raise ValueError("Failed to set trigger setting.")
            else:
                raise ValueError("Failed to set trigger setting.")

            print(f"Iteration {i + 1} passed.\n")

        except Exception as e:
            print(f"Test failed on iteration {i + 1}: {e}")
            break

if __name__ == "__main__":
    print("Starting LIFU Test Script...")
    interface = LIFUInterface()

    # Number of iterations to run
    test_iterations = 1000  # Change this to the desired number of iterations

    run_test(interface, test_iterations)
