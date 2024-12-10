import asyncio

from openlifu.io.LIFUInterface import LIFUInterface


async def main():
    """
    Test script to automate:
    1. Connect to the device.
    2. Test HVController: Turn HV on/off and check voltage.
    3. Test Device functionality.
    """
    print("Starting LIFU Test Script...")
    interface = LIFUInterface(test_mode=True)

    # Connect signals for debugging
    def on_connected(port):
        print(f"[Connected] Device connected on {port}")

    def on_disconnected():
        print("[Disconnected] Device disconnected")

    def on_data_received(data):
        print(f"[Data Received] {data}")

    interface.signal_connect.connect(on_connected)
    interface.signal_disconnect.connect(on_disconnected)
    interface.signal_data_received.connect(on_data_received)

    try:
        # Start monitoring in the background
        monitor_task = asyncio.create_task(interface.start_monitoring())
        print("Monitoring started...")

        # Wait for connection
        while not interface.is_device_connected():
            print("Waiting for device to connect...")
            await asyncio.sleep(1)

        if interface.is_device_connected():
            print("Device is connected.")
        else:
            print("Device is not connected.")
            return  # Exit if device isn't connected

        # Test HVController
        print("Testing HVController...")
        hv_controller = interface.hvcontroller

        try:
            hv_controller.turn_on()
            print("HV turned on successfully.")
            await asyncio.sleep(2)

            hv_controller.set_voltage(5.0)  # Set voltage to 5.0V
            print(f"HV Voltage set to {hv_controller.get_voltage():.2f}V.")
            await asyncio.sleep(2)

            hv_controller.turn_off()
            print("HV turned off successfully.")
        except Exception as e:
            print(f"Error during HVController testing: {e}")

        # Test Device functionality
        print("Testing Device...")
        device = interface.Device
        try:
            device.start_sonication()
            print("Sonication started.")
            await asyncio.sleep(5)
            device.stop_sonication()
            print("Sonication stopped.")
        except Exception as e:
            print(f"Error during Device testing: {e}")

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        print("Stopping monitoring...")
        interface.stop_monitoring()
        monitor_task.cancel()  # Cancel the background task if running
        print("LIFU Test Script completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test script interrupted.")
