import asyncio

from openlifu.io.LIFUInterface import LIFUInterface


async def main():
    """
    Test script to automate:
    1. Connect to the device.
    2. Send a ping.
    3. Turn treatment on.
    4. Wait for 5 seconds.
    5. Turn treatment off.
    6. Send another ping.
    7. Exit.
    """
    print("Starting LIFU Test Script...")
    interface = LIFUInterface()

    # Connect signals for debugging
    def on_connected(port):
        print(f"[Connected] Device connected on {port}")

    def on_disconnected():
        print("[Disconnected] Device disconnected")

    def on_data_received(data):
        print(f"[Data Received] {data}")

    interface.connected.connect(on_connected)
    interface.disconnected.connect(on_disconnected)
    interface.data_received.connect(on_data_received)

    try:
        # Start monitoring in the background
        monitor_task = asyncio.create_task(interface.start_monitoring())  # Store the task reference
        print("Monitoring started...")

        # Wait for connection
        while not interface.uart.port:
            print("Waiting for device to connect...")
            await asyncio.sleep(1)

        # Send initial ping
        print("Sending initial ping...")
        interface.send_ping()
        await asyncio.sleep(2)  # Allow time for data reception

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
