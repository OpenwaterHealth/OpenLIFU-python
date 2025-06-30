from __future__ import annotations

import asyncio
import logging
import threading
import time

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_async.py

# Setup logging
logging.basicConfig(level=logging.INFO)

interface = None

# Callbacks
def on_connect(descriptor, port):
    print(f"🔌 CONNECTED: {descriptor} on port {port}")

def on_disconnect(descriptor, port):
    print(f"❌ DISCONNECTED: {descriptor} from port {port}")

def on_data_received(descriptor, packet):
    print(f"📦 DATA [{descriptor}]: {packet}")

def monitor_interface():
    """Run the device monitor loop in a separate thread using asyncio."""
    asyncio.run(interface.start_monitoring(interval=1))

def rebind_tx_callbacks():
    """Bind callbacks to the TX UART, if present."""
    if interface.txdevice and interface.txdevice.uart:
        interface.txdevice.uart.signal_connect.connect(on_connect)
        interface.txdevice.uart.signal_disconnect.connect(on_disconnect)
        interface.txdevice.uart.signal_data_received.connect(on_data_received)

def run_menu():
    while True:
        print("\n--- LIFU MENU ---")
        print("1. Turn ON 12V")
        print("2. Turn OFF 12V")
        print("3. Ping TX")
        print("4. Show Connection Status")
        print("5. Exit")
        choice = input("Enter choice: ").strip()

        tx_connected, hv_connected = interface.is_device_connected()

        if choice == "1":
            if hv_connected:
                print("⚡ Sending 12V ON...")
                interface.hvcontroller.turn_12v_on()
                time.sleep(2.0)
                print("🔄 Reinitializing TX...")
                rebind_tx_callbacks()
            else:
                print("⚠️ HV not connected.")

        elif choice == "2":
            if hv_connected:
                print("🛑 Sending 12V OFF...")
                interface.hvcontroller.turn_12v_off()
            else:
                print("⚠️ HV not connected.")

        elif choice == "3":
            if tx_connected:
                print("📡 Sending PING to TX...")
                resp = interface.txdevice.ping()
                if resp:
                    print("✅ TX responded to PING.")
                else:
                    print("❌ No response or error.")
            else:
                print("⚠️ TX not connected.")

        elif choice == "4":
            print("Status:")
            print(f"  TX: {'✅ Connected' if tx_connected else '❌ Not connected'}")
            print(f"  HV: {'✅ Connected' if hv_connected else '❌ Not connected'}")

        elif choice == "5":
            print("Exiting...")
            interface.stop_monitoring()
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    interface = LIFUInterface(HV_test_mode=False, run_async=False)

    # Bind callbacks for HV and (initially connected) TX
    if interface.hvcontroller.uart:
        interface.hvcontroller.uart.signal_connect.connect(on_connect)
        interface.hvcontroller.uart.signal_disconnect.connect(on_disconnect)
        interface.hvcontroller.uart.signal_data_received.connect(on_data_received)

    rebind_tx_callbacks()

    print("🔍 Starting LIFU monitoring...")
    monitor_thread = threading.Thread(target=monitor_interface, daemon=True)
    monitor_thread.start()

    try:
        run_menu()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        interface.stop_monitoring()
