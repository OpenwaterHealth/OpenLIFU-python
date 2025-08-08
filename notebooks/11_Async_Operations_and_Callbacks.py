# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# # 11: Asynchronous Operations and Callbacks
#
# This notebook demonstrates how to use `LIFUInterface` in an asynchronous manner. This allows your program to monitor hardware connection status and receive data in the background without blocking the main execution thread. This is particularly useful for:
# *   Graphical User Interfaces (GUIs) that need to remain responsive.
# *   Long-running applications that perform other tasks while keeping an eye on hardware events.
# *   Reacting immediately to device connection or disconnection.
#
# We will use Python's `asyncio` library for the asynchronous parts and `threading` to run the asyncio event loop in the background, making it suitable for a Jupyter notebook environment.

# ## 1. Imports

# +
from __future__ import annotations

import asyncio
import logging
import threading
import time

from openlifu.io.LIFUInterface import LIFUInterface

# -

# ## 2. Logging and Global State
# For observing events from callbacks and managing the interface and thread.

# +
logger = logging.getLogger("AsyncDemo")
logger.setLevel(logging.INFO)
logger.handlers.clear()
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'))
logger.addHandler(ch)
logger.propagate = False

# Global variables to hold the interface and monitoring thread
lifu_interface_async = None
monitor_thread_async = None
stop_monitor_event = threading.Event() # Used to signal the monitoring wrapper to stop
# -

# ## 3. Defining Callback Functions
#
# These functions will be called when specific events occur (e.g., device connected, data received).
# They will run in the thread context of the `LIFUInterface`'s internal event handling (often the thread running `start_monitoring`).

# +
def on_device_connected(descriptor: str, port: str):
    """Callback for when a device connects."""
    logger.info(f"🔌 EVENT: Device CONNECTED! Descriptor: '{descriptor}', Port: '{port}'")

def on_device_disconnected(descriptor: str, port: str):
    """Callback for when a device disconnects."""
    logger.info(f"❌ EVENT: Device DISCONNECTED! Descriptor: '{descriptor}', Port: '{port}'")

def on_data_received(descriptor: str, packet_data: bytes | str): # Data type might vary
    """Callback for when data is received from a device."""
    # Note: For OpenLIFU, direct data packets might be less common than command-response.
    # This callback is more relevant if devices send unsolicited status updates or stream data.
    logger.info(f"📦 EVENT: Data RECEIVED from '{descriptor}': {packet_data}")
# -

# ## 4. Initializing `LIFUInterface` and Binding Callbacks
#
# We initialize `LIFUInterface`. The `run_async` parameter in the constructor can set up internal async mechanisms if available, but the key is how `start_monitoring` is run.
#
# The callback signals are typically available on the `uart` attribute of the `hvcontroller` and `txdevice` objects.

# +
try:
    # run_async=True might prepare internal async structures if supported by the interface directly.
    # However, test_async.py uses run_async=False and manages the asyncio loop externally.
    # We will follow the model of managing the asyncio loop externally for clarity here.
    logger.info("Initializing LIFUInterface for asynchronous monitoring...")
    lifu_interface_async = LIFUInterface(run_async=False) # Let's manage loop explicitly

    # Bind callbacks to HVController's UART signals if available
    if lifu_interface_async.hvcontroller and hasattr(lifu_interface_async.hvcontroller, 'uart'):
        logger.info("Binding HVController UART signals...")
        lifu_interface_async.hvcontroller.uart.signal_connect.connect(on_device_connected)
        lifu_interface_async.hvcontroller.uart.signal_disconnect.connect(on_device_disconnected)
        lifu_interface_async.hvcontroller.uart.signal_data_received.connect(on_data_received)
    else:
        logger.warning("HVController UART or signals not available for binding.")

    # Bind callbacks to TXDevice's UART signals if available
    # This needs to be robust in case txdevice is not initially connected (e.g. 12V off)
    def bind_tx_callbacks():
        if lifu_interface_async and lifu_interface_async.txdevice and hasattr(lifu_interface_async.txdevice, 'uart'):
            logger.info("Binding TXDevice UART signals...")
            lifu_interface_async.txdevice.uart.signal_connect.connect(on_device_connected)
            lifu_interface_async.txdevice.uart.signal_disconnect.connect(on_device_disconnected)
            lifu_interface_async.txdevice.uart.signal_data_received.connect(on_data_received)
            return True
        else:
            logger.warning("TXDevice UART or signals not available for binding (is 12V on?).")
            return False

    bind_tx_callbacks() # Attempt initial bind

    logger.info("LIFUInterface initialized and callbacks bound (if devices available).")

except Exception as e:
    logger.exception(f"Error initializing LIFUInterface: {e}")
    lifu_interface_async = None
# -

# ## 5. Starting the Asynchronous Monitoring
#
# The `interface.start_monitoring()` method is an `async` function. To run it in the background from a synchronous environment like a Jupyter notebook, we'll execute it within an `asyncio.run()` call inside a separate thread.

# +
def async_monitoring_loop(interface: LIFUInterface):
    """Target function for the monitoring thread."""
    thread_name = threading.current_thread().name
    logger.info(f"Async monitoring thread '{thread_name}' started.")
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Prepare the monitoring task
        monitor_task = interface.start_monitoring(interval=1.0) # Check every 1 second

        # Run until stop_event is set or monitor_task completes (e.g. on error)
        # We need a way to also make start_monitoring itself cancellable or responsive to stop_event
        # A simpler way for this demo is to rely on interface.stop_monitoring() to halt it.
        loop.run_until_complete(monitor_task)

    except asyncio.CancelledError:
        logger.info(f"Monitoring task in '{thread_name}' was cancelled.")
    except Exception as e:
        logger.exception(f"Exception in async monitoring thread '{thread_name}': {e}")
    finally:
        if loop and not loop.is_closed(): # Ensure loop is closed
            loop.close()
        logger.info(f"Async monitoring thread '{thread_name}' finished.")


if lifu_interface_async:
    stop_monitor_event.clear() # Ensure event is clear
    logger.info("Starting background monitoring thread...")
    monitor_thread_async = threading.Thread(
        target=async_monitoring_loop,
        args=(lifu_interface_async, stop_monitor_event),
        name="LIFUMonitorThread",
        daemon=True # Allow main program to exit even if this thread is running
    )
    monitor_thread_async.start()
    logger.info("Background monitoring thread should now be running.")
    # Give it a moment to start up and potentially detect initial connections
    time.sleep(2)
else:
    logger.exception("LIFUInterface not initialized. Cannot start monitoring.")
# -

# ## 6. Interacting While Monitoring
#
# Now that monitoring is (potentially) running in the background, the main thread (this notebook cell) is free.
# Callbacks should trigger automatically if you, for example, physically connect/disconnect a device's USB.
# We can also try some simple interactions.

# +
if lifu_interface_async and monitor_thread_async and monitor_thread_async.is_alive():
    logger.info("--- Main thread interactions while monitoring ---")

    # Check current status
    tx_conn, hv_conn = lifu_interface_async.is_device_connected()
    logger.info(f"Current Status: TX Connected = {tx_conn}, HV Connected = {hv_conn}")

    # Try pinging if HV is connected
    if hv_conn and lifu_interface_async.hvcontroller:
        logger.info("Pinging HV controller from main thread...")
        try:
            if lifu_interface_async.hvcontroller.ping():
                logger.info("  HV Ping successful.")
            else:
                logger.info("  HV Ping failed.")
        except Exception as e:
            logger.exception(f"  HV Ping error: {e}")

    # Try pinging if TX is connected
    if tx_conn and lifu_interface_async.txdevice:
        logger.info("Pinging TX device from main thread...")
        try:
            if lifu_interface_async.txdevice.ping():
                logger.info("  TX Ping successful.")
            else:
                logger.info("  TX Ping failed.")
        except Exception as e:
            logger.exception(f"  TX Ping error: {e}")

    logger.info("\n--- Test Device Connection/Disconnection ---")
    logger.info("Try physically disconnecting and reconnecting one of the OpenLIFU USB devices.")
    logger.info("You should see 'Device DISCONNECTED' and 'Device CONNECTED' messages from the callbacks in the log output.")
    logger.info("(Waiting for a few seconds to observe any events...)")

    # If TX was not connected initially, try turning on 12V and rebinding (example)
    if not tx_conn and hv_conn and lifu_interface_async.hvcontroller:
        logger.info("TX was not connected. Attempting to turn on 12V to see if it appears...")
        try:
            lifu_interface_async.hvcontroller.turn_12v_on()
            logger.info("12V turned ON. Waiting for TX to enumerate and connect...")
            time.sleep(3) # Wait for TX to appear
            # The monitoring thread should detect the new TX device.
            # We might need to re-bind callbacks if the txdevice object was created late.
            # This is a common challenge with dynamically appearing devices.
            # LIFUInterface might handle re-binding internally, or we might need to call bind_tx_callbacks()
            # after a short delay if a connection event for TX is seen.
            # For simplicity, we assume monitoring thread handles it or initial bind was enough if txdevice object existed.
            # A more robust way is to have the on_connected callback try to re-bind for the new device.
            if not bind_tx_callbacks(): # Try to bind again if it failed initially
                logger.info("Attempted to re-bind TX callbacks after 12V on.")

        except Exception as e:
            logger.exception(f"Error during 12V power on for TX test: {e}")


    # Keep this cell running for a bit to observe events
    # In a real application, the main thread would do other work or run its own event loop.
    for _ in range(10): # Wait for 10 seconds (10 * 1s)
        if not (monitor_thread_async and monitor_thread_async.is_alive()):
            logger.warning("Monitoring thread seems to have stopped prematurely.")
            break
        time.sleep(1)
        # logger.debug("Main thread alive tick...") # For verbose check

    logger.info("Finished observation period for async events.")

else:
    logger.warning("Monitoring not active. Cannot perform interactions or observe events.")
# -

# ## 7. Stopping Asynchronous Monitoring
#
# When done, it's important to stop the monitoring loop and clean up resources.

# +
if lifu_interface_async:
    logger.info("--- Stopping Asynchronous Monitoring ---")

    # Signal the interface's internal monitoring loop to stop
    # This should cause the asyncio task `interface.start_monitoring()` to finish.
    logger.info("Calling interface.stop_monitoring()...")
    lifu_interface_async.stop_monitoring()

    # Signal our wrapper thread's controlling event (if it was designed to use one explicitly for exit)
    # In this setup, `stop_monitoring()` should be enough to make `start_monitoring()` return.
    # stop_monitor_event.set() # Not strictly needed if start_monitoring is well-behaved on stop_monitoring

    if monitor_thread_async and monitor_thread_async.is_alive():
        logger.info("Waiting for monitoring thread to join...")
        monitor_thread_async.join(timeout=5.0) # Wait for up to 5 seconds
        if monitor_thread_async.is_alive():
            logger.warning("Monitoring thread did not join cleanly after timeout.")
        else:
            logger.info("Monitoring thread joined successfully.")
    else:
        logger.info("Monitoring thread was not active or already finished.")

    # Clean up the interface object
    # del lifu_interface_async # Python's GC will handle it, but explicit can be clearer
    # lifu_interface_async = None
    logger.info("Asynchronous monitoring stopped.")

else:
    logger.info("LIFUInterface not initialized, no monitoring to stop.")
# -

# ## 8. Conclusion
#
# This notebook demonstrated the basics of setting up asynchronous monitoring with `LIFUInterface`.
# Key aspects:
# *   Defining callback functions for hardware events.
# *   Binding these callbacks to signals from the `txdevice` and `hvcontroller`.
# *   Running the `interface.start_monitoring()` asyncio function in a background thread.
# *   The main thread remains free for other tasks while events are handled by callbacks.
# *   Properly stopping the monitoring and cleaning up.
#
# This asynchronous pattern is fundamental for building responsive applications that need to interact with OpenLIFU hardware continuously or react to hardware-initiated events.
#
# **Next Steps:**
# *   The final planned notebook is `13_NIFTI_Integration_for_Targeting.py` (if `test_nifti.py` is suitable).
# *   Consider how this async pattern could be integrated into a larger application (e.g., a PyQt or Kivy GUI).

# End of Notebook 11
