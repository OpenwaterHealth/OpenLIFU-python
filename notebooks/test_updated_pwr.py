from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_updated_if.py
"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""
print("Starting LIFU Test Script...")
interface = LIFUInterface(test_mode=False)
tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')


print("Ping the device")
interface.hvcontroller.ping()

print("Toggle LED")
interface.hvcontroller.toggle_led()

print("Get Version")
version = interface.hvcontroller.get_version()
print(f"Version: {version}")

print("Echo Data")
echo, echo_len = interface.hvcontroller.echo(echo_data=b'Hello LIFU!')
if echo_len > 0:
    print(f"Echo: {echo.decode('utf-8')}")  # Echo: Hello LIFU!
else:
    print("Echo failed.")

print("Get HW ID")
hw_id = interface.hvcontroller.get_hardware_id()
print(f"HWID: {hw_id}")

print("Test 12V...")
if interface.hvcontroller.turn_12v_on():
    print("12V ON Press enter to TURN OFF:")
    input()  # Wait for the user to press Enter
    if interface.hvcontroller.turn_12v_off():
        print("12V OFF.")
    else:
        print("Failed to turn off 12V")
else:
    print("Failed to turn on 12V.")
