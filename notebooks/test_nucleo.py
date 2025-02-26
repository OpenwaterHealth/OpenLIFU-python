from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_nucleo.py
"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""
print("Starting LIFU Test Script...")
interface = LIFUInterface(test_mode=False, run_async=False)
tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')

print("Ping the device")
interface.txdevice.ping()

print("Toggle LED")
interface.txdevice.toggle_led()

print("Get Version")
version = interface.txdevice.get_version()
print(f"Version: {version}")

print("Echo Data")
echo, echo_len = interface.txdevice.echo(echo_data=b'Hello LIFU!')
if echo_len > 0:
    print(f"Echo: {echo.decode('utf-8')}")  # Echo: Hello LIFU!
else:
    print("Echo failed.")

print("Get HW ID")
hw_id = interface.txdevice.get_hardware_id()
print(f"HWID: {hw_id}")

print("Get Temperature")
temperature = interface.txdevice.get_temperature()
print(f"Temperature: {temperature} °C")

print("Run Self OneWire Test")
interface.txdevice.run_test()

print("Reset Device:")
# Ask the user for confirmation
user_input = input("Do you want to reset the device? (y/n): ").strip().lower()

if user_input == 'y':
    if interface.txdevice.soft_reset():
        print("Reset Successful.")
elif user_input == 'n':
    print("Reset canceled.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")

print("Update Device:")
# Ask the user for confirmation
user_input = input("Do you want to update the device? (y/n): ").strip().lower()

if user_input == 'y':
    if interface.txdevice.enter_dfu():
        print("Entering DFU Mode.")
elif user_input == 'n':
    print("Update canceled.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")
