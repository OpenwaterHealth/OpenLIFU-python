from __future__ import annotations

import sys

from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_vmon.py

def main():
    print("Starting LIFU Test Script...")
    interface = LIFUInterface()
    tx_connected, hv_connected = interface.is_device_connected()

    if not hv_connected:
        print("✅ LIFU Console not connected.")
        sys.exit(1)

    print("\nReading voltage monitor values...")
    voltages = interface.hvcontroller.get_vmon_values()

    print("\n" + "="*80)
    print("VOLTAGE MONITOR READINGS")
    print("="*80)

    for channel_data in voltages:
        print(f"\nChannel {channel_data['channel']}:")
        print(f"  Raw ADC:           {channel_data['raw_adc']}")
        print(f"  Voltage:           {channel_data['voltage']:.3f} V")
        print(f"  Converted Voltage: {channel_data['converted_voltage']:.3f} V")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
