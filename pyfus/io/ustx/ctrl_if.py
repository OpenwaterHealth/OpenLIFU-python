from .core import *
from .config import *
from .utils import *
import struct
import json

from .afe_if import AFE_IF

# handle response

class CTRL_IF:
    
    _delay = 0.02

    def __init__(self, uart: UART):
        """
        Initialize the AFE_IF class with an instance of UART.

        :param uart: Instance of the UART class for communication.
        """
        self.uart = uart
        self.packet_count = 0
        self._afe_instances = []

    def ping(self, packet_id=None):
    
        # Send USTX_PING command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count

        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_PING)        
        self.uart.clear_buffer()

        # handle response
        return response

    def pong(self, packet_id=None):
    
        # Send USTX_PING command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_PONG)        
        self.uart.clear_buffer()

        # handle response
        return response

    def echo(self, data = None, packet_id=None):
        """
        Send an echo command to the device.

        :param packet_id: Packet ID for the command.
        :param data: Data to send with the command.
        :return: Response from the device.
        """
        # Send USTX_ECHO command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_ECHO, data=data)
        self.uart.clear_buffer()
        # handle response
        return response
    
    def toggle_led(self, packet_id=None):
        """
        Toggle the LED on the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_TOGGLE_LED command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_TOGGLE_LED)
        self.uart.clear_buffer()
        # handle response
        return response
    
    def version(self, packet_id=None):
        """
        Get the version of the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_VERSION command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_VERSION)
        self.uart.clear_buffer()
        # handle response
        return response
    
    def chipid(self, packet_id=None):
        """
        Get the version of the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_VERSION command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_HWID)
        self.uart.clear_buffer()
        # handle response
        return response
    
    def reset(self, packet_id=None):
        """
        Get the version of the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_VERSION command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CMD_RESET)
        self.uart.clear_buffer()
        # handle response
        return response
    
    def enum_i2c_devices(self, packet_id=None):
        """
        Enumerate the AFEs on the i2c bus.

        :return: A list of AFE addresses.
        """
        # Send USTX_AFE_ENUM command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_SCAN_I2C)
        self.uart.clear_buffer()
        
        # for byte in response:
        #     print(f"{byte:02X}", end=' ')
        # print()

        # handle response
        ret_val = []
        self._afe_instances.clear()

        try:
            retPacket = UartPacket(buffer=response)
            for i in range(retPacket.data_len):
                # Assuming each data element represents an I2C address
                i2c_address = retPacket.data[i]
                afe_instance = AFE_IF(i2c_address, self)
                self._afe_instances.append(afe_instance)
                ret_val.append(i2c_address)
        except Exception as e:
            print("Error decoding packet:", e)
        return ret_val
    
    def set_trigger(self, data = None, packet_id=None):
        # Prepare data payload        
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        if data:
            try:
                json_string = json.dumps(data)
            except json.JSONDecodeError as e:
                # Handle the error if data is not valid JSON
                print(f"Data must be valid JSON: {e}")
                return  

            payload = json_string.encode('utf-8')
        else:
            payload = None # assume a byte buffer

        # Send Set Trigger command
        response = self.uart.send_ustx(id=1, packetType=OW_CONTROLLER, command=OW_CTRL_SET_SWTRIG, data=payload)
        # clear buffer
        self.uart.clear_buffer()
        # handle response
        return response

    def get_trigger(self, packet_id=None):
        # Prepare data payload        
        # Send Get Trigger command
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        response = self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_GET_SWTRIG, data=None)
        # clear buffer
        self.uart.clear_buffer()
        # handle response
        #format_and_print_hex(response)
        data_object = None
        try:
            parsedResp = UartPacket(buffer=response)
            data_object = json.loads(parsedResp.data.decode('utf-8'))
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        return data_object
    
    def start_trigger(self, packet_id=None):
        # Prepare data payload
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_START_SWTRIG, data=None)
        # handle response

        # clear buffer
        self.uart.clear_buffer()

    def stop_trigger(self, packet_id=None):
        # Prepare data payload
        if packet_id is None:
            self.packet_count += 1
            packet_id = self.packet_count
        
        time.sleep(self._delay)
        self.uart.send_ustx(id=packet_id, packetType=OW_CONTROLLER, command=OW_CTRL_STOP_SWTRIG, data=None)
        # handle response

        # clear buffer
        self.uart.clear_buffer()

    @property
    def afe_devices(self):
        """
        Get the I2C address of the AFE.
        """
        return self._afe_instances
    
    def print(self):
        print("Controller Instance Information")
        print("  UART Port:")
        self.uart.print()

        print("  AFE Instances:")        
        for i in range(len(self._afe_instances)):
            afe_device = self._afe_instances[i]
            afe_device.print()
