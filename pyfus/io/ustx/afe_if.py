from .core import *
from .config import *
import struct

from .tx7332_if import TX7332_IF
from .i2c_status_packet import I2C_STATUS_Packet

class AFE_IF:
    _delay = 0.02

    def __init__(self, i2c_addr: int, controller):
        """
        Initialize the AFE_IF class.

                :param i2c_addr: I2C address of the AFE.
        :param ctrl_if: Instance of CTRL_IF for communication.
        """
        self.i2c_addr=i2c_addr
        self.ctrl_if = controller
        self._tx_instances = []
        self._uart = controller.uart
        
    def ping(self, packet_id=None):
        """
        Sends a ping command to the AFE.

        :param packet_id: Packet ID for the command.
        :return: Response from the AFE.
        """
        # Send OW_CMD_PING command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count

        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_PING, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # Handle response
        return response
        
    def pong(self, packet_id=None):
        """
        Sends a pong command to the AFE.

        :param packet_id: Packet ID for the command.
        :return: Response from the AFE.
        """
        # Send OW_CMD_PING command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_PONG, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # Handle response
        return response

    def echo(self, packet_id=None, data=None):
        """
        Send an echo command to the device.

        :param packet_id: Packet ID for the command.
        :param data: Data to send with the command.
        :return: Response from the device.
        """
        # Send USTX_ECHO command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_ECHO, addr=self.i2c_addr, data=data)
        self._uart.clear_buffer()
        # handle response
        return response
    
    def toggle_led(self, packet_id=None):
        """
        Toggle the LED on the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send OW_CMD_TOGGLE_LED command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_TOGGLE_LED, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # handle response
        return response
    
    def version(self, packet_id=None):
        """
        Get the version of the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send OW_CMD_VERSION command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_VERSION, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # handle response
        return response
    
    def chipid(self, packet_id=None):
        """
        Get the version of the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send OW_CMD_HWID command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_HWID, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # handle response
        return response
        
    def tx7332_demo(self, packet_id=None):
        """
        write tx7332 demo registers to device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send OW_CMD_HWID command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_DEMO, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # handle response
        return response
    
    def reset(self, packet_id=None):
        """
        reset the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_VERSION command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_CMD_RESET, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # handle response
        return response
    
    def enum_tx7332_devices(self, packet_id=None):
        """
        Enumerate TX7332 attached devices.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_VERSION command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        
        self._tx_instances.clear()
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_AFE_ENUM_TX7332, addr=self.i2c_addr)
        self._uart.clear_buffer()
        rUartPacket = UartPacket(buffer=response)
        afe_resp = I2C_STATUS_Packet()
        afe_resp.from_buffer(buffer=rUartPacket.data)
        afe_resp.print_packet()
        if(afe_resp.status == 0):
            for i in range(afe_resp.reserved):
                self._tx_instances.append(TX7332_IF(self,i))
        # handle response
        return response
    
    def status(self, packet_id=None):
        """
        Get the status of the device.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_VERSION command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_STATUS, command=OW_CMD_NOP, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # handle response
        print(response)
        rUartPacket = UartPacket(buffer=response)
        rUartPacket.print_packet()
        afe_resp = I2C_STATUS_Packet()
        afe_resp.from_buffer(buffer=rUartPacket.data)
        afe_resp.print_packet()

        return response    
        
    def read_data(self, packet_id=None, packet_len: int = 0):
        """
        Read data packet from AFE.

        :param packet_id: Packet ID for the command.
        :return: Response from the device.
        """
        # Send USTX_VERSION command
        if packet_id is None:
            self.ctrl_if.packet_count += 1
            packet_id = self.ctrl_if.packet_count
        time.sleep(self._delay)
        response = self._uart.send_ustx(id=packet_id, packetType=OW_AFE_READ, command=packet_len, addr=self.i2c_addr)
        self._uart.clear_buffer()
        # handle response
        return response
    

    @property
    def tx_devices(self):
        """
        Get the I2C address of the AFE.
        """
        return self._tx_instances
    
    @property
    def i2c_address(self):
        """
        Get the I2C address of the AFE.
        """
        return self.i2c_addr

    def print(self):
        """
        Print information about the AFE interface instance.
        """
        print("  AFE Instance Information")
        formatted_hex = '0x{:02X}'.format(self.i2c_address)
        formatted_hex = formatted_hex.replace(' ', '')  # Remove space
        print(f"    I2C Address: {formatted_hex}")
        print("    Connected TX7332 Devices:")
        for tx_instance in self._tx_instances:
            tx_instance.print()