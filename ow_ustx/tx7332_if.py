from .core import *
from .config import *
import struct
from typing import List

from .i2c_status_packet import I2C_STATUS_Packet
from .i2c_data_packet import I2C_DATA_Packet

class TX7332_IF:
    def __init__(self, afe_interface, identifier: int = -1):
        """
        Initialize the Tx7332 class with an instance of UART.

        :param index: Index ID of TX7332 IC.
        """
        self.afe_interface = afe_interface
        self.uart = afe_interface._uart 
        self.identifier = identifier
    
    def get_index(self) -> int:
        """
        Get the index ID of the TX7332 IC.

        :return: Index ID of the TX7332 IC.
        """
        return self.identifier
    
    def write_register(self, address: int, value: int, packet_id=None):
        """
        Write a 32-bit value to a 16-bit address.

        :param address: The 16-bit address where the value will be written.
        :param value: The 32-bit value to write.
        :param index: The tx chip index (0-3) to target. Default is 0.        
        """
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        # Prepare data payload
        data = struct.pack('<HI', address, value)

        # Send USTX_WRITE7332 command with data
        if packet_id is None:
            self.afe_interface.ctrl_if.packet_count += 1
            packet_id = self.afe_interface.ctrl_if.packet_count
        
        response = self.uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_WREG, addr=self.afe_interface.i2c_address, reserved=self.identifier, data = data)
        
        # handle response check for error raise exception on error

        # clear buffer
        self.uart.clear_buffer()
        return response
    
    def read_register(self, address: int, packet_id=None):
        """
        Read a 32-bit value from a 16-bit address.

        :param address: The 16-bit address to read from.
        :param index: The tx chip index (0-3) to target. Default is 0.      
        :return: The 32-bit value read from the address.  
        """
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        if packet_id is None:
            self.afe_interface.ctrl_if.packet_count += 1
            packet_id = self.afe_interface.ctrl_if.packet_count
        
        # Prepare data payload
        data = struct.pack('<H', address)

        # Send USTX_READ7332 command with data
        response = self.uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_RREG, addr=self.afe_interface.i2c_address, reserved=self.identifier, data = data)
        # clear buffer
        self.uart.clear_buffer()

        # handle response (check if none)
        rUartPacket = UartPacket(buffer=response)
        afe_resp = I2C_STATUS_Packet()
        afe_resp.from_buffer(buffer=rUartPacket.data)
        #afe_resp.print_packet()
        if(afe_resp.status == 0):
            response = self.afe_interface.read_data(packet_len=afe_resp.data_len)

        ret_val = 0
        try:
            # Print the response in hex format
            # response_hex = ' '.join(f'{byte:02X}' for byte in response)
            # print(f"Response in hexadecimal: {response_hex}")
            retPacket = UartPacket(buffer=response)
            #retPacket.print_packet()
            data_packet = I2C_DATA_Packet()
            data_packet.from_buffer(buffer=retPacket.data)
            #data_packet.print_packet()
            if data_packet.data_len == 4:
                ret_val = struct.unpack('<I', data_packet.pData)[0]

        except Exception as e:
            print("Error reading response:", e)
            
        return ret_val

    def write_block(self, start_address: int, reg_values: List[int], packet_id=None):
        if self.identifier < 0:
            raise ValueError("TX Chip address NOT SET")
        if self.identifier > 1:
            raise ValueError("TX Chip address must be in the range 0-1")

        # Define the maximum number of register values per block
        max_regs_per_block = 62

        # Calculate the number of chunks needed
        num_chunks = (len(reg_values) + max_regs_per_block - 1) // max_regs_per_block
        responses = []

        for i in range(num_chunks):
            chunk_start = i * max_regs_per_block
            chunk_end = min((i + 1) * max_regs_per_block, len(reg_values))
            chunk = reg_values[chunk_start:chunk_end]

            if packet_id is None:
                self.afe_interface.ctrl_if.packet_count += 1
                packet_id = self.afe_interface.ctrl_if.packet_count

            # Pack the start address for the current chunk and the number of register values in this chunk
            data_format = '<HBB' + 'I' * len(chunk)
            data = struct.pack(data_format, start_address + chunk_start, len(chunk), 0, *chunk)

            # Send the data packet
            response = self.uart.send_ustx(id=packet_id, packetType=OW_AFE_SEND, command=OW_TX7332_WBLOCK, addr=self.afe_interface.i2c_address, reserved=self.identifier, data=data)
    
            # Clear the UART buffer after sending
            self.uart.clear_buffer()

            # Store the response for this chunk
            responses.append(response)

            self.afe_interface.ctrl_if.packet_count += 1
            packet_id = self.afe_interface.ctrl_if.packet_count
        
        return response    

    def print(self):
        print("Controller Instance Information")
        print(f"  Transmitter: {self.identifier}")