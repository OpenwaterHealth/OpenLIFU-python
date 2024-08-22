import struct
import crcmod

class I2C_DATA_Packet:
    def __init__(self):
        self.id = 0
        self.cmd = 0
        self.reserved = 0
        self.data_len = 0
        self.pData = b''
        self.crc = 0

    @property
    def pkt_len(self):
        # Calculate packet length dynamically based on the size of the buffer
        return struct.calcsize('<BHBBB') + self.data_len + 2  # Packet header + data length + CRC size

    def calculate_crc(self):
        crc16 = crcmod.predefined.Crc('crc-ccitt-false')
        buffer = struct.pack('<BHBBB', self.pkt_len, self.id, self.cmd, self.reserved, self.data_len)
        buffer += self.pData
        crc16.update(buffer)
        return crc16.crcValue

    def to_buffer(self):
        self.crc = self.calculate_crc()
        buffer = struct.pack('<BHBBB', self.pkt_len, self.id, self.cmd, self.reserved, self.data_len)
        buffer += self.pData
        buffer += struct.pack('<H', self.crc)
        return buffer

    def calc_crc(self, buffer):
        crc16 = crcmod.predefined.Crc('crc-ccitt-false')
        crc16.update(buffer)
        return crc16.crcValue

    def from_buffer(self, buffer):
        #print("Received Buffer: ", buffer.hex())
        # Extract fields from the buffer
        pktLen = 0
        pktLen, self.id, self.cmd, self.reserved, self.data_len = struct.unpack('<BHBBB', buffer[:6])
        #print("Packet Length: ", hex(pktLen))
        #print("Data Length: ", hex(self.data_len))
        self.pData = buffer[6:6 + self.data_len]
        received_crc = struct.unpack('<H', buffer[-2:])[0]  # Unpack the received CRC value
        #print("Received CRC: ", hex(received_crc))

        # Calculate CRC of received data (excluding start and end bytes)
        calculated_crc = self.calc_crc(buffer[0:-2])

        #print("Calculated CRC: ", hex(calculated_crc))

        #print("Self Packet Length: ", hex(self.pkt_len))
        if pktLen != self.pkt_len:
            raise ValueError("Packet length validation failed.")

        # Validate CRC
        if received_crc != calculated_crc:
            raise ValueError("CRC validation failed.")

        # Set CRC to the received CRC
        self.crc = received_crc

        return self


    def print_packet(self):
        print("I2C Data Packet")
        print("  Packet Length:", hex(self.pkt_len))
        print("  Packet ID:", self.id)
        print("  Command:", hex(self.cmd))
        print("  Reserved:", hex(self.reserved))
        print("  Data Length:", self.data_len)
        print("  Data:", self.pData.hex())
        print("  CRC:", hex(self.crc))


    def print_bytes(self, buffer=None):
        if buffer:
            print("Byte Buffer: ", buffer.hex())
        else:
            print("Byte Buffer: ", self.to_buffer().hex())


    @staticmethod
    def main():
        # Create an instance of the packet
        packet = I2C_DATA_Packet()

        # Set packet data
        packet.id = 1
        packet.cmd = 2
        packet.reserved = 3
        packet.data_len = 4
        packet.pData = b'\x11\x22\x33\x44'

        # Print packet details
        print("Original Packet:")
        packet.print_packet()

        # Convert packet to buffer
        buffer = packet.to_buffer()

        # Print byte buffer
        packet.print_bytes(buffer)

        # Create a new packet instance and parse from buffer
        new_packet = I2C_DATA_Packet().from_buffer(buffer)

        # Print parsed packet details
        print("\nParsed Packet:")
        new_packet.print_packet()

        print("New Packet:")
        new_packet.print_packet()

        # Validate CRC
        print("\nCRC Validation:")
        if packet.crc == new_packet.crc:
            print("CRC Matched!")
        else:
            print("CRC Mismatched!")

if __name__ == "__main__":
    I2C_DATA_Packet.main()
