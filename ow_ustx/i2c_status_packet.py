import struct
import crcmod

class I2C_STATUS_Packet:
    def __init__(self):
        self.id = 0
        self.cmd = 0
        self.status = 0
        self.reserved = 0
        self.data_len = 0

    @property
    def crc(self):
        return self.calculate_crc()
    
    def calculate_crc(self):
        crc16 = crcmod.predefined.Crc('crc-ccitt-false')
        buffer = struct.pack('<HBBBB', self.id, self.cmd, self.status, self.reserved, self.data_len)
        crc16.update(buffer)
        return crc16.crcValue
    
    def to_buffer(self):
        return struct.pack('<HBBBBH', self.id, self.cmd, self.status, self.reserved, self.data_len, self.crc)

    def from_buffer(self, buffer):
        packetCrc = 0
        self.id, self.cmd, self.status, self.reserved, self.data_len, packetCrc = struct.unpack('<HBBBBH', buffer)
        if(packetCrc != self.calculate_crc()):
            raise ValueError("CRC validation failed.")
        return self

    def print_packet(self):
        print("Status Packet:")
        print("  ID:", self.id)
        print("  Command:", hex(self.cmd))
        print("  Status:", hex(self.status))
        print("  Reserved:", hex(self.reserved))
        print("  Data Length:", self.data_len)
        print("  CRC:", hex(self.crc))

    @staticmethod
    def main():
        # Create an instance of the packet
        packet = I2C_STATUS_Packet()

        # Set packet data
        packet.id = 1
        packet.cmd = 2
        packet.status = 3
        packet.reserved = 0
        packet.data_len = 4

        # Print packet details
        print("Original Packet:")
        packet.print_packet()

        # Convert packet to buffer
        buffer = packet.to_buffer()

        # Create a new packet instance and parse from buffer
        new_packet = I2C_STATUS_Packet().from_buffer(buffer)

        # Print parsed packet details
        print("\nParsed Packet:")
        new_packet.print_packet()

if __name__ == "__main__":
    I2C_STATUS_Packet.main()
