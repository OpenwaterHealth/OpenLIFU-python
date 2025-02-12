# Configuration

SERIAL_PORT = 'COM24'  # Change this to your serial port
BAUD_RATE = 921600

# Packet structure constants
OW_START_BYTE = 0xAA
OW_END_BYTE = 0xDD
ID_COUNTER = 0  # Initializing the ID counter

# Packet Types
OW_ACK = 0xE0
OW_NAK = 0xE1
OW_CMD = 0xE2
OW_RESP = 0xE3
OW_DATA = 0xE4
OW_ONE_WIRE = 0xE5
OW_TX7332 = 0xE6
OW_AFE_READ = 0xE7
OW_AFE_SEND = 0xE8
OW_I2C_PASSTHRU = 0xE9
OW_CONTROLLER = 0xEA
OW_POWER = 0xEB
OW_BAD_PARSE = 0xEC
OW_BAD_CRC = 0xED
OW_UNKNOWN = 0xEE
OW_ERROR = 0xEF

OW_CODE_SUCCESS = 0x00
OW_CODE_IDENT_ERROR = 0xFD
OW_CODE_DATA_ERROR = 0xFE
OW_CODE_ERROR = 0xFF

# Global Commands
OW_CMD_PING = 0x00
OW_CMD_PONG = 0x01
OW_CMD_VERSION = 0x02
OW_CMD_ECHO = 0x03
OW_CMD_TOGGLE_LED = 0x04
OW_CMD_HWID = 0x05
OW_CMD_GET_TEMP = 0x06
OW_CMD_NOP = 0x0E
OW_CMD_RESET = 0x0F

# Controller Commands
OW_CTRL_SET_SWTRIG = 0x13
OW_CTRL_GET_SWTRIG = 0x14
OW_CTRL_START_SWTRIG = 0x15
OW_CTRL_STOP_SWTRIG = 0x16
OW_CTRL_STATUS_SWTRIG = 0x17
OW_CTRL_RESET = 0x1F

# TX7332 Commands
OW_TX7332_STATUS = 0x20
OW_TX7332_ENUM = 0x21
OW_TX7332_WREG = 0x22
OW_TX7332_RREG = 0x23
OW_TX7332_WBLOCK = 0x24
OW_TX7332_VWREG = 0x25
OW_TX7332_VWBLOCK = 0x26
OW_TX7332_DEMO = 0x2D
OW_TX7332_RESET = 0x2F

# Power Commands
OW_POWER_STATUS = 0x30
OW_POWER_SET_HV = 0x31
OW_POWER_GET_HV = 0x32
OW_POWER_HV_ON = 0x33
OW_POWER_HV_OFF = 0x34
OW_POWER_12V_ON = 0x35
OW_POWER_12V_OFF = 0x36
