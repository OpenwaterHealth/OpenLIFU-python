# Import the necessary modules
import sys
from ow_ustx import *
import json
import time
from pyfus.io.ustx import PulseProfile, DelayProfile, TxModule, TxArray, Tx7332Registers, print_regs
from pyfus.xdc import Transducer, Element
import numpy as np
import json
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox, QFormLayout
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.ustx_ctrl = None
        self.configured = False  # System configuration status
        self.trigger_on = False  # Trigger status
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Open-LIFU Test App')
        self.setGeometry(100, 100, 640, 480)  # Set window size to 640x480

        # Main layout
        main_layout = QVBoxLayout()

        # Beam Focus Section
        beam_focus_group = QGroupBox('Beam Focus')
        beam_focus_layout = QFormLayout()

        self.left_input = QLineEdit(self)
        self.front_input = QLineEdit(self)
        self.down_input = QLineEdit(self)

        # Set default values for Beam Focus
        self.left_input.setText('0')
        self.front_input.setText('0')
        self.down_input.setText('0')

        beam_focus_layout.addRow('Left:', self.left_input)
        beam_focus_layout.addRow('Front:', self.front_input)
        beam_focus_layout.addRow('Down:', self.down_input)

        beam_focus_group.setLayout(beam_focus_layout)
        main_layout.addWidget(beam_focus_group)

        # Pulse Profile Section
        pulse_profile_group = QGroupBox('Pulse Profile')
        pulse_profile_layout = QFormLayout()

        self.frequency_input = QLineEdit(self)
        self.cycles_input = QLineEdit(self)

        # Set default values for Pulse Profile
        self.frequency_input.setText('400e3')
        self.cycles_input.setText('3')

        pulse_profile_layout.addRow('Frequency:', self.frequency_input)
        pulse_profile_layout.addRow('Cycles:', self.cycles_input)

        pulse_profile_group.setLayout(pulse_profile_layout)
        main_layout.addWidget(pulse_profile_group)

        # Trigger and System Configuration Status Labels
        status_layout = QHBoxLayout()  # Layout for Trigger and System status
        self.trigger_status_label = QLabel('Trigger:', self)
        self.trigger_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.trigger_label = QLabel('OFF', self)
        self.trigger_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.trigger_label.setFixedSize(25, 25)  # Set a fixed size for the label
        self.trigger_label.setStyleSheet('background-color: gray; border-radius: 25px; color: white;')

        status_layout.addWidget(self.trigger_status_label)
        status_layout.addWidget(self.trigger_label)

        # Spacer to push System status to the right
        status_layout.addStretch()
        
        # System Configuration label
        self.config_status_label = QLabel('System: Not Configured', self)
        self.config_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        status_layout.addWidget(self.config_status_label)

        main_layout.addLayout(status_layout)

        # Reset and Configure buttons
        self.reset_button = QPushButton('Reset', self)
        self.configure_button = QPushButton('Configure', self)

        # Button layout for Reset and Configure
        top_button_layout = QHBoxLayout()
        top_button_layout.addWidget(self.reset_button)
        top_button_layout.addWidget(self.configure_button)

        main_layout.addLayout(top_button_layout)

        # Start and Stop buttons
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)

        # Button layout for Start and Stop
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.addWidget(self.start_button)
        bottom_button_layout.addWidget(self.stop_button)

        main_layout.addLayout(bottom_button_layout)

        # Set the layout to the window
        self.setLayout(main_layout)

        # Add some styling
        self.setStyleSheet("""
            QWidget {
                font-size: 16px;
            }
            QGroupBox {
                font-weight: bold;
                margin-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 10px;
            }
            QLabel {
                min-width: 80px;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
            }
        """)

        # Connect buttons to their functions
        self.reset_button.clicked.connect(self.reset_fields)
        self.configure_button.clicked.connect(self.set_configuration)
        self.start_button.clicked.connect(self.start_trigger)
        self.stop_button.clicked.connect(self.stop_trigger)

        # Disable start and stop buttons initially
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # initialize USTx controller
        self.init_ustx()

        # Show the window
        self.show()

    def find_usb_comm(self):
        vid = 1155  # Example VID for demonstration
        pid = 22446  # Example PID for demonstration
        
        com_port = list_vcp_with_vid_pid(vid, pid)
        serial_obj = None
        if com_port is None:
            print("No device found")
        else:
            print("Device found at port: ", com_port)
            # Select communication port
            serial_obj = UART(com_port, timeout=5)

        return serial_obj

    def init_ustx(self):
        comm_port = self.find_usb_comm()
        if comm_port is None:
            print("No device found")
        else:
            # Initialize the USTx controller object
            self.ustx_ctrl = CTRL_IF(comm_port)
            if self.ustx_ctrl is not None:
                print("USTx controller initialized")
                # enumerate devices
                print("Enumerate I2C Devices")
                self.ustx_ctrl.enum_i2c_devices()
                print("Enumerate TX7332 Chips on AFE devices")
                for afe_device in self.ustx_ctrl.afe_devices:
                    print("Enumerate TX7332 Chips on AFE device")
                    rUartPacket = afe_device.enum_tx7332_devices()
            else:
                print("Failed to initialize USTx controller")

    def reset_fields(self):
        """Reset all input fields to their default values."""
        self.left_input.setText('0')
        self.front_input.setText('0')
        self.down_input.setText('0')
        self.frequency_input.setText('400e3')
        self.cycles_input.setText('3')

    def set_configuration(self):
        """Function to set registers. Implement the actual logic here."""
        # Example: print values to console (replace with actual register-setting logic)
        left = int(self.left_input.text())
        front = int(self.front_input.text())
        down = int(self.down_input.text())
        frequency = float(self.frequency_input.text())
        cycles = int(self.cycles_input.text())
        print(f'Setting registers: Left={left}, Front={front}, Down={down}, Frequency={frequency}, Cycles={cycles}')
        
        # Set the focus and pulse profile
        focus = np.array([left, front, down]) #set focus #left, front, down 
        pulse_profile = PulseProfile(profile=1, frequency=frequency, cycles=cycles)
        afe_dict = {afe.i2c_addr:afe for afe in self.ustx_ctrl.afe_devices}
        arr = Transducer.from_file(R"pinmap.json")
        arr.elements = np.array(arr.elements)[np.argsort([el.pin for el in arr.elements])].tolist()
        distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm"))**2, 1))
        tof = distances*1e-3 / 1500
        delays = tof.max() - tof 
        txa = TxArray(i2c_addresses=list(afe_dict.keys()))
        array_delay_profile = DelayProfile(1, delays.tolist())
        txa.add_delay_profile(array_delay_profile)
        txa.add_pulse_profile(pulse_profile)
        regs = txa.get_registers(profiles="configured", pack=True) 
        for addr, rm in regs.items():
            print(f'I2C: 0x{addr:02x}')
            for i, r in enumerate(rm):
                print(f'MODULE {i}')
                print_regs(r)
            print('')  #calculate register state for 7332s, settings for board (bits, purpose), #change focus!!
        print("Writing Registers to Device")
        # Write Registers to Device #series of loops for programming tx chips 
        for i2c_addr, module_regs in regs.items():
            afe = afe_dict[i2c_addr] 
            for i in range(len(module_regs)):
                device = afe.tx_devices[i]
                r = module_regs[i]
                device.write_register(0,1) #resetting the device
                for address, value in r.items():
                    if isinstance(value, list):
                        print(f"0x{i2c_addr:x}[{i}] Writing {len(value)}-value block starting at register 0x{address:X}")
                        device.write_block(address, value)
                    else:
                        print(f"0x{i2c_addr:x}[{i}] Writing value 0x{value:X} to register 0x{address:X}")
                        device.write_register(address, value)
                    time.sleep(0.1)
        self.configured = True
        self.config_status_label.setText('System: Configured')
        # Enable start and stop buttons when configured
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def start_trigger(self):
        """Function to handle start trigger."""
        # Implement the actual start trigger logic here
        print("Turn Trigger On")
        self.ustx_ctrl.start_trigger()
        self.trigger_on = True
        self.update_trigger_label()

    def stop_trigger(self):
        """Function to handle stop trigger."""
        # Implement the actual stop trigger logic here
        print("Turn Trigger Off")
        self.ustx_ctrl.stop_trigger()
        self.trigger_on = False
        self.update_trigger_label()

    def update_trigger_label(self):
        """Function to update trigger label color."""
        if self.trigger_on:
            self.trigger_label.setText('ON')
            self.trigger_label.setStyleSheet('background-color: green; border-radius: 25px; color: white;')
        else:
            self.trigger_label.setText('OFF')
            self.trigger_label.setStyleSheet('background-color: gray; border-radius: 25px; color: white;')

    def closeEvent(self, event):
        """Function to handle close event."""
        self.shutdown_ustx()
        event.accept()

    def shutdown_ustx(self):
        """Function to shutdown USTX. Implement the actual logic here."""
        print('Shutting down USTX...')
        if self.trigger_on:
            self.stop_trigger()
        if self.ustx_ctrl is not None:
            self.ustx_ctrl.uart.close()

# Function to launch the application
def launch_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    ex = App()
    app.exec_()

# Launch the application
launch_app()
