import sys
import time

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.io.ustx import DelayProfile, PulseProfile, TxModule, print_regs
from openlifu.xdc import Transducer


class App(QWidget):
    CTRL_BOARD = True  # Change to False and specify PORT_NAME for Nucleo Board
    PORT_NAME = "COM16"

    def __init__(self):
        super().__init__()
        self.interface = None
        self.configured = False  # System configuration status
        self.trigger_on = False  # Trigger status
        self.initUI()
        self.init_ustx()

    def initUI(self):
        self.setWindowTitle('Open-LIFU Test App')
        self.setGeometry(100, 100, 640, 480)

        # Main layout
        main_layout = QVBoxLayout()

        # Beam Focus Section
        beam_focus_group = QGroupBox('Beam Focus')
        beam_focus_layout = QFormLayout()
        self.left_input = QLineEdit(self)
        self.front_input = QLineEdit(self)
        self.down_input = QLineEdit(self)
        self.left_input.setText('0')
        self.front_input.setText('0')
        self.down_input.setText('0')
        beam_focus_layout.addRow('Left(X):', self.left_input)
        beam_focus_layout.addRow('Front(Y):', self.front_input)
        beam_focus_layout.addRow('Down(Z):', self.down_input)
        beam_focus_group.setLayout(beam_focus_layout)
        main_layout.addWidget(beam_focus_group)

        # Pulse Profile Section
        pulse_profile_group = QGroupBox('Pulse Profile')
        pulse_profile_layout = QFormLayout()
        self.frequency_input = QLineEdit(self)
        self.cycles_input = QLineEdit(self)
        self.frequency_input.setText('400e3')
        self.cycles_input.setText('3')
        pulse_profile_layout.addRow('Frequency:', self.frequency_input)
        pulse_profile_layout.addRow('Cycles:', self.cycles_input)
        pulse_profile_group.setLayout(pulse_profile_layout)
        main_layout.addWidget(pulse_profile_group)

        # Trigger Configuration Section
        trigger_config_group = QGroupBox('Trigger Configuration')
        trigger_config_layout = QHBoxLayout()
        self.trigger_freq_input = QLineEdit(self)
        self.trigger_freq_input.setText('50')
        self.set_trigger_button = QPushButton('Set Trigger Frequency', self)
        self.set_trigger_button.clicked.connect(self.set_trigger_frequency)
        trigger_config_layout.addWidget(QLabel('Frequency (Hz):'))
        trigger_config_layout.addWidget(self.trigger_freq_input)
        trigger_config_layout.addWidget(self.set_trigger_button)
        trigger_config_group.setLayout(trigger_config_layout)
        main_layout.addWidget(trigger_config_group)

        # Status and Control Buttons
        status_layout = QHBoxLayout()
        self.trigger_status_label = QLabel('Trigger:', self)
        self.trigger_label = QLabel('OFF', self)
        self.trigger_label.setFixedSize(25, 25)
        self.trigger_label.setStyleSheet('background-color: gray; border-radius: 25px; color: white;')
        status_layout.addWidget(self.trigger_status_label)
        status_layout.addWidget(self.trigger_label)
        status_layout.addStretch()
        self.config_status_label = QLabel('System: Not Configured', self)
        status_layout.addWidget(self.config_status_label)
        main_layout.addLayout(status_layout)

        # Buttons
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_configuration)
        self.configure_button = QPushButton('Configure', self)
        self.configure_button.clicked.connect(self.set_configuration)
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_trigger)
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_trigger)

        # Add buttons to layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.configure_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        # Disable start and stop buttons initially
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        # Set layout
        self.setLayout(main_layout)
        self.reset_fields()

    def init_ustx(self):
        self.interface = LIFUInterface(test_mode=False)
        tx_connected, hv_connected = self.interface.is_device_connected()
        if tx_connected and hv_connected:
            print("LIFU Device Fully connected.")
        else:
            print(f"LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}")

        if tx_connected:
            self.interface.txdevice.ping()
            self.interface.txdevice.enum_tx7332_devices()

    def reset_fields(self):
        self.left_input.setText('0')
        self.front_input.setText('0')
        self.down_input.setText('0')
        self.frequency_input.setText('400e3')
        self.cycles_input.setText('3')
        self.trigger_freq_input.setText('50')

    def reset_configuration(self):
        self.configured = False
        self.config_status_label.setText('System: Not Configured')
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.reset_fields()

    def set_configuration(self):
        left = int(self.left_input.text())
        front = int(self.front_input.text())
        down = int(self.down_input.text())
        frequency = float(self.frequency_input.text())
        cycles = int(self.cycles_input.text())
        print(f'Setting configuration: Left={left}, Front={front}, Down={down}, Frequency={frequency}, Cycles={cycles}')

        focus = np.array([left, front, down])
        pulse_profile = PulseProfile(profile=1, frequency=frequency, cycles=cycles)

        arr = Transducer.from_file(R"notebooks\pinmap.json")
        arr.elements = np.array(arr.elements)[np.argsort([el.pin for el in arr.elements])].tolist()
        distances = np.sqrt(np.sum((focus - arr.get_positions(units="mm")) ** 2, 1))
        tof = distances * 1e-3 / 1500
        delays = tof.max() - tof

        txm = TxModule()
        array_delay_profile = DelayProfile(1, delays.tolist())
        txm.add_delay_profile(array_delay_profile)
        txm.add_pulse_profile(pulse_profile)
        regs = txm.get_registers(profiles="configured", pack=True)
        for i, r in enumerate(regs):
            print(f'MODULE {i}')
            print_regs(r)
        print('')  #calculate register state for 7332s, settings for board (bits, purpose), #change focus!!

        print(f"Transmitter Count {len(self.interface.txdevice.tx_devices)}")
        for i, reg in enumerate(regs):
            print(f"Transmitter {i} has {len(reg)} registers.")


        print("Write TX Chips")
        for tx, txregs in zip(self.interface.txdevice.tx_devices, regs):
            print(f"Writing to TX{tx.get_index()}")
            for address, value in txregs.items():
                if isinstance(value, list):
                    print(f"Writing {len(value)}-value block starting at register 0x{address:X}")
                    self.interface.txdevice.write_block(identifier=tx.identifier, start_address=address, reg_values=value)
                else:
                    print(f"Writing value 0x{value:X} to register 0x{address:X}")
                    self.interface.txdevice.write_register(identifier=tx.identifier, address=address, value=value)
                time.sleep(0.1)

        self.configured = True
        self.config_status_label.setText('System: Configured')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        print("Configuration Complete! Device is READY!")

    def set_trigger_frequency(self):
        new_frequency = int(self.trigger_freq_input.text())
        print(f'Setting trigger frequency to {new_frequency} Hz')
        trigger_config = {
            "TriggerFrequencyHz": new_frequency,
            "TriggerMode": 1,
            "TriggerPulseCount": 0,
            "TriggerPulseWidthUsec": 5000,
        }
        self.interface.txdevice.set_trigger(trigger_config)

    def start_trigger(self):
        self.interface.txdevice.start_trigger()
        self.trigger_on = True
        self.update_trigger_label()

    def stop_trigger(self):
        self.interface.txdevice.stop_trigger()
        self.trigger_on = False
        self.update_trigger_label()

    def update_trigger_label(self):
        if self.trigger_on:
            self.trigger_label.setText('ON')
            self.trigger_label.setStyleSheet('background-color: green; border-radius: 25px; color: white;')
        else:
            self.trigger_label.setText('OFF')
            self.trigger_label.setStyleSheet('background-color: gray; border-radius: 25px; color: white;')


# Launch the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
