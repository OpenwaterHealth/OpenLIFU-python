import asyncio
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
from qasync import QEventLoop

from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.io.ustx import DelayProfile, PulseProfile, TxModule, print_regs
from openlifu.xdc import Transducer


class App(QWidget):

    CTRL_BOARD = True  # change to false and specify PORT_NAME for Nucleo Board
    PORT_NAME = "COM16"

    def __init__(self):
        super().__init__()
        self.interface = None
        self.configured = False  # System configuration status
        self.trigger_on = False  # Trigger status
        self.tasks = []
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

        # Trigger and System Configuration Status Labels
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

        # Reset and Configure buttons
        self.reset_button = QPushButton('Reset', self)
        self.configure_button = QPushButton('Configure', self)
        top_button_layout = QHBoxLayout()
        top_button_layout.addWidget(self.reset_button)
        top_button_layout.addWidget(self.configure_button)
        main_layout.addLayout(top_button_layout)

        # Start and Stop buttons
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
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
        self.configure_button.clicked.connect(lambda: asyncio.create_task(self.set_configuration()))
        self.start_button.clicked.connect(lambda: asyncio.create_task(self.start_trigger()))
        self.stop_button.clicked.connect(lambda: asyncio.create_task(self.stop_trigger()))

        # Disable start and stop buttons initially
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        # Show the window
        self.show()

    async def async_init(self):
        await self.init_ustx()

    async def init_ustx(self):
        self.interface = LIFUInterface(test_mode=False)
        tx_connected, hv_connected = self.interface.is_device_connected()
        # TODO: handle not fully connected
        if tx_connected and hv_connected:
            print("LIFU Device Fully connected.")
        else:
            print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')

        print("USTx Interface initialized")
        try:
            if self.interface.txdevice.ping():
                print("Ping successful")
            self.get_trigger()
        except ValueError as e:
            print(f"{e}")
            sys.exit(0)

        print("Enumerate TX Chips")
        r = self.interface.txdevice.enum_tx7332_devices()
        print("TX Device Count:", len(r))

    def reset_fields(self):
        """Reset all input fields to their default values."""
        self.left_input.setText('0')
        self.front_input.setText('0')
        self.down_input.setText('0')
        self.frequency_input.setText('400e3')
        self.cycles_input.setText('3')
        self.trigger_freq_input.setText('0')

    async def set_configuration(self):
        """Function to set registers."""
        left = int(self.left_input.text())
        front = int(self.front_input.text())
        down = int(self.down_input.text())
        frequency = float(self.frequency_input.text())
        cycles = int(self.cycles_input.text())
        print(f'Setting registers: Left(X)={left}, Front(Y)={front}, Down(Z)={down}, Frequency={frequency}, Cycles={cycles}')
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

        print("Write TX Chips")
        # Write Registers to Device #series of loops for programming tx chips
        for tx, txregs in zip(self.interface.txdevice.tx_devices(), regs):
            print(f"Writing to TX{tx.get_index()}")
            await tx.write_register(0, 1)
            for address, value in txregs.items():
                if isinstance(value, list):
                    print(f"Writing {len(value)}-value block starting at register 0x{address:X}")
                    await tx.write_block(address, value)
                else:
                    print(f"Writing value 0x{value:X} to register 0x{address:X}")
                    await tx.write_register(address, value)
                time.sleep(0.1)

        self.configured = True
        self.config_status_label.setText('System: Configured')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        print("Configuration Complete! Device is READY!")

    def set_trigger_frequency(self):
        new_frequency = int(self.trigger_freq_input.text())
        print(f'Setting trigger frequency: {new_frequency}')
        self.tasks.append(asyncio.create_task(self.set_trigger(new_frequency)))

    async def get_trigger(self):
        """Function to get the current trigger status."""
        if self.interface is not None:
            trigger_config = self.interface.txdevice.get_trigger()
            if trigger_config is not None:
                self.trigger_freq_input.setText(str(trigger_config.get('TriggerFrequencyHz', 0)))
            return trigger_config
        return False

    async def set_trigger(self, new_frequency):
        """Function to set the trigger status."""
        if self.interface is not None:
            trigger_config = {
                "TriggerFrequencyHz": 10,
                "TriggerMode": 1,
                "TriggerPulseCount": 0,
                "TriggerPulseWidthUsec":  5000
            }
            trigger_config["TriggerFrequencyHz"] = new_frequency
            r = self.interface.txdevice.set_trigger(data=trigger_config)

    async def start_trigger(self):
        """Function to handle start trigger."""
        print("Turn Trigger On")
        self.interface.txdevice.start_trigger()
        self.trigger_on = True
        self.update_trigger_label()

    async def stop_trigger(self):
        """Function to handle stop trigger."""
        print("Turn Trigger Off")
        self.interface.txdevice.stop_trigger()
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
        self.tasks.append(asyncio.create_task(self.shutdown_ustx()))
        event.accept()

    async def shutdown_ustx(self):
        """Function to shutdown USTX."""
        print('Shutting down USTX...')
        if self.trigger_on:
            self.stop_trigger()
        if self.interface is not None:
            self.interface.close()


# Function to launch the application
def launch_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    ex = App()
    loop.create_task(ex.async_init())
    with loop:
        loop.run_forever()

# Launch the application
if __name__ == "__main__":
    launch_app()
