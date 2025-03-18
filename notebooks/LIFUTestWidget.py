from __future__ import annotations

import asyncio
import logging
import sys

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qasync import QEventLoop

from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Pulse, Solution

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Define system states
DISCONNECTED = 0
TX_CONNECTED = 1
CONFIGURED = 2
READY = 3
RUNNING = 4


class LIFUConnector(QObject):
    # Ensure signals are correctly defined
    signalConnected = pyqtSignal(str, str)  # (descriptor, port)
    signalDisconnected = pyqtSignal(str, str)  # (descriptor, port)
    signalDataReceived = pyqtSignal(str, str)  # (descriptor, data)
    plotGenerated = pyqtSignal(str)  # Signal to notify QML when a new plot is ready
    solutionConfigured = pyqtSignal(str)  # Signal for solution configuration feedback

    powerStatusReceived = pyqtSignal(bool, bool)  # Signal for power status updates
    rgbStateReceived = pyqtSignal(int, str)  # Emit both integer value and text

    # New Signals for data updates
    hvDeviceInfoReceived = pyqtSignal(str, str)  # (firmwareVersion, deviceId)
    txDeviceInfoReceived = pyqtSignal(str, str)  # (firmwareVersion, deviceId)

    def __init__(self, hv_test_mode=False):
        super().__init__()
        self.interface = LIFUInterface(HV_test_mode=hv_test_mode, run_async=True)
        self._txConnected = False
        self._hvConnected = False
        self._configured = False
        self._state = DISCONNECTED

    def connect_signals(self):
        """Connect LIFUInterface signals to QML."""
        self.interface.signal_connect.connect(self.on_connected)
        self.interface.signal_disconnect.connect(self.on_disconnected)
        self.interface.signal_data_received.connect(self.on_data_received)

    def update_state(self):
        """Update system state based on connection and configuration."""
        if not self._txConnected and not self._hvConnected:
            self._state = DISCONNECTED
        elif self._txConnected and not self._configured:
            self._state = TX_CONNECTED
        elif self._txConnected and self._hvConnected and self._configured:
            self._state = READY
        elif self._txConnected and self._configured:
            self._state = CONFIGURED
        self.stateChanged.emit()  # Notify QML of state update
        logger.info(f"Updated state: {self._state}")


    @pyqtSlot()
    async def start_monitoring(self):
        """Start monitoring for device connection asynchronously."""
        try:
            logger.info("Starting device monitoring...")
            await self.interface.start_monitoring()
        except Exception as e:
            logger.exception(f"Error in start_monitoring: {e}")

    @pyqtSlot()
    def stop_monitoring(self):
        """Stop monitoring device connection."""
        try:
            logger.info("Stopping device monitoring...")
            self.interface.stop_monitoring()
        except Exception as e:
            logger.exception(f"Error while stopping monitoring: {e}")






class LIFUTestWidget(QWidget):
    # Signals now accept two arguments: descriptor and port/data.
    signal_connected = pyqtSignal(str, str)
    signal_disconnected = pyqtSignal(str, str)
    signal_data_received = pyqtSignal(str, str)

    def __init__(self, lifu_instance: LIFUInterface, parent=None):
        super().__init__()
        self.interface = lifu_instance
        # Maintain connection status for both descriptors
        self.connections = {"TX": False, "HV": False}
        self.treatment_running = False
        self.monitoring_task = None  # Track the monitoring task
        self.init_ui()
        self.connect_signals()

        # Connect our widget signals to slots
        self.signal_connected.connect(self.on_connected)
        self.signal_disconnected.connect(self.on_disconnected)
        self.signal_data_received.connect(self.on_data_received)


    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Open LIFU")
        self.setGeometry(100, 100, 300, 450)

        # Status label shows connection status for both devices
        self.status_label = QLabel("TX: Disconnected, HV: Disconnected", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # New checkboxes to enable/disable test mode
        self.tx_test_checkbox = QCheckBox("TX Test Mode", self)
        self.tx_test_checkbox.setChecked(False)
        self.tx_test_checkbox.toggled.connect(self.on_tx_test_mode_toggled)

        self.hv_test_checkbox = QCheckBox("HV Test Mode", self)
        self.hv_test_checkbox.setChecked(False)
        self.hv_test_checkbox.toggled.connect(self.on_hv_test_mode_toggled)

        # Existing buttons
        self.send_tx_ping_button = QPushButton("Send TX Ping", self)
        self.send_tx_ping_button.setEnabled(False)
        self.send_tx_ping_button.clicked.connect(self.send_tx_ping_command)

        self.send_hv_ping_button = QPushButton("Send HV Ping", self)
        self.send_hv_ping_button.setEnabled(False)
        self.send_hv_ping_button.clicked.connect(self.send_hv_ping_command)

        self.treatment_button = QPushButton("Run Treatment (Off)", self)
        self.treatment_button.setEnabled(False)
        self.treatment_button.clicked.connect(self.toggle_treatment_run)

        # New buttons to call interface methods:
        self.load_solution_button = QPushButton("Load Solution", self)
        self.load_solution_button.clicked.connect(self.load_solution)

        self.start_sonication_button = QPushButton("Start Sonication", self)
        self.start_sonication_button.clicked.connect(self.start_sonication)

        self.stop_sonication_button = QPushButton("Stop Sonication", self)
        self.stop_sonication_button.clicked.connect(self.stop_sonication)

        self.get_status_button = QPushButton("Get Status", self)
        self.get_status_button.clicked.connect(self.get_status)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.tx_test_checkbox)
        layout.addWidget(self.hv_test_checkbox)
        layout.addWidget(self.status_label)
        layout.addWidget(self.send_tx_ping_button)
        layout.addWidget(self.send_hv_ping_button)
        layout.addWidget(self.treatment_button)
        layout.addWidget(self.load_solution_button)
        layout.addWidget(self.start_sonication_button)
        layout.addWidget(self.stop_sonication_button)
        layout.addWidget(self.get_status_button)
        self.setLayout(layout)

    def connect_signals(self):
        """Connect the signals from the LIFU interface to the UI."""
        tx_uart = getattr(self.interface.txdevice, 'uart', None)
        hv_uart = getattr(self.interface.hvcontroller, 'uart', None)

        if tx_uart is not None:
            try:
                tx_uart.signal_connect.disconnect()
                tx_uart.signal_disconnect.disconnect()
                tx_uart.signal_data_received.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting TX signals: {e}")

            # Connect TX signals
            tx_uart.signal_connect.connect(self.signal_connected.emit)
            tx_uart.signal_disconnect.connect(self.signal_disconnected.emit)
            tx_uart.signal_data_received.connect(self.signal_data_received.emit)
        else:
            logger.warning("TX UART interface not found in LIFUInterface.")

        if hv_uart is not None:
            # Only connect HV signals if it is not the same as TX
            if hv_uart is not tx_uart:
                try:
                    hv_uart.signal_connect.disconnect()
                    hv_uart.signal_disconnect.disconnect()
                    hv_uart.signal_data_received.disconnect()
                except Exception as e:
                    logger.debug(f"Error disconnecting HV signals: {e}")

                hv_uart.signal_connect.connect(self.signal_connected.emit)
                hv_uart.signal_disconnect.connect(self.signal_disconnected.emit)
                hv_uart.signal_data_received.connect(self.signal_data_received.emit)
            else:
                logger.debug("TX and HV share the same UART instance; connected only once.")
        else:
            logger.warning("HV UART interface not found in LIFUInterface.")

        # Disconnect any existing connections on widget signals to avoid duplicates
        try:
            self.signal_connected.disconnect()
            self.signal_disconnected.disconnect()
            self.signal_data_received.disconnect()
        except Exception as e:
            logger.debug(f"Error disconnecting widget signals: {e}")

    @pyqtSlot(bool)
    def on_tx_test_mode_toggled(self, checked: bool):
        """Handle toggling of TX test mode."""
        # Retrieve current HV test mode from the interface.
        hv_state, _ = self.interface.get_test_mode()  # get_test_mode returns (hv, tx)
        self.interface.set_test_mode(hv_state, checked)


    @pyqtSlot(bool)
    def on_hv_test_mode_toggled(self, checked: bool):
        """Handle toggling of HV test mode."""
        # Retrieve current TX test mode from the interface.
        _, tx_state = self.interface.get_test_mode()
        self.interface.set_test_mode(checked, tx_state)

    @pyqtSlot(str, str)
    def on_connected(self, descriptor, port):
        """Handle the connected signal."""
        self.connections[descriptor] = True
        status_text = (
            f"TX: {'Connected' if self.connections['TX'] else 'Disconnected'}, "
            f"HV: {'Connected' if self.connections['HV'] else 'Disconnected'}"
        )
        self.status_label.setText(status_text)
        # Enable buttons if TX is connected (assuming TX is needed for ping/treatment)
        if self.connections["TX"]:
            self.send_tx_ping_button.setEnabled(True)
            self.treatment_button.setEnabled(True)
        if self.connections["HV"]:
            self.send_hv_ping_button.setEnabled(True)

        self.update()

    @pyqtSlot(str, str)
    def on_disconnected(self, descriptor, port):
        """Handle the disconnected signal."""
        self.connections[descriptor] = False
        status_text = (
            f"TX: {'Connected' if self.connections['TX'] else 'Disconnected'}, "
            f"HV: {'Connected' if self.connections['HV'] else 'Disconnected'}"
        )
        self.status_label.setText(status_text)
        # Disable TX buttons if TX is disconnected
        if not self.connections["TX"]:
            self.send_tx_ping_button.setEnabled(False)
            self.treatment_button.setEnabled(False)
        if not self.connections["HV"]:
            self.send_hv_ping_button.setEnabled(False)

        self.update()

    @pyqtSlot(str, str)
    def on_data_received(self, descriptor, data):
        """Handle the data received signal."""
        self.status_label.setText(f"{descriptor} Received: {data}")
        self.update()

    def send_tx_ping_command(self):
        """Send a ping command on the TX device."""
        if hasattr(self.interface.txdevice, 'ping'):
            self.interface.txdevice.ping()
        else:
            logger.warning("TX device does not support ping.")

    def send_hv_ping_command(self):
        """Send a ping command on the HV device."""
        if hasattr(self.interface.hvcontroller, 'ping'):
            self.interface.hvcontroller.ping()
        else:
            logger.warning("HV device does not support ping.")

    def toggle_treatment_run(self):
        """Toggle the treatment run state."""
        self.interface.toggle_treatment_run(self.treatment_running)
        self.treatment_running = not self.treatment_running
        self.treatment_button.setText(
            "Run Treatment (On)" if self.treatment_running else "Stop Treatment (Off)"
        )

    def load_solution(self):
        """Call the interface's set_solution method using a dummy solution."""
        try:
            # Create a fake solution for testing
            fake_solution = Solution(name="Test Solution", pulse=Pulse(amplitude=5))
            result = self.interface.set_solution(fake_solution)
            if result:
                self.status_label.setText("Solution loaded successfully.")
            else:
                self.status_label.setText("Failed to load solution.")
        except Exception as e:
            self.status_label.setText(f"Error loading solution: {e}")
            logger.error("Error loading solution: %s", e)

    def start_sonication(self):
        """Call the interface's start_sonication method."""
        try:
            result = self.interface.start_sonication()
            if result:
                self.status_label.setText("Sonication started.")
            else:
                self.status_label.setText("Failed to start sonication.")
        except Exception as e:
            self.status_label.setText(f"Error starting sonication: {e}")
            logger.error("Error starting sonication: %s", e)

    def stop_sonication(self):
        """Call the interface's stop_sonication method."""
        try:
            result = self.interface.stop_sonication()
            if result:
                self.status_label.setText("Sonication stopped.")
            else:
                self.status_label.setText("Failed to stop sonication.")
        except Exception as e:
            self.status_label.setText(f"Error stopping sonication: {e}")
            logger.error("Error stopping sonication: %s", e)

    def get_status(self):
        """Call the interface's get_status method and display the status."""
        try:
            status = self.interface.get_status()
            self.status_label.setText(f"Status: {status}")
        except Exception as e:
            self.status_label.setText(f"Error getting status: {e}")
            logger.error("Error getting status: %s", e)

    def paintEvent(self, event):
        """Draw LEDs and labels for connection status."""
        painter = QPainter(self)

        # Draw TX LED
        TX_dot_color = QColor("green") if self.connections['TX'] else QColor("red")
        painter.setBrush(QBrush(TX_dot_color))
        painter.drawEllipse(160, 20, 20, 20)

        # Draw HV LED
        HV_dot_color = QColor("green") if self.connections['HV'] else QColor("red")
        painter.setBrush(QBrush(HV_dot_color))
        painter.drawEllipse(240, 20, 20, 20)

        # Add labels under the LEDs
        font = QFont()
        font.setPointSize(10)  # Set font size
        painter.setFont(font)

        # Draw "TX" label
        painter.drawText(162, 45, 40, 20, 0, "TX")

        # Draw "HV" label
        painter.drawText(242, 45, 40, 20, 0, "HV")

    def closeEvent(self, event):
        """Handle application closure."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        super().closeEvent(event)

def main():
    """Main function to run the LIFU test widget."""
    app = QApplication(sys.argv)
    interface = LIFUInterface( run_async=True)
    widget  = LIFUTestWidget(interface)
    loop = QEventLoop(app)
    widget .show()

    asyncio.set_event_loop(loop)
    async def main_async():
        """Start LIFU monitoring before event loop runs."""
        logger.info("Starting LIFU monitoring...")
        await interface.start_monitoring()

    async def shutdown():
        """Ensure LIFUConnector stops monitoring before closing."""
        logger.info("Shutting down LIFU monitoring...")
        interface.stop_monitoring()

        pending_tasks = [t for t in asyncio.all_tasks() if not t.done()]
        if pending_tasks:
            logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        logger.info("LIFU monitoring stopped. Application shutting down.")

    def handle_exit():
        """Ensure QML cleans up before Python exit without blocking."""
        logger.info("Application closing...")
        ret = asyncio.ensure_future(shutdown())
        # Connect shutdown process to app quit event
        app.aboutToQuit.connect(handle_exit)
        ret.done()

    try:
        with loop:
            loop.run_until_complete(main_async())  # Start monitoring before running event loop
            loop.run_forever()
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
    except KeyboardInterrupt:
        logger.info("Application interrupted.")
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

if __name__ == "__main__":
    main()
