from __future__ import annotations

import asyncio
import logging
import sys

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from qasync import QEventLoop

from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import (  # Assuming Pulse is needed to create a Solution
    Pulse,
    Solution,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LIFUTestWidget(QWidget):
    # Signals now accept two arguments: descriptor and port/data.
    signal_connected = pyqtSignal(str, str)
    signal_disconnected = pyqtSignal(str, str)
    signal_data_received = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.interface = LIFUInterface(run_async=True)
        # Maintain connection status for both descriptors
        self.connections = {"TX": False, "HV": False}
        self.treatment_running = False
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Open LIFU")
        self.setGeometry(100, 100, 300, 350)

        # Status label shows connection status for both devices
        self.status_label = QLabel("TX: Disconnected, HV: Disconnected", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Existing buttons
        self.send_ping_button = QPushButton("Send Ping", self)
        self.send_ping_button.setEnabled(False)
        self.send_ping_button.clicked.connect(self.send_ping_command)

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
        layout.addWidget(self.status_label)
        layout.addWidget(self.send_ping_button)
        layout.addWidget(self.treatment_button)
        layout.addWidget(self.load_solution_button)
        layout.addWidget(self.start_sonication_button)
        layout.addWidget(self.stop_sonication_button)
        layout.addWidget(self.get_status_button)
        self.setLayout(layout)

    def connect_signals(self):
        """Connect the signals from the LIFU interface to the UI."""
        # Connect TX signals
        if hasattr(self.interface.txdevice, 'uart'):
            self.interface.txdevice.uart.signal_connect.connect(self.signal_connected.emit)
            self.interface.txdevice.uart.signal_disconnect.connect(self.signal_disconnected.emit)
            self.interface.txdevice.uart.signal_data_received.connect(self.signal_data_received.emit)
        else:
            logger.warning("TX UART interface not found in LIFUInterface.")

        # Connect HV signals
        if hasattr(self.interface.hvcontroller, 'uart'):
            self.interface.hvcontroller.uart.signal_connect.connect(self.signal_connected.emit)
            self.interface.hvcontroller.uart.signal_disconnect.connect(self.signal_disconnected.emit)
            self.interface.hvcontroller.uart.signal_data_received.connect(self.signal_data_received.emit)
        else:
            logger.warning("HV UART interface not found in LIFUInterface.")

        # Connect our widget signals to slots
        self.signal_connected.connect(self.on_connected)
        self.signal_disconnected.connect(self.on_disconnected)
        self.signal_data_received.connect(self.on_data_received)

    async def start_monitoring(self):
        """Start monitoring for USB device connections."""
        await self.interface.start_monitoring()

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
            self.send_ping_button.setEnabled(True)
            self.treatment_button.setEnabled(True)
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
            self.send_ping_button.setEnabled(False)
            self.treatment_button.setEnabled(False)
        self.update()

    @pyqtSlot(str, str)
    def on_data_received(self, descriptor, data):
        """Handle the data received signal."""
        self.status_label.setText(f"{descriptor} Received: {data}")
        self.update()

    def send_ping_command(self):
        """Send a ping command on the TX device."""
        if hasattr(self.interface.txdevice, 'ping'):
            self.interface.txdevice.ping()
        else:
            logger.warning("TX device does not support ping.")

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
        """Draw the connection status indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dot_radius = 20
        dot_color = QColor("green") if any(self.connections.values()) else QColor("red")
        brush = QBrush(dot_color)
        painter.setBrush(brush)
        rect = self.rect()
        painter.drawEllipse(
            rect.center().x() - dot_radius // 2,
            rect.top() + 20,
            dot_radius,
            dot_radius
        )

    def closeEvent(self, event):
        """Handle application closure."""
        self.cleanup_task = asyncio.create_task(self.cleanup_tasks())
        super().closeEvent(event)

    async def cleanup_tasks(self):
        """Stop monitoring and cancel running tasks."""
        self.interface.stop_monitoring()
        loop = asyncio.get_running_loop()
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    widget = LIFUTestWidget()
    widget.show()

    async def main():
        await widget.start_monitoring()

    with loop:
        widget._monitor_task = asyncio.ensure_future(main())
        loop.run_forever()
