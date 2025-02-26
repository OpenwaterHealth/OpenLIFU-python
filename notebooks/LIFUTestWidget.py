import asyncio
import logging
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from qasync import QEventLoop

from openlifu.io.LIFUInterface import LIFUInterface

# Configure logging to print debug messages
logging.basicConfig(
    level=logging.DEBUG,  # Set log level to DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format output with timestamp
)

# Example debug log
logger = logging.getLogger(__name__)
logger.debug("Debug logging is enabled!")  # This will now be printed
running_task = None

class LIFUTestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.interface = LIFUInterface(run_async=True)
        self.treatment_running = False  # State to track treatment status
        self.connected_status = False  # Connection status for the UI indicator

        self.init_ui()
        logger.debug("Connect Signals to UI")
        self.connect_signals()

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Open LIFU")
        self.setGeometry(100, 100, 240, 200)

        # Status label
        self.status_label = QLabel("Status: Disconnected", self)
        self.status_label.setAlignment(Qt.AlignCenter)

        # Ping button
        self.send_ping_button = QPushButton("Send Ping", self)
        self.send_ping_button.setEnabled(False)
        self.send_ping_button.clicked.connect(self.send_ping_command)

        # Treatment button
        self.treatment_button = QPushButton("Run Treatment (Off)", self)
        self.treatment_button.setEnabled(False)
        self.treatment_button.clicked.connect(self.toggle_treatment_run)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.send_ping_button)
        layout.addWidget(self.treatment_button)
        self.setLayout(layout)

    def connect_signals(self):
        """Connect the signals from the LIFU interface to the UI."""
        self.interface.txdevice.uart.signal_connect.connect(self.on_connected)
        self.interface.txdevice.uart.signal_disconnect.connect(self.on_disconnected)
        self.interface.txdevice.uart.signal_data_received.connect(self.on_data_received)

    async def start_monitoring(self):
        """Start monitoring for USB device connections."""
        await self.interface.start_monitoring()

    def on_connected(self, port):
        """Handle the connected signal."""
        self.status_label.setText(f"Status: Connected on {port}")
        self.send_ping_button.setEnabled(True)
        self.treatment_button.setEnabled(True)
        self.connected_status = True
        self.update()

    def on_disconnected(self):
        """Handle the disconnected signal."""
        self.status_label.setText("Status: Disconnected")
        self.send_ping_button.setEnabled(False)
        self.treatment_button.setEnabled(False)
        self.connected_status = False
        self.update()

    def send_ping_command(self):
        """Send a ping command."""
        self.interface.txdevice.ping()

    def toggle_treatment_run(self):
        """Toggle the treatment run state."""
        self.interface.toggle_treatment_run(self.treatment_running)
        self.treatment_running = not self.treatment_running
        self.treatment_button.setText(
            "Run Treatment (On)" if self.treatment_running else "Stop Treatment (Off)"
        )

    def on_data_received(self, data):
        """Handle the data received signal."""
        self.status_label.setText(f"Received: {data}")

    def paintEvent(self, event):
        """Draw the connection status indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw connection status dot
        dot_radius = 20
        dot_color = QColor("green") if self.connected_status else QColor("red")
        brush = QBrush(dot_color)
        painter.setBrush(brush)
        rect = self.rect()
        painter.drawEllipse(
            rect.center().x() - dot_radius // 2,
            rect.top() + 20,  # Adjust y-position as needed
            dot_radius,
            dot_radius
        )

    def closeEvent(self, event):
        """Handle application closure."""
        self.cleanup_tasks()
        super().closeEvent(event)

    def cleanup_tasks(self):
        """Stop monitoring and cancel running tasks."""
        self.interface.stop_monitoring()

        # Cancel all asyncio tasks
        loop = asyncio.get_event_loop()
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()

        # Optionally, gather and await tasks asynchronously
        running_task = asyncio.create_task(self._await_tasks(tasks))
        if running_task:
            logger.debug("task started")

    async def _await_tasks(self, tasks):
        """Await the cancellation of tasks asynchronously."""
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.debug("Cancelled")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    widget = LIFUTestWidget()
    widget.show()

    async def main():
        await widget.start_monitoring()

    try:
        with loop:
            loop.run_until_complete(main())
            loop.run_forever()
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
