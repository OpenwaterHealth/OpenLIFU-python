from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import select


from openlifu.bf import apod_methods, focal_patterns
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan import Protocol
from openlifu.sim import SimSetup


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers and cluttered terminal output
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

log_interval = 1  # seconds; you can adjust this variable as needed

# set focus
xInput = 0
yInput = 0
zInput = 50

frequency_kHz = 400 # Frequency in kHz
duration_msec = 0.1 # Pulse Duration in milliseconds
interval_msec = 20 # Pulse Repetition Interval in milliseconds
num_modules = 2 # Number of modules in the system

use_external_power_supply = False # Select whether to use console or power supply

console_shutoff_temp_C = 70.0 # Console shutoff temperature in Celsius
tx_shutoff_temp_C = 70.0 # TX device shutoff temperature in Celsius
ambient_shutoff_temp_C = 70.0 # Ambient shutoff temperature in Celsius

#TODO: script_timeout_minutes = 30 # Prevent unintentionally leaving unit on for too long
#TODO: log_temp_to_csv_file = True # Log readings to only terminal or both terminal and CSV file

# Fail-safe parameters if the temperature jumps too fast
rapid_temp_shutoff_C = 40 # Cutoff temperature in Celsius if it jumps too fast
rapid_temp_shutoff_seconds = 5 # Time in seconds to reach rapid temperature shutoff
rapid_temp_increase_per_second_shutoff_C = 3 # Rapid temperature climbing shutoff in Celsius

here = Path(__file__).parent.resolve()
db_path = here / ".." / "notebooks"
db = Database(db_path)
arr = db.load_transducer(f"OpenLIFU_{num_modules}x_1")

arr.sort_by_pin()