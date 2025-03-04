from __future__ import annotations

from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown


def gpu_available() -> bool:
    """Check the system for an nvidia gpu and return whether one is available."""
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        nvmlShutdown()
        return device_count > 0
    except Exception: # exception could occur if there is a driver issue, for example
        return False
