import asyncio
import serial
import logging
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("UART")

class AsyncSerial:
    def __init__(self, port, baudrate, timeout=10):
        self.serial_port = serial.Serial(port, baudrate, timeout=timeout)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop = asyncio.get_event_loop()

    async def read(self, size=1):
        return await self.loop.run_in_executor(self.executor, self.serial_port.read, size)

    async def read_all(self):
        return await self.loop.run_in_executor(self.executor, self.serial_port.read_all)

    async def write(self, data):
        return await self.loop.run_in_executor(self.executor, self.serial_port.write, data)

    def close(self):
        self.serial_port.close()
        self.executor.shutdown()
