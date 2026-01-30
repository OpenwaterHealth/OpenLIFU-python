import threading
import time
import traceback
from typing import Any, Callable

from openlifu.cloud.const import DEBUG
from openlifu.cloud.status import Status


class SyncThread:
    DEBOUNCE_SEC = 1.0

    def __init__(self, status_callback: Callable[[Status], None]):
        self._pending_syncs: dict[Any, float] = {}
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread = None
        self._status_callback = status_callback

    def post(self, item: Any):
        self._pending_syncs[item] = time.time()

    def emit_status(self, status: Status):
        if self._status_callback is not None:
            self._status_callback(status)

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join()
            self._worker_thread = None

    def _worker(self):
        from openlifu.cloud.components.abstract_component import AbstractComponent
        self.emit_status(Status(Status.STATUS_IDLE))

        while True:
            time.sleep(0.1)
            now = time.time()
            with self._lock:
                ready = [
                    path for path, ts in self._pending_syncs.items()
                    if now - ts > self.DEBOUNCE_SEC
                ]
                if len(ready) == 0:
                    if not self._running and len(self._pending_syncs) == 0:
                        return
                    continue

                item = ready[0]

                self.emit_status(Status(Status.STATUS_SYNCHRONIZING, component_type=item.get_component_type_plural()))

                del self._pending_syncs[item]
                if isinstance(item, AbstractComponent):
                    try:
                        item.sync()
                    except Exception as e:
                        if DEBUG:
                            traceback.print_exc()

                        self._pending_syncs.clear()
                        self.emit_status(Status(Status.STATUS_ERROR, component_type=item.get_component_type_plural(), ex=e))
                        continue

                idle = len(self._pending_syncs) == 0

                if idle and len(ready) > 0:
                    self.emit_status(Status(Status.STATUS_IDLE))

                if idle and not self._running:
                    return
