from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, List

from openlifu.cloud.status import Status
from openlifu.cloud.utils import logger_cloud


class SyncThread:
    DEBOUNCE_SEC = 1.0

    def __init__(self, status_callback: Callable[[Status], None]):
        self._pending_syncs: dict[tuple[Any, Path | None], float] = {}
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread = None
        self._status_callback = status_callback
        self._ignore_paths: List[Path] = []

    def add_path_to_ignore_list(self, path: Path):
        self._ignore_paths.append(path)

    def is_path_in_ignore_list(self, path: Path) -> bool:
        return path in self._ignore_paths

    def post(self, item: Any, path: Path | None):
        with self._lock:
            self._pending_syncs[(item, path)] = time.time()

    def emit_status(self, status: Status):
        if self._status_callback is not None:
            self._status_callback(status)

    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=False, name="SyncThread")
        self._worker_thread.start()

    def stop(self):
        if not self._running:
            return
        self._worker_thread.join(timeout=5)
        self._running = False
        self._worker_thread = None

    def _worker(self):
        from openlifu.cloud.components.abstract_component import AbstractComponent
        self.emit_status(Status(Status.STATUS_IDLE))

        while self._running:
            time.sleep(0.01)
            #logger_cloud.debug('SyncThread: Checking for pending syncs...')
            now = time.time()
            item_to_process = None
            with self._lock:
                ready = [
                    item for item, ts in self._pending_syncs.items()
                    if now - ts > self.DEBOUNCE_SEC
                ]

                if ready:
                    item_to_process = ready[0]
                    del self._pending_syncs[item_to_process]

            if not item_to_process:
                continue

            component, path = item_to_process

            if isinstance(component, AbstractComponent):
                try:
                    component.sync(path)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    traceback.print_exc()
                    logger_cloud.error(f"Error syncing component {component.get_component_type_plural()} for path {path}: {e}")
                    #self._pending_syncs.clear()
                    #self.emit_status(Status(
                    #    Status.STATUS_ERROR, component_type=component.get_component_type_plural(), ex=e))
                    continue

            idle = len(self._pending_syncs) == 0

            if idle and len(ready) > 0:
                self._ignore_paths.clear()
                self.emit_status(Status(Status.STATUS_IDLE))

        logger_cloud.info("SyncThread: Worker thread exiting.")
