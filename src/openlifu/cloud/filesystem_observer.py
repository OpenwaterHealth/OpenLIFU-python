from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer


class FilesystemObserver(FileSystemEventHandler):

    def __init__(self, changed_path_callback: Callable[[Path], None]):
        self._observer = Observer()
        self._changed_path_callback = changed_path_callback

    def start(self, db_path: Path):
        self._observer.unschedule_all()
        self._observer.schedule(self, db_path, recursive=True)
        self._observer.start()

    def stop(self):
        self._observer.stop()
        self._observer.join()

    def on_any_event(self, event: FileSystemEvent):
        if event.dest_path != "" and event.dest_path != event.src_path:
            self._changed_path_callback(Path(event.dest_path))
        self._changed_path_callback(Path(event.src_path))
