from __future__ import annotations

from pathlib import Path
from typing import Callable

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from openlifu.cloud.utils import logger_cloud


class FilesystemObserver(FileSystemEventHandler):

    def __init__(self, changed_path_callback: Callable[[Path], None]):
        self._observer: Observer | None = None
        self._changed_path_callback = changed_path_callback
        self._running = False

    def start(self, db_path: Path):
        if self._running:
            return
        self._running = True
        self._observer = Observer()
        self._observer.daemon = False

        self._observer.schedule(self, db_path, recursive=True)
        self._observer.start()

    def stop(self):
        if self._running:
            self._running = False
            self._observer.stop()
            self._observer.unschedule_all()
            self._observer = None

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        """Called when a file or a directory is moved or renamed.

        :param event:
            Event representing file/directory movement.
        :type event:
            :class:`DirMovedEvent` or :class:`FileMovedEvent`
        """
        if event.dest_path not in ('', event.src_path):
            logger_cloud.debug(f"FS_DEBUG: on_moved: {event.dest_path}\n")
            self._changed_path_callback(Path(event.dest_path))
        logger_cloud.debug(f"FS_DEBUG: on_moved: {event.src_path}\n")
        self._changed_path_callback(Path(event.src_path))

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        """Called when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`DirCreatedEvent` or :class:`FileCreatedEvent`
        """
        logger_cloud.debug(f"FS_DEBUG: on_created: {event.src_path}\n")
        self._changed_path_callback(Path(event.src_path))

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        """Called when a file or directory is deleted.

        :param event:
            Event representing file/directory deletion.
        :type event:
            :class:`DirDeletedEvent` or :class:`FileDeletedEvent`
        """
        logger_cloud.debug(f"FS_DEBUG: on_deleted: {event.src_path}\n")
        self._changed_path_callback(Path(event.src_path))

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        """Called when a file or directory is modified.

        :param event:
            Event representing file/directory modification.
        :type event:
            :class:`DirModifiedEvent` or :class:`FileModifiedEvent`
        """
        logger_cloud.debug(f"FS_DEBUG: on_modified: {event.src_path}\n")
        self._changed_path_callback(Path(event.src_path))
