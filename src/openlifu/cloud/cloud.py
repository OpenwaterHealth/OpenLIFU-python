import os
import time
from pathlib import Path
from typing import List, Optional, Callable

from openlifu.cloud.const import DEBUG
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.components.protocols import Protocols
from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import ClaimDbDto, DatabaseDto
from openlifu.cloud.filesystem_observer import FilesystemObserver
from openlifu.cloud.components.subjects import Subjects
from openlifu.cloud.components.systems import Systems
from openlifu.cloud.components.transducers import Transducers
from openlifu.cloud.components.users import Users
from openlifu.cloud.status import Status
from openlifu.cloud.sync_thread import SyncThread
from openlifu.cloud.utils import get_mac_address


class Cloud:

    def __init__(self):
        self._filesystem_observer = FilesystemObserver(self._on_path_changed)
        self._api = Api()
        self._components: List[AbstractComponent] = []
        self._sync_thread = SyncThread(self._on_status_changed)
        self._db_path = None
        self._db: Optional[DatabaseDto] = None
        self._status_callback: Optional[Callable[[Status], None]] = None

    def set_access_token(self, token: str):
        self._api.authenticate(token)

    def set_status_callback(self, callback: Callable[[Status], None]):
        self._status_callback = callback

    def start(self, db_path: Path):
        if not db_path.exists():
            raise ValueError("Database path does not exist.")
        if not db_path.is_dir():
            raise ValueError("Database path is not a directory.")

        self._db_path = db_path

        mac_address = get_mac_address()
        if mac_address is None:
            raise ValueError("MAC address is unavailable.")

        if DEBUG:
            print("Using DB path: {}".format(db_path))
            print("Mac address: {}".format(mac_address))

        self._db = self._api.databases().claim_database(
            ClaimDbDto(
                db_path=str(db_path),
                mac_address=mac_address,
                description=None
            )
        )

        self._create_components()
        self._sync_thread.start()

    def stop(self):
        self._sync_thread.stop()
        if self._db is not None:
            self._api.databases().release_database(self._db.id)
        self._api.logout()
        self.stop_background_sync()
        self._db_path = None
        self._db = None

    def sync(self):
        for component in self._components:
            self._sync_thread.post(component)

    def start_background_sync(self):
        if self._db_path is None:
            raise ValueError("Database path does not exist.")
        self._filesystem_observer.start(self._db_path)

    def stop_background_sync(self):
        self._filesystem_observer.stop()

    def _on_path_changed(self, path: Path):
        for component in self._components:
            component.on_filesystem_change(path)

    def _on_status_changed(self, status: Status):
        if self._status_callback is not None:
            self._status_callback(status)

    def _create_components(self):
        self._components.clear()
        self._components.append(Users(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(Protocols(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(Systems(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(Transducers(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(Subjects(self._api, self._db_path, self._db.id, self._sync_thread))


if __name__ == "__main__":
    cloud = Cloud()
    token = os.getenv("TOKEN")
    db_path = os.getenv("DB_PATH")

    cloud.set_access_token(token)
    cloud.set_status_callback(lambda status: print("Status: {}".format(status)))

    cloud.start(Path(db_path))
    cloud.sync()
    cloud.start_background_sync()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cloud.stop()
