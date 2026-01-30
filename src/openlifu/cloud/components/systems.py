from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import CreateObjectRequestDto, DatabaseSyncRequestDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.sync_thread import SyncThread


class Systems(AbstractComponent):
    CONNECTED_SYSTEM_FILE = "connected_system.txt"

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Systems, self).__init__(api, db_path, database_id, sync_thread)
        self._cloud_items = []

    def get_component_type(self) -> str:
        return "system"

    def get_component_type_plural(self) -> str:
        return "systems"

    def get_cloud_modification_date(self) -> Optional[datetime]:
        return self.api.databases().get_database(self.db_id).systems_sync_date

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        if not remote_id:
            remote_id = self.api.systems().create(
                CreateObjectRequestDto(database_id=self.db_id, local_id=local_id)
            ).id

        self.api.systems().upload_config(remote_id, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.systems().get_config(remote_id)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.systems().delete(remote_id)

    def create_sync_request(self, sync_date: datetime) -> DatabaseSyncRequestDto:
        return DatabaseSyncRequestDto(systems_sync_date=sync_date)

    def get_cloud_items(self) -> List[Any]:
        self._cloud_items = self.api.systems().get_all(self.db_id)
        return self._cloud_items
