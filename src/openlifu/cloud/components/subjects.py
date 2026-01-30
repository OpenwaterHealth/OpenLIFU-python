from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import CreateObjectRequestDto, DatabaseSyncRequestDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.sync_thread import SyncThread


class Subjects(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Subjects, self).__init__(api, db_path, database_id, sync_thread)

    def get_component_type(self) -> str:
        return "subject"

    def get_component_type_plural(self) -> str:
        return "subjects"

    def get_cloud_modification_date(self) -> Optional[datetime]:
        return self.api.databases().get_database(self.db_id).subjects_sync_date

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        if not remote_id:
            remote_id = self.api.subjects().create(
                CreateObjectRequestDto(database_id=self.db_id, local_id=local_id)
            ).id

        self.api.subjects().upload_config(remote_id, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.subjects().get_config(remote_id)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.subjects().delete(remote_id)

    def create_sync_request(self, sync_date: datetime) -> DatabaseSyncRequestDto:
        return DatabaseSyncRequestDto(subjects_sync_date=sync_date)

    def get_cloud_items(self) -> List[Any]:
        return self.api.subjects().get_all(self.db_id)
