from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import SessionSyncRequestDto, CreateRunRequest
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import CONFIG_FILE
from openlifu.cloud.sync_thread import SyncThread


class Runs(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Runs, self).__init__(api, db_path, database_id, sync_thread)

    def get_config_ids_key(self) -> str:
        return "run_ids"

    def get_component_type_plural(self) -> str:
        return "runs"

    def get_sync_date_from_cloud(self) -> Optional[datetime]:
        self._raise_if_no_parent()
        return self.api.sessions().get_one(self.parent_id).runs_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.sessions().update_session_sync_date(self.parent_id, SessionSyncRequestDto(runs_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        self._raise_if_no_parent()

        if not remote_id:
            remote_id = self.api.runs().create(
                CreateRunRequest(session_id=self.parent_id, local_id=local_id)
            ).id

        self.api.runs().upload_file(remote_id, CONFIG_FILE, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.runs().get_file(remote_id, CONFIG_FILE)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.runs().delete(remote_id)

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return self.api.runs().get_all(self.parent_id)

    def _raise_if_no_parent(self):
        if self.parent_id is None:
            raise ValueError("Parent ID is required")
