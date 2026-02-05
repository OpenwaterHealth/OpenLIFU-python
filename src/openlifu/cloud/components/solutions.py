import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import SessionSyncRequestDto, CreateSolutionRequest
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import CONFIG_FILE, DATA_FILE
from openlifu.cloud.sync_thread import SyncThread


class Solutions(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Solutions, self).__init__(api, db_path, database_id, sync_thread)

    def get_config_ids_key(self) -> str:
        return "solution_ids"

    def get_component_type_plural(self) -> str:
        return "solutions"

    def get_sync_date_from_cloud(self) -> Optional[datetime]:
        self._raise_if_no_parent()
        return self.api.sessions().get_one(self.parent_id).solutions_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.sessions().update_session_sync_date(self.parent_id, SessionSyncRequestDto(solutions_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        self._raise_if_no_parent()

        if not remote_id:
            config = json.loads(data.decode())
            protocol_local_id = config["protocol_id"]
            transducer_local_id = config["transducer_id"]

            protocols = self.api.protocols().get_all(self.db_id)
            transducers = self.api.transducers().get_all(self.db_id)

            protocol_id = list(filter(lambda p: p.local_id == protocol_local_id, protocols))[0].id
            transducer_id = list(filter(lambda t: t.local_id == transducer_local_id, transducers))[0].id

            remote_id = self.api.solutions().create(
                CreateSolutionRequest(
                    session_id=self.parent_id,
                    local_id=local_id,
                    protocol_id=protocol_id,
                    transducer_id=transducer_id
                )
            ).id

        self.api.solutions().upload_file(remote_id, CONFIG_FILE, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.solutions().get_file(remote_id, CONFIG_FILE)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.solutions().delete(remote_id)

    def upload_data_files(self, local_id: str, remote_id: int, config: dict, modification_date: datetime) -> None:
        file_name = local_id + '.nc'
        path = self.get_directory_path() / local_id / file_name
        if path.is_file():
            data = path.read_bytes()
            self.api.solutions().upload_file(remote_id, DATA_FILE, data, modification_date)

    def download_data_files(self, local_id: str, remote_id: int, config: dict):
        file_name = local_id + '.nc'
        path = self.get_directory_path() / local_id / file_name
        try:
            data = self.api.solutions().get_file(remote_id, DATA_FILE)
            path.write_bytes(data)
        except Exception as e:
            print(e)

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return self.api.solutions().get_all(self.parent_id)

    def _raise_if_no_parent(self):
        if self.parent_id is None:
            raise ValueError("Parent ID is required")
