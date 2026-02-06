from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Tuple

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import CreateObjectRequestDto, DatabaseSyncRequestDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import BODY_DATA_FILE, SURFACE_DATA_FILE, CONFIG_FILE
from openlifu.cloud.sync_thread import SyncThread


class Transducers(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Transducers, self).__init__(api, db_path, database_id, sync_thread)

    def get_config_ids_key(self) -> str:
        return "transducer_ids"

    def get_component_type_plural(self) -> str:
        return "transducers"

    def get_sync_date_from_cloud(self) -> Optional[datetime]:
        return self.api.databases().get_database(self.db_id).transducers_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self.api.databases().update_database_sync_date(self.db_id, DatabaseSyncRequestDto(transducers_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        if not remote_id:
            remote_id = self.api.transducers().create(
                CreateObjectRequestDto(database_id=self.db_id, local_id=local_id)
            ).id

        self.api.transducers().upload_file(remote_id, CONFIG_FILE, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.transducers().get_file(remote_id, CONFIG_FILE)

    def upload_data_files(self, local_id: str, remote_id: int, config: dict, modification_date: datetime) -> None:
        for path, file_type in self._get_data_file_paths(local_id):
            if path.is_file():
                data = path.read_bytes()
                self.api.transducers().upload_file(remote_id, file_type, data, modification_date)

    def download_data_files(self, local_id: str, remote_id: int, config: dict):
        for path, file_type in self._get_data_file_paths(local_id):
            try:
                self._sync_thread.add_path_to_ignore_list(path)
                data = self.api.transducers().get_file(remote_id, file_type)
                path.write_bytes(data)
            except Exception as e:
                print(e)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.transducers().delete(remote_id)

    def get_cloud_items(self) -> List[Any]:
        return self.api.transducers().get_all(self.db_id)

    def _get_data_file_paths(self, local_id: str) -> List[Tuple[Path, str]]:
        body_file_path = self.get_directory_path() / local_id / f"{local_id}.body.obj"
        surf_file_path = self.get_directory_path() / local_id / f"{local_id}.surf.obj"
        return [
            (body_file_path, BODY_DATA_FILE),
            (surf_file_path, SURFACE_DATA_FILE)
        ]
