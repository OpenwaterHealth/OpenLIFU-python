from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import CreateVolumeRequest, SubjectSyncRequestDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import DATA_FILE, CONFIG_FILE
from openlifu.cloud.sync_thread import SyncThread


class Volumes(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Volumes, self).__init__(api, db_path, database_id, sync_thread)

    def get_config_ids_key(self) -> str:
        return "volume_ids"

    def get_component_type_plural(self) -> str:
        return "volumes"

    def get_sync_date_from_cloud(self) -> Optional[datetime]:
        self._raise_if_no_parent()
        return self.api.subjects().get_one(self.parent_id).volumes_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.subjects().update_subject_sync_date(self.parent_id, SubjectSyncRequestDto(volumes_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        self._raise_if_no_parent()

        if not remote_id:
            remote_id = self.api.volumes().create(
                CreateVolumeRequest(subject_id=self.parent_id, local_id=local_id)
            ).id

        self.api.volumes().upload_file(remote_id, CONFIG_FILE, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.volumes().get_file(remote_id, CONFIG_FILE)

    def upload_data_files(self, local_id: str, remote_id: int, config: dict, modification_date: datetime) -> None:
        if "data_filename" not in config:
            return
        file_name = config["data_filename"]
        path = self.get_directory_path() / local_id / file_name
        if path.is_file():
            data = path.read_bytes()
            self.api.volumes().upload_file(remote_id, DATA_FILE, data, modification_date)

    def download_data_files(self, local_id: str, remote_id: int, config: dict):
        if "data_filename" not in config:
            return
        file_name = config["data_filename"]
        path = self.get_directory_path() / local_id / file_name
        try:
            data = self.api.volumes().get_file(remote_id, DATA_FILE)
            path.write_bytes(data)
        except Exception as e:
            print(e)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.volumes().delete(remote_id)

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return self.api.volumes().get_all(self.parent_id)

    def _raise_if_no_parent(self):
        if self.parent_id is None:
            raise ValueError("Parent ID is required")
