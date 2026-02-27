from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List

from requests import HTTPError

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import SubjectSyncRequestDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import CONFIG_FILE, DATA_FILE
from openlifu.cloud.sync_thread import SyncThread
from openlifu.cloud.utils import logger_cloud


class Volumes(AbstractComponent):

    def __init__(self, api: Api, parent_path: Path, database_id: int, sync_thread: SyncThread):
        super().__init__(api, parent_path, database_id, sync_thread, download_only=True)

    def get_config_ids_key(self) -> str:
        return "volume_ids"

    def get_component_type_plural(self) -> str:
        return "volumes"

    def get_sync_date_from_cloud(self) -> datetime | None:
        self._raise_if_no_parent()
        return self.api.subjects().get_one(self.parent_id).volumes_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.subjects().update_subject_sync_date(self.parent_id, SubjectSyncRequestDto(volumes_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: int | None) -> int:
        # NOOP
        return 0

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.volumes().get_file(remote_id, CONFIG_FILE)

    def upload_data_files(self, local_id: str, remote_id: int, config: dict, modification_date: datetime) -> None:
        # NOOP
        return

    def download_data_files(self, local_id: str, remote_id: int, config: dict):
        if "data_filename" not in config:
            return
        file_name = config["data_filename"]
        path = self.get_directory_path() / local_id / file_name
        try:
            self._sync_thread.add_path_to_ignore_list(path)
            data = self.api.volumes().get_file(remote_id, DATA_FILE)
            path.write_bytes(data)
        except (HTTPError, TypeError, OSError) as e:
            logger_cloud.error(f"Failed to download data file {path}: {e}")

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.volumes().delete(remote_id)

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return self.api.volumes().get_all(self.parent_id)

    def _raise_if_no_parent(self):
        if self.parent_id is None:
            raise ValueError("Parent ID is required")
