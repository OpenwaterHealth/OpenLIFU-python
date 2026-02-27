from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

from requests import HTTPError

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import PHOTOSCAN_STATUS_FINISHED, SessionSyncRequestDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import (
    CONFIG_FILE,
    MATERIAL_MTL_FILE,
    MATERIAL_PNG_FILE,
    TEXTURED_MESH_FILE,
)
from openlifu.cloud.sync_thread import SyncThread
from openlifu.cloud.utils import logger_cloud


class Photoscans(AbstractComponent):

    def __init__(self, api: Api, parent_path: Path, database_id: int, sync_thread: SyncThread):
        super().__init__(api, parent_path, database_id, sync_thread, download_only=True)

    def get_config_ids_key(self) -> str:
        return "photoscan_ids"

    def get_component_type_plural(self) -> str:
        return "photoscans"

    def get_sync_date_from_cloud(self) -> datetime | None:
        self._raise_if_no_parent()
        return self.api.sessions().get_one(self.parent_id).photoscans_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.sessions().update_session_sync_date(self.parent_id, SessionSyncRequestDto(photoscans_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: int | None) -> int:
        # NOOP
        return 0

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.photoscans().get_file(remote_id, CONFIG_FILE)

    def upload_data_files(self, local_id: str, remote_id: int, config: dict, modification_date: datetime) -> None:
        # NOOP
        return

    def download_data_files(self, local_id: str, remote_id: int, config: dict):
        for path, file_type in self._get_data_file_paths(local_id):
            try:
                self._sync_thread.add_path_to_ignore_list(path)
                data = self.api.photoscans().get_file(remote_id, file_type)
                path.write_bytes(data)
            except (HTTPError, TypeError, OSError) as e:
                logger_cloud.error(f"Failed to download data file {path}: {e}")

    def delete_on_cloud(self, local_id: str, remote_id: int):
        # NOOP
        return

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return [
            p
            for p in self.api.photoscans().get_all(self.parent_id).photoscans
            if p.local_id is not None and p.status == PHOTOSCAN_STATUS_FINISHED
        ]

    def _raise_if_no_parent(self):
        if self.parent_id is None:
            raise ValueError("Parent ID is required")

    def _get_data_file_paths(self, local_id: str) -> List[Tuple[Path, str]]:
        base = self.get_directory_path() / local_id
        png_path = base / "material_0.png"
        mtl_path = base / "material.mtl"
        obj_path = base / "texturedMesh.obj"
        return [
            (png_path, MATERIAL_PNG_FILE),
            (mtl_path, MATERIAL_MTL_FILE),
            (obj_path, TEXTURED_MESH_FILE)
        ]
