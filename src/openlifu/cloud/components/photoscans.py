from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Tuple

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import SessionSyncRequestDto, CreatePhotoscanRequest
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import CONFIG_FILE, MATERIAL_PNG_FILE, MATERIAL_MTL_FILE, TEXTURED_MESH_FILE
from openlifu.cloud.sync_thread import SyncThread


class Photoscans(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Photoscans, self).__init__(api, db_path, database_id, sync_thread)

    def get_config_ids_key(self) -> str:
        return "photoscan_ids"

    def get_component_type_plural(self) -> str:
        return "photoscans"

    def get_sync_date_from_cloud(self) -> Optional[datetime]:
        self._raise_if_no_parent()
        return self.api.sessions().get_one(self.parent_id).photoscans_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.sessions().update_session_sync_date(self.parent_id, SessionSyncRequestDto(photoscans_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        if not remote_id:
            photocollection_local_id = local_id[:local_id.index("_")] if "_" in local_id else local_id
            photocollections = self.api.photocollections().get_all(self.parent_id)
            photocollection_id = None
            for photocollection in photocollections:
                if photocollection.name == photocollection_local_id:
                    photocollection_id = photocollection.id

            if photocollection_id is None:
                raise ValueError(f"Can't find photocollection for photoscan {local_id}")

            remote_id = self.api.photoscans().create(
                CreatePhotoscanRequest(
                    session_id=self.parent_id,
                    photocollection_id=photocollection_id,
                    local_id=local_id
                )
            ).id

        self.api.photoscans().upload_file(remote_id, CONFIG_FILE, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.photoscans().get_file(remote_id, CONFIG_FILE)

    def upload_data_files(self, local_id: str, remote_id: int, config: dict, modification_date: datetime) -> None:
        for path, file_type in self._get_data_file_paths(local_id):
            if path.is_file():
                data = path.read_bytes()
                self.api.photoscans().upload_file(remote_id, file_type, data, modification_date)

    def download_data_files(self, local_id: str, remote_id: int, config: dict):
        for path, file_type in self._get_data_file_paths(local_id):
            try:
                self._sync_thread.add_path_to_ignore_list(path)
                data = self.api.photoscans().get_file(remote_id, file_type)
                path.write_bytes(data)
            except Exception as e:
                print(e)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.photoscans().delete(remote_id)

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return [p for p in self.api.photoscans().get_all(self.parent_id).photoscans if p.local_id is not None]

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