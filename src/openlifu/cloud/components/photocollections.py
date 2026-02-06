import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import SessionSyncRequestDto, CreatePhotocollectionRequest, PhotoDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.status import Status
from openlifu.cloud.sync_thread import SyncThread
from openlifu.cloud.utils import mtime


class Photocollections(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Photocollections, self).__init__(api, db_path, database_id, sync_thread)

    def get_config_ids_key(self) -> str:
        return "reference_numbers"

    def get_component_type_plural(self) -> str:
        return "photocollections"

    def get_sync_date_from_cloud(self) -> Optional[datetime]:
        self._raise_if_no_parent()
        return self.api.sessions().get_one(self.parent_id).photocollections_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.sessions().update_session_sync_date(self.parent_id, SessionSyncRequestDto(photocollections_sync_date=sync_date))

    def get_cloud_item_local_id(self, cloud_item) -> str:
        return cloud_item.name

    def sync_item(self, local_id: str, remote_id: int, remote_mtime: Optional[datetime]):
        dir_path = self.get_directory_path() / local_id
        if dir_path.exists():
            self._sync_thread.add_path_to_ignore_list(dir_path)

            local_photos = [
                PhotoDto(
                    file_name=p.name,
                    file_size=p.stat().st_size,
                    modification_date=mtime(p)
                )
                for p in self._get_local_photo_paths(local_id)
            ]
            remote_photos = self.api.photocollections().get_one(remote_id).photos
            if remote_photos is None:
                remote_photos = []

            local_by_name: Dict[str, PhotoDto] = {p.file_name: p for p in local_photos}
            remote_by_name: Dict[str, PhotoDto] = {p.file_name: p for p in remote_photos}

            photos_to_download: List[PhotoDto] = []
            photos_to_upload: List[PhotoDto] = []

            last_local_modification_date = max(
                (p.modification_date for p in local_photos),
                default=datetime.min,
            )

            # Decide sync direction
            sync_from_cloud = remote_mtime > last_local_modification_date if remote_mtime else False

            if sync_from_cloud:
                # ========= CLOUD -> LOCAL =========

                # Delete local files not present in cloud
                for name, local in local_by_name.items():
                    if name not in remote_by_name:
                        file_path = dir_path / name
                        file_path.unlink(missing_ok=True)

                # Download new or updated cloud files
                for name, remote in remote_by_name.items():
                    local = local_by_name.get(name)
                    if (
                            local is None
                            or remote.modification_date > local.modification_date
                            or remote.file_size != local.file_size
                    ):
                        photos_to_download.append(remote)
            else:
                # ========= LOCAL -> CLOUD =========

                # Delete cloud files not present locally
                for name, remote in remote_by_name.items():
                    if name not in local_by_name:
                        self.api.photocollections().delete_photo(remote_id, name)

                # Upload new or updated local files
                for name, local in local_by_name.items():
                    remote = remote_by_name.get(name)
                    if (
                            remote is None
                            or local.file_size != remote.file_size
                    ):
                        photos_to_upload.append(local)

            if len(photos_to_upload) > 0:
                self._upload_photos(remote_id, local_id, [dir_path / p.file_name for p in photos_to_upload])

            if len(photos_to_download) > 0:
                self._download_photos(remote_id, local_id, dir_path, [p.file_name for p in photos_to_download])
        else:
            self.download(local_id, remote_id)

    def upload(self, local_id: str, remote_id: Optional[int]):
        self._sync_thread.add_path_to_ignore_list(self.get_directory_path() / local_id)
        paths = self._get_local_photo_paths(local_id)
        if len(paths) > 0:
            if not remote_id:
                remote_id = self.api.photocollections().create(
                    CreatePhotocollectionRequest(session_id=self.parent_id, name=local_id)
                ).id
            self._upload_photos(remote_id, local_id, list(paths))

    def download(self, local_id: str, remote_id: int):
        dir_path = self.get_directory_path() / local_id
        self._sync_thread.add_path_to_ignore_list(dir_path)

        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path.mkdir(parents=True)

        photocollection = self.api.photocollections().get_one(remote_id)
        if photocollection.photos is None:
            return
        self._download_photos(remote_id, local_id, dir_path, [p.file_name for p in photocollection.photos])

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.photocollections().delete(remote_id)

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return self.api.photocollections().get_all(self.parent_id)

    def _upload_photos(self, photocollection_id: int, local_id: str, file_paths: List[Path]):
        self.emit_status(local_id, Status.STATUS_UPLOADING)
        def ul(photo_path):
            self.api.photocollections().upload_photo(
                photocollection_id, photo_path.name, photo_path.read_bytes(), mtime(photo_path)
            )
        with ThreadPoolExecutor(max_workers=16) as ex:
            ex.map(ul, file_paths)

    def _download_photos(self, photocollection_id: int, local_id: str, dir_path: Path, file_names: List[str]):
        self.emit_status(local_id, Status.STATUS_DOWNLOADING)

        def dl(file_name):
            data = self.api.photocollections().get_photo(photocollection_id, file_name)
            photo_path = dir_path / file_name
            self._sync_thread.add_path_to_ignore_list(photo_path)
            photo_path.write_bytes(data)

        with ThreadPoolExecutor(max_workers=16) as ex:
            ex.map(dl, file_names)

    def _get_local_photo_paths(self, local_id: str) -> List[Path]:
        path = self.get_directory_path() / local_id
        if not path.is_dir():
            return []
        exts = [".jpg", ".jpeg"]
        paths = sorted(path.rglob("*"), key=lambda p: p.name)
        return list(filter(lambda p: p.is_file() and p.suffix.lower() in exts, paths))

    def _raise_if_no_parent(self):
        if self.parent_id is None:
            raise ValueError("Parent ID is required")

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        pass #NOOP

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        pass #NOOP
