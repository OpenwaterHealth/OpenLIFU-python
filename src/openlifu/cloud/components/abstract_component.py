import json
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import DatabaseSyncRequestDto
from openlifu.cloud.const import DEBUG
from openlifu.cloud.status import Status
from openlifu.cloud.sync_thread import SyncThread
from openlifu.cloud.utils import mtime


class AbstractComponent(ABC):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        self.api = api
        self.db_path = db_path
        self.db_id = database_id
        self._sync_thread = sync_thread

    @abstractmethod
    def get_component_type(self) -> str:
        pass

    @abstractmethod
    def get_component_type_plural(self) -> str:
        pass

    @abstractmethod
    def get_cloud_modification_date(self) -> Optional[datetime]:
        pass

    @abstractmethod
    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        pass

    @abstractmethod
    def download_config(self, local_id: str, remote_id: int) -> bytes:
        pass

    @abstractmethod
    def delete_on_cloud(self, local_id: str, remote_id: int):
        pass

    @abstractmethod
    def create_sync_request(self, sync_date: datetime) -> DatabaseSyncRequestDto:
        pass

    @abstractmethod
    def get_cloud_items(self) -> List[Any]:
        pass

    def get_cloud_item_id(self, cloud_item) -> int:
        return cloud_item.id

    def get_cloud_item_local_id(self, cloud_item) -> str:
        return cloud_item.local_id

    def upload_data_files(self, local_id: str, remote_id: int, modification_date: datetime):
        pass

    def download_data_files(self, local_id: str, remote_id: int):
        pass

    def on_filesystem_change(self, path: Path):
        if path.resolve().is_relative_to(self.get_directory_path().resolve()):
            self._sync_thread.post(self)

    def get_directory_path(self) -> Path:
        return self.db_path / self.get_component_type_plural()

    def get_ids_file_name(self) -> str:
        return f"{self.get_component_type_plural()}.json"

    def read_local_ids(self) -> List[str]:
        with open(self.get_directory_path() / self.get_ids_file_name(), "r") as f:
            data = json.load(f)
            return data[f"{self.get_component_type()}_ids"]

    def write_local_ids(self, local_ids: List[str]):
        with open(self.get_directory_path() / self.get_ids_file_name(), "w") as f:
            data = {f"{self.get_component_type()}_ids": local_ids}
            f.write(json.dumps(data))

    def get_local_modification_date(self) -> datetime:
        path = self.get_directory_path() / self.get_ids_file_name()
        return mtime(path)

    def upload(self, local_id: str, remote_id: Optional[int]):
        path = self.get_directory_path() / f"{local_id}/{local_id}.json"
        if path.is_file():
            self._sync_thread.emit_status(Status(Status.STATUS_UPLOADING, self.get_component_type_plural(), local_id))
            data = path.read_text()
            config_mtime = mtime(path)

            remote_id = self.upload_config(data.encode(), config_mtime, local_id, remote_id)
            self.upload_data_files(local_id, remote_id, config_mtime)

    def download(self, local_id: str, remote_id: int):
        self._sync_thread.emit_status(Status(Status.STATUS_DOWNLOADING, self.get_component_type_plural(), local_id))
        dir_path = self.get_directory_path() / local_id
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        data = self.download_config(local_id, remote_id)

        with open(dir_path / f"{local_id}.json", "w") as f:
            f.write(data.decode())

        self.download_data_files(local_id, remote_id)

    def delete_local(self, local_id: str):
        self._sync_thread.emit_status(Status(Status.STATUS_DELETING, self.get_component_type_plural(), local_id))
        dir_path = self.get_directory_path() / local_id
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)

    def sync_item(self, local_id: str, remote_id: int, remote_mtime: Optional[datetime]):
        path = self.get_directory_path() / local_id / f"{local_id}.json"
        if path.exists():
            local_mtime = mtime(path)

            if remote_mtime > local_mtime:
                self.download(local_id, remote_id)
            elif remote_mtime < local_mtime:
                self.upload(local_id, remote_id)
        else:
            self.download(local_id, remote_id)

    def sync(self):
        if DEBUG:
            print(f"================== {self.get_component_type_plural()} SYNC =================")

        local_ids = self.read_local_ids()
        remote_items = self.get_cloud_items()
        local_mtime = self.get_local_modification_date()
        remote_mtime = self.get_cloud_modification_date()

        # no added/deleted items, all components known
        if local_mtime == remote_mtime:
            for cloud_item in remote_items:
                remote_id = self.get_cloud_item_id(cloud_item)
                local_id = self.get_cloud_item_local_id(cloud_item)
                self.sync_item(local_id, remote_id, cloud_item.modification_date)

        # first time components
        if not remote_mtime:
            for local_id in local_ids:
                remote_id = None
                for cloud_item in remote_items:
                    if local_id == self.get_cloud_item_local_id(cloud_item):
                        remote_id = self.get_cloud_item_id(cloud_item)

                self.upload(local_id, remote_id)
        elif remote_mtime > local_mtime: # cloud has more recent data
            if DEBUG:
                print(
                    f"===== SYNC FROM CLOUD ===== {self.get_component_type_plural()}: {local_ids}, LOCAL: {local_mtime}, REMOTE: {remote_mtime}")

            self.sync_from_cloud(local_ids, remote_items)
        elif remote_mtime < local_mtime: # local data is more recent than on cloud
            if DEBUG:
                print(f"===== SYNC TO CLOUD ===== {self.get_component_type_plural()}: {local_ids}, LOCAL: {local_mtime}, REMOTE: {remote_mtime}")

            self.sync_to_cloud(local_ids, remote_items)

        self.api.databases().update_database_sync_date(self.db_id, self.create_sync_request(local_mtime))

    def sync_from_cloud(self, local_ids: List[str], cloud_items: List[Any]):
        local_ids_from_cloud = [self.get_cloud_item_local_id(item) for item in cloud_items]

        # delete local items that doesn't exist on cloud
        for local_id in local_ids:
            if local_id not in local_ids_from_cloud:
                self.delete_local(local_id)

        for cloud_item in cloud_items:
            remote_id = self.get_cloud_item_id(cloud_item)
            local_id = self.get_cloud_item_local_id(cloud_item)

            # update local item
            if local_id in local_ids:
                self.sync_item(local_id, remote_id, cloud_item.modification_date)
            else:  # create new local item
                self.download(local_id, remote_id)

        self.write_local_ids(local_ids_from_cloud)

    def sync_to_cloud(self, local_ids: List[str], cloud_items: List[Any]):
        local_ids_from_cloud = [self.get_cloud_item_local_id(item) for item in cloud_items]

        for cloud_item in cloud_items:
            remote_id = self.get_cloud_item_id(cloud_item)
            local_id = self.get_cloud_item_local_id(cloud_item)

            if local_id not in local_ids: # delete item on cloud
                self._sync_thread.emit_status(Status(Status.STATUS_DELETING, self.get_component_type_plural(), local_id))
                self.delete_on_cloud(local_id, remote_id)
            else: # item exists locally and on cloud
                self.sync_item(local_id, remote_id, cloud_item.modification_date)

        for local_id in local_ids:
            # create new item on cloud
            if local_id not in local_ids_from_cloud:
                self.upload(local_id, None)
