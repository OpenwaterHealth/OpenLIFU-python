import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from requests import HTTPError

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import DatabaseSyncRequestDto, CreateUserRequest
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.sync_thread import SyncThread


class Users(AbstractComponent):

    def __init__(self, api: Api, db_path: Path, database_id: int, sync_thread: SyncThread):
        super(Users, self).__init__(api, db_path, database_id, sync_thread)

    def get_config_ids_key(self) -> str:
        return "user_ids"

    def get_component_type_plural(self) -> str:
        return "users"

    def get_cloud_item_id(self, cloud_item) -> int:
        # not available for users
        return 0

    def get_cloud_item_local_id(self, cloud_item) -> str:
        return cloud_item.uid

    def get_sync_date_from_cloud(self) -> Optional[datetime]:
        return self.api.databases().get_database(self.db_id).users_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self.api.databases().update_database_sync_date(self.db_id, DatabaseSyncRequestDto(users_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: Optional[int]) -> int:
        data_dict = json.loads(data.decode())

        try:
            user = self.api.users().get_one(self.db_id, local_id)

            user.name = data_dict["name"]
            user.password_hash = data_dict["password_hash"]
            user.roles = data_dict["roles"]
            user.description = data_dict["description"]

            self.api.users().update(self.db_id, local_id, user, modification_date)
        except HTTPError as e:
            if e.response.status_code == 404:
                self.api.users().create(
                    CreateUserRequest(
                        uid=local_id,
                        database_id=self.db_id,
                        roles=data_dict["roles"],
                        name=data_dict["name"],
                        password_hash=data_dict["password_hash"],
                        description=data_dict["description"]
                    )
                )
            else:
                raise e
        return 0

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        user = self.api.users().get_one(self.db_id, local_id)
        data = {
            "id": user.uid,
            "password_hash": user.password_hash,
            "roles": user.roles,
            "name": user.name,
            "description": user.description
        }
        return json.dumps(data).encode()

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.users().delete(self.db_id, local_id)

    def get_cloud_items(self) -> List[Any]:
        return self.api.users().get_all(self.db_id)
