from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import CreateSessionRequest, SubjectSyncRequestDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.const import CONFIG_FILE
from openlifu.cloud.sync_thread import SyncThread


class Sessions(AbstractComponent):

    def __init__(self, api: Api, parent_path: Path, database_id: int, sync_thread: SyncThread):
        super().__init__(api, parent_path, database_id, sync_thread, download_only=False)

    def get_config_ids_key(self) -> str:
        return "session_ids"

    def get_component_type_plural(self) -> str:
        return "sessions"

    def get_sync_date_from_cloud(self) -> datetime | None:
        self._raise_if_no_parent()
        return self.api.subjects().get_one(self.parent_id).sessions_sync_date

    def send_sync_date_to_cloud(self, sync_date: datetime):
        self._raise_if_no_parent()
        self.api.subjects().update_subject_sync_date(self.parent_id, SubjectSyncRequestDto(sessions_sync_date=sync_date))

    def upload_config(self, data: bytes, modification_date: datetime, local_id: str, remote_id: int | None) -> int:
        self._raise_if_no_parent()

        def find_first_id(fn, lst) -> int | None:
            filtered = list(filter(fn, lst))
            if len(filtered) > 0:
                return filtered[0].id
            return None

        if not remote_id:
            config = json.loads(data.decode())
            protocol_local_id = config["protocol_id"]
            volume_local_id = config["volume_id"]
            transducer_local_id = config["transducer_id"]

            protocols = self.api.protocols().get_all(self.db_id)
            volumes = self.api.volumes().get_all(self.parent_id)
            transducers = self.api.transducers().get_all(self.db_id)

            protocol_id = find_first_id(lambda p: p.local_id == protocol_local_id, protocols)
            volume_id = find_first_id(lambda v: v.local_id == volume_local_id, volumes)
            transducer_id = find_first_id(lambda t: t.local_id == transducer_local_id, transducers)

            remote_id = self.api.sessions().create(
                CreateSessionRequest(
                    subject_id=self.parent_id,
                    local_id=local_id,
                    protocol_id=protocol_id,
                    volume_id=volume_id,
                    transducer_id=transducer_id
                )
            ).id

        self.api.sessions().upload_file(remote_id, CONFIG_FILE, data, modification_date)
        return remote_id

    def download_config(self, local_id: str, remote_id: int) -> bytes:
        return self.api.sessions().get_file(remote_id, CONFIG_FILE)

    def delete_on_cloud(self, local_id: str, remote_id: int):
        self.api.sessions().delete(remote_id)

    def get_cloud_items(self) -> List[Any]:
        self._raise_if_no_parent()
        return self.api.sessions().get_all(self.parent_id)

    def should_ignore_path(self, path: Path) -> bool:
        return 'photocollections' in str(path)

    def _raise_if_no_parent(self):
        if self.parent_id is None:
            raise ValueError("Parent ID is required")
