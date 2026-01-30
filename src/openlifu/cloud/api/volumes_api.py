from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import VolumeDto, CreateVolumeRequest
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class VolumesApi:
    CONFIG_FILE = "config"
    DATA_FILE = "data"

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, subject_id: int) -> List[VolumeDto]:
        response = self._request.get(f"/volumes?subject_id={subject_id}")
        return from_json(List[VolumeDto], response)

    def get_one(self, volume_id: int) -> VolumeDto:
        response = self._request.get(f"/volumes/{volume_id}")
        return from_json(VolumeDto, response)

    def create(self, dto: CreateVolumeRequest) -> VolumeDto:
        response = self._request.post("/volumes", dto)
        return from_json(VolumeDto, response)

    def delete(self, volume_id: int):
        self._request.delete(f"/volumes/{volume_id}")

    def get_file(self, volume_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/volumes/{volume_id}/file/{file_type}")

    def upload_file(self, volume_id: int, file_type: str, file: bytes, modification_date: datetime):
        url = f"/volumes/{volume_id}/file/{file_type}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)
