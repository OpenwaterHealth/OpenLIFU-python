from __future__ import annotations

from typing import List

from openlifu.cloud.api.dto import VolumeDto
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json


class VolumesApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, subject_id: int) -> List[VolumeDto]:
        response = self._request.get(f"/volumes?subject_id={subject_id}")
        return from_json(List[VolumeDto], response)

    def get_one(self, volume_id: int) -> VolumeDto:
        response = self._request.get(f"/volumes/{volume_id}")
        return from_json(VolumeDto, response)

    def delete(self, volume_id: int):
        self._request.delete(f"/volumes/{volume_id}")

    def get_file(self, volume_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/volumes/{volume_id}/file/{file_type}")
