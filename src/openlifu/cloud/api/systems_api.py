from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import SystemDto, CreateObjectRequestDto
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class SystemsApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, database_id: int) -> List[SystemDto]:
        response = self._request.get(f"/systems?database_id={database_id}")
        return from_json(List[SystemDto], response)

    def get_one(self, system_id: int) -> SystemDto:
        response = self._request.get(f"/systems/{system_id}")
        return from_json(SystemDto, response)

    def get_config(self, system_id: int) -> bytes:
        return self._request.get_bytes(f"/systems/{system_id}/file/config")

    def upload_config(self, system_id: int, file: bytes, modification_date: datetime):
        url = f"/systems/{system_id}/file/config?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)

    def create(self, dto: CreateObjectRequestDto) -> SystemDto:
        response = self._request.post("/systems", dto)
        return from_json(SystemDto, response)

    def delete(self, system_id: int):
        self._request.delete(f"/systems/{system_id}")
