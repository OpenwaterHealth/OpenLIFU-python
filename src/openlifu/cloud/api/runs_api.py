from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import VolumeDto, RunDto, CreateRunRequest
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class RunsApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, session_id: int) -> List[RunDto]:
        response = self._request.get(f"/runs?session_id={session_id}")
        return from_json(List[RunDto], response)

    def get_one(self, run_id: int) -> VolumeDto:
        response = self._request.get(f"/runs/{run_id}")
        return from_json(RunDto, response)

    def create(self, dto: CreateRunRequest) -> RunDto:
        response = self._request.post("/runs", dto)
        return from_json(RunDto, response)

    def delete(self, run_id: int):
        self._request.delete(f"/runs/{run_id}")

    def get_file(self, run_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/runs/{run_id}/file/{file_type}")

    def upload_file(self, run_id: int, file_type: str, file: bytes, modification_date: datetime):
        url = f"/runs/{run_id}/file/{file_type}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)
