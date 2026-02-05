from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import VolumeDto, RunDto, SolutionDto, CreateSolutionRequest
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class SolutionsApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, session_id: int) -> List[SolutionDto]:
        response = self._request.get(f"/solutions?session_id={session_id}")
        return from_json(List[RunDto], response)

    def get_one(self, solution_id: int) -> VolumeDto:
        response = self._request.get(f"/solutions/{solution_id}")
        return from_json(SolutionDto, response)

    def create(self, dto: CreateSolutionRequest) -> SolutionDto:
        response = self._request.post("/solutions", dto)
        return from_json(SolutionDto, response)

    def delete(self, solution_id: int):
        self._request.delete(f"/solutions/{solution_id}")

    def get_file(self, solution_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/solutions/{solution_id}/file/{file_type}")

    def upload_file(self, solution_id: int, file_type: str, file: bytes, modification_date: datetime):
        url = f"/solutions/{solution_id}/file/{file_type}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)
