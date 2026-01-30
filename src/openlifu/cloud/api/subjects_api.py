from datetime import datetime
from typing import List

from openlifu.cloud.api.request import Request
from openlifu.cloud.api.dto import SubjectDto, CreateObjectRequestDto
from openlifu.cloud.utils import from_json, to_isoformat


class SubjectsApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, database_id: int) -> List[SubjectDto]:
        response = self._request.get(f"/subjects?database_id={database_id}")
        return from_json(List[SubjectDto], response)

    def get_one(self, subject_id: int) -> SubjectDto:
        response = self._request.get(f"/subjects/{subject_id}")
        return from_json(SubjectDto, response)

    def get_config(self, subject_id: int) -> bytes:
        return self._request.get_bytes(f"/subjects/{subject_id}/file/config")

    def upload_config(self, subject_id: int, file: bytes, modification_date: datetime):
        url = f"/subjects/{subject_id}/file/config?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)

    def create(self, dto: CreateObjectRequestDto) -> SubjectDto:
        response = self._request.post("/subjects", dto)
        return from_json(SubjectDto, response)

    def delete(self, subject_id: int):
        self._request.delete(f"/subjects/{subject_id}")
