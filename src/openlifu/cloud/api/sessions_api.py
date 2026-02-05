from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import VolumeDto, SessionDto, CreateSessionRequest, \
    SessionSyncRequestDto
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class SessionsApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, subject_id: int) -> List[SessionDto]:
        response = self._request.get(f"/sessions?subject_id={subject_id}")
        return from_json(List[SessionDto], response)

    def get_one(self, session_id: int) -> SessionDto:
        response = self._request.get(f"/sessions/{session_id}")
        return from_json(SessionDto, response)

    def create(self, dto: CreateSessionRequest) -> SessionDto:
        response = self._request.post("/sessions", dto)
        return from_json(SessionDto, response)

    def delete(self, session_id: int):
        self._request.delete(f"/sessions/{session_id}")

    def get_file(self, session_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/sessions/{session_id}/file/{file_type}")

    def upload_file(self, session_id: int, file_type: str, file: bytes, modification_date: datetime):
        url = f"/sessions/{session_id}/file/{file_type}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)

    def update_session_sync_date(self, session_id: int, dto: SessionSyncRequestDto):
        self._request.put(f"/sessions/{session_id}/sync", dto)
