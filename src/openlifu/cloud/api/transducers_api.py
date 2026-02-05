from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import TransducerDto, CreateObjectRequestDto
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class TransducersApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, database_id: int) -> List[TransducerDto]:
        response = self._request.get(f"/transducers?database_id={database_id}")
        return from_json(List[TransducerDto], response)

    def get_one(self, transducer_id: int) -> TransducerDto:
        response = self._request.get(f"/transducers/{transducer_id}")
        return from_json(TransducerDto, response)

    def create(self, dto: CreateObjectRequestDto) -> TransducerDto:
        response = self._request.post("/transducers", dto)
        return from_json(TransducerDto, response)

    def delete(self, transducer_id: int):
        self._request.delete(f"/transducers/{transducer_id}")

    def get_file(self, transducer_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/transducers/{transducer_id}/file/{file_type}")

    def upload_file(self, transducer_id: int, file_type: str, file: bytes, modification_date: datetime):
        url = f"/transducers/{transducer_id}/file/{file_type}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)
