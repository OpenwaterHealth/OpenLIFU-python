from datetime import datetime
from typing import List

from openlifu.cloud.api.request import Request
from openlifu.cloud.api.dto import ProtocolDto, CreateObjectRequestDto
from openlifu.cloud.const import CONFIG_FILE
from openlifu.cloud.utils import from_json, to_isoformat


class ProtocolsApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, database_id: int) -> List[ProtocolDto]:
        response = self._request.get(f"/protocols?database_id={database_id}")
        return from_json(List[ProtocolDto], response)

    def get_one(self, protocol_id: int) -> ProtocolDto:
        response = self._request.get(f"/protocols/{protocol_id}")
        return from_json(ProtocolDto, response)

    def get_config(self, protocol_id: int) -> bytes:
        return self._request.get_bytes(f"/protocols/{protocol_id}/file/{CONFIG_FILE}")

    def upload_config(self, protocol_id: int, file: bytes, modification_date: datetime):
        url = f"/protocols/{protocol_id}/file/{CONFIG_FILE}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)

    def create(self, protocol: CreateObjectRequestDto) -> ProtocolDto:
        response = self._request.post("/protocols", protocol)
        return from_json(ProtocolDto, response)

    def delete(self, protocol_id: int):
        self._request.delete(f"/protocols/{protocol_id}")
