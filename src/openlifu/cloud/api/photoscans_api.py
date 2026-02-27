from __future__ import annotations

from openlifu.cloud.api.dto import (
    PagedPhotoscansResponse,
    PhotoscanDto,
)
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json


class PhotoscansApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, session_id: int) -> PagedPhotoscansResponse:
        response = self._request.get(f"/photoscans?session_id={session_id}")
        return from_json(PagedPhotoscansResponse, response)

    def get_one(self, photoscan_id: int) -> PhotoscanDto:
        response = self._request.get(f"/photoscans/{photoscan_id}")
        return from_json(PhotoscanDto, response)

    def get_file(self, photoscan_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/photoscans/{photoscan_id}/file/{file_type}")
