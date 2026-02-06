from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import PhotoscanDto, CreatePhotoscanRequest, PagedPhotoscansResponse
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class PhotoscansApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, session_id: int) -> PagedPhotoscansResponse:
        response = self._request.get(f"/photoscans?session_id={session_id}")
        return from_json(PagedPhotoscansResponse, response)

    def get_one(self, photoscan_id: int) -> PhotoscanDto:
        response = self._request.get(f"/photoscans/{photoscan_id}")
        return from_json(PhotoscanDto, response)

    def create(self, dto: CreatePhotoscanRequest) -> PhotoscanDto:
        response = self._request.post("/photoscans", dto)
        return from_json(PhotoscanDto, response)

    def delete(self, photoscan_id: int):
        self._request.delete(f"/photoscans/{photoscan_id}")

    def get_file(self, photoscan_id: int, file_type: str) -> bytes:
        return self._request.get_bytes(f"/photoscans/{photoscan_id}/file/{file_type}")

    def upload_file(self, photoscan_id: int, file_type: str, file: bytes, modification_date: datetime):
        url = f"/photoscans/{photoscan_id}/file/{file_type}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)
