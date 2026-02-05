from datetime import datetime
from typing import List

from openlifu.cloud.api.dto import RunDto, PhotocollectionDto, CreatePhotocollectionRequest
from openlifu.cloud.api.request import Request
from openlifu.cloud.utils import from_json, to_isoformat


class PhotocollectionsApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, session_id: int) -> List[PhotocollectionDto]:
        response = self._request.get(f"/photocollections?session_id={session_id}")
        return from_json(List[PhotocollectionDto], response)

    def get_one(self, _id: int) -> PhotocollectionDto:
        response = self._request.get(f"/photocollections/{_id}?join_photos=true")
        return from_json(PhotocollectionDto, response)

    def create(self, dto: CreatePhotocollectionRequest) -> RunDto:
        response = self._request.post("/photocollections", dto)
        return from_json(PhotocollectionDto, response)

    def delete(self, _id: int):
        self._request.delete(f"/photocollections/{_id}")

    def get_photo(self, photocollection_id: int, file_name: str) -> bytes:
        return self._request.get_bytes(f"/photocollections/{photocollection_id}/photo/{file_name}")

    def delete_photo(self, photocollection_id: int, file_name: str):
        self._request.delete(f"/photocollections/{photocollection_id}/photo/{file_name}")

    def upload_photo(self, photocollection_id: int, file_name: str, file: bytes, modification_date: datetime):
        url = f"/photocollections/{photocollection_id}/photo/{file_name}?modification_date={to_isoformat(modification_date)}"
        self._request.post_bytes(url, file)
