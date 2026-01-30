from datetime import datetime
from typing import List

from openlifu.cloud.api.request import Request
from openlifu.cloud.api.dto import UserDto, CreateUserRequest
from openlifu.cloud.utils import from_json, to_isoformat


class UsersApi:

    def __init__(self, request: Request):
        self._request = request

    def get_all(self, database_id: int) -> List[UserDto]:
        response = self._request.get(f"/users/local?database_id={database_id}")
        return from_json(List[UserDto], response)

    def get_one(self, database_id: int, uid: str) -> UserDto:
        response = self._request.get(f"/users/local/{database_id}/{uid}")
        return from_json(UserDto, response)

    def create(self, user: CreateUserRequest) -> UserDto:
        response = self._request.post("/users/local", user)
        return from_json(UserDto, response)

    def update(self, database_id: int, uid: str, user: UserDto, modification_date: datetime) -> UserDto:
        url = f"/users/local/{database_id}/{uid}?modification_date={to_isoformat(modification_date)}"
        response = self._request.put(url, user)
        return from_json(UserDto, response)

    def delete(self, database_id: int, uid: str):
        self._request.delete(f"/users/local/{database_id}/{uid}")
