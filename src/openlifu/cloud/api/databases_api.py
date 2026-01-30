from openlifu.cloud.api.request import Request
from openlifu.cloud.api.dto import ClaimDbDto, DatabaseDto, DatabaseSyncRequestDto
from openlifu.cloud.utils import from_json


class DatabasesApi:

    def __init__(self, request: Request):
        self._request = request

    def claim_database(self, dto: ClaimDbDto) -> DatabaseDto:
        response = self._request.put("/databases/claim", dto)
        return from_json(DatabaseDto, response)

    def release_database(self, database_id: int):
        self._request.delete(f"/databases/{database_id}/owner")

    def get_database(self, database_id: int) -> DatabaseDto:
        response = self._request.get(f"/databases/{database_id}")
        return from_json(DatabaseDto, response)

    def update_database_sync_date(self, database_id: int, dto: DatabaseSyncRequestDto):
        self._request.put(f"/databases/{database_id}/sync", dto)

