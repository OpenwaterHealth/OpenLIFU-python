from openlifu.cloud.api.databases_api import DatabasesApi
from openlifu.cloud.api.protocols_api import ProtocolsApi
from openlifu.cloud.api.request import Request
from openlifu.cloud.api.subjects_api import SubjectsApi
from openlifu.cloud.api.systems_api import SystemsApi
from openlifu.cloud.api.transducers_api import TransducersApi
from openlifu.cloud.api.users_api import UsersApi
from openlifu.cloud.api.volumes_api import VolumesApi


class Api:

    def __init__(self):
        self._request = Request()
        self._request.debug_log = True
        self._databases = DatabasesApi(self._request)
        self._protocols = ProtocolsApi(self._request)
        self._users = UsersApi(self._request)
        self._systems = SystemsApi(self._request)
        self._transducers = TransducersApi(self._request)
        self._subjects = SubjectsApi(self._request)
        self._volumes = VolumesApi(self._request)

    def authenticate(self, token: str):
        self._request.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def logout(self):
        self._request.headers = {}

    def databases(self) -> DatabasesApi:
        return self._databases

    def protocols(self) -> ProtocolsApi:
        return self._protocols

    def users(self) -> UsersApi:
        return self._users

    def systems(self) -> SystemsApi:
        return self._systems

    def transducers(self) -> TransducersApi:
        return self._transducers

    def subjects(self) -> SubjectsApi:
        return self._subjects

    def volumes(self) -> VolumesApi:
        return self._volumes
