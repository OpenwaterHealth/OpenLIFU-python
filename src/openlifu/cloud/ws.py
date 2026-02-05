from typing import Optional, Callable

import socketio

from openlifu.cloud.const import DEBUG, API_URL

DATABASE_UPDATES_NS = "/database_updates"

class Websocket:

    def __init__(self, update_callback: Callable[[dict], None]):
        self._sio: Optional[socketio.Client] = None
        self._database_id = None
        self._auth = {}
        self._update_callback = update_callback

    def authenticate(self, access_token: str):
        self._auth = {"token": f"Bearer {access_token}"}
        # reconnect with new token
        if self._database_id is not None:
            self.connect(self._database_id)

    def connect(self, database_id: int):
        if self._sio is not None:
            self.disconnect()

        self._database_id = database_id

        if len(self._auth) == 0:
            return

        self._sio = socketio.Client()

        @self._sio.event(namespace=DATABASE_UPDATES_NS)
        def connect():
            if DEBUG:
                print(f"Websocket connected to server, waiting for updates for database {database_id}\n")
            self._sio.emit("subscribe", {"database_id": database_id}, namespace=DATABASE_UPDATES_NS)

        @self._sio.event(namespace=DATABASE_UPDATES_NS)
        def disconnect():
            if DEBUG:
                print("Websocket disconnected from server")

        @self._sio.on("update", namespace=DATABASE_UPDATES_NS)
        def on_update(data):
            if DEBUG:
                print(f"Websocket update: {data}")
            if self._update_callback is not None:
                self._update_callback(data)

        self._sio.connect(
            f"{API_URL}/socket.io",
            auth=self._auth,
            namespaces=[DATABASE_UPDATES_NS],
            transports=["websocket"]
        )

    def disconnect(self):
        if self._sio is not None:
            self._sio.disconnect()
            self._sio.wait()
            self._sio = None
