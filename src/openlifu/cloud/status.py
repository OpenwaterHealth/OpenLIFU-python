from __future__ import annotations


class Status:
    STATUS_IDLE = "idle"
    STATUS_SYNCHRONIZING = "synchronizing"
    STATUS_UPLOADING = "uploading"
    STATUS_DOWNLOADING = "downloading"
    STATUS_DELETING = "deleting"
    STATUS_ERROR = "error"

    def __init__(self, status: str,
                 component_type: str | None = None,
                 local_id: str | None = None,
                 ex: Exception | None=None):
        self.status = status
        self.component_type = component_type
        self.local_id = local_id
        self.exception = ex

    def __str__(self):
        return str(self.__dict__)
