from typing import Optional


class Status:
    STATUS_IDLE = "idle"
    STATUS_SYNCHRONIZING = "synchronizing"
    STATUS_UPLOADING = "uploading"
    STATUS_DOWNLOADING = "downloading"
    STATUS_DELETING = "deleting"
    STATUS_ERROR = "error"

    def __init__(self, status: str,
                 component_type: Optional[str] = None,
                 local_id: Optional[str] = None,
                 ex: Optional[Exception]=None):
        self.status = status
        self.component_type = component_type
        self.local_id = local_id
        self.exception = ex

    def __str__(self):
        return str(self.__dict__)
