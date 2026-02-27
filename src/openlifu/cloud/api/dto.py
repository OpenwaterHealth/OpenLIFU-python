from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class UidDto:
    uid: str | None


@dataclass
class CreateObjectRequestDto:
    database_id: int
    local_id: str


@dataclass
class UploadFileArgs:
    modification_date: datetime | None


@dataclass
class ClaimDbDto:
    db_path: str
    mac_address: str
    description: str | None


@dataclass
class DatabaseDto:
    id: int | None
    institution_id: int
    owner_uid: str | None
    db_path: str
    mac_address: str
    description: str | None
    connected_system_id: int | None

    creation_date: datetime | None
    modification_date: datetime | None

    protocols_sync_date: datetime | None
    subjects_sync_date: datetime | None
    systems_sync_date: datetime | None
    transducers_sync_date: datetime | None
    users_sync_date: datetime | None


@dataclass
class DatabaseSyncRequestDto:
    protocols_sync_date: datetime | None = field(default=None)
    subjects_sync_date: datetime | None = field(default=None)
    systems_sync_date: datetime | None = field(default=None)
    transducers_sync_date: datetime | None = field(default=None)
    users_sync_date: datetime | None = field(default=None)


@dataclass
class ProtocolDto:
    id: int | None
    database_id: int
    local_id: str
    name: str | None
    description: str | None

    config_file_size: int | None
    allowed_roles: List[str] | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class SystemDto:
    id: int | None
    database_id: int
    local_id: str
    name: str | None

    config_file_size: int | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class TransducerDto:
    id: int | None
    database_id: int
    local_id: str
    name: str | None

    config_file_size: int | None

    registration_surface_filename: str | None
    registration_surface_file_size: int | None

    transducer_body_filename: str | None
    transducer_body_file_size: int | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class CreateRunRequest:
    session_id: int
    local_id: str


@dataclass
class RunDto:
    id: int | None

    session_id: int
    local_id: str

    name: str | None

    config_file_size: int | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class CreateSessionRequest:
    subject_id: int
    local_id: str
    protocol_id: int
    volume_id: int
    transducer_id: int


@dataclass
class SessionDto:
    id: int | None

    subject_id: int
    local_id: str

    name: str | None

    protocol_id: int | None
    volume_id: int | None
    transducer_id: int | None

    config_file_size: int | None

    photoscans_sync_date: datetime | None
    runs_sync_date: datetime | None
    solutions_sync_date: datetime | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class SessionSyncRequestDto:
    photoscans_sync_date: datetime | None = field(default=None)
    runs_sync_date: datetime | None = field(default=None)
    solutions_sync_date: datetime | None = field(default=None)


@dataclass
class CreateSolutionRequest:
    session_id: int
    local_id: str
    protocol_id: int
    transducer_id: int


@dataclass
class SolutionDto:
    id: int | None
    local_id: str
    name: str | None

    session_id: int
    protocol_id: int
    transducer_id: int

    description: str | None
    approved: bool

    data_file_size: int

    config_file_size: int | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class CreateSubjectRequest:
    database_id: int
    local_id: str


@dataclass
class SubjectDto:
    id: int | None

    database_id: int
    local_id: str

    name: str | None

    config_file_size: int | None

    sessions_sync_date: datetime | None
    volumes_sync_date: datetime | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class SubjectSyncRequestDto:
    sessions_sync_date: datetime | None = field(default=None)
    volumes_sync_date: datetime | None = field(default=None)


@dataclass
class VolumeDto:
    id: int | None

    subject_id: int
    local_id: str

    name: str | None

    config_file_size: int | None

    data_filename: str | None
    data_file_size: int | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class UserDto:
    uid: str
    database_id: int
    roles: List[str]
    name: str | None
    password_hash: str | None
    description: str | None

    creation_date: datetime | None
    modification_date: datetime | None


@dataclass
class CreateUserRequest:
    uid: str
    database_id: int
    roles: List[str]
    name: str | None
    password_hash: str | None
    description: str | None


@dataclass
class PhotoDto:
    file_name: str
    file_size: int
    modification_date: datetime


@dataclass
class PhotoscanDto:
    id: int
    account_id: str
    photocollection_id: int
    session_id: int | None
    local_id: str | None
    creation_date: datetime
    modification_date: datetime

    status: str | None
    message: str | None
    progress: int
    status_update_date: datetime | None


@dataclass
class PagedPhotoscansResponse:
    photoscans: List[PhotoscanDto]


PHOTOSCAN_STATUS_FINISHED = "FINISHED"
