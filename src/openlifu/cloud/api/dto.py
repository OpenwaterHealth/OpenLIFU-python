from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class UidDto:
    uid: Optional[str]


@dataclass
class CreateObjectRequestDto:
    database_id: int
    local_id: str


@dataclass
class UploadFileArgs:
    modification_date: Optional[datetime]


@dataclass
class ClaimDbDto:
    db_path: str
    mac_address: str
    description: Optional[str]


@dataclass
class DatabaseDto:
    id: Optional[int]
    institution_id: int
    owner_uid: Optional[str]
    db_path: str
    mac_address: str
    description: Optional[str]
    connected_system_id: Optional[int]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]

    protocols_sync_date: Optional[datetime]
    subjects_sync_date: Optional[datetime]
    systems_sync_date: Optional[datetime]
    transducers_sync_date: Optional[datetime]
    users_sync_date: Optional[datetime]


@dataclass
class DatabaseSyncRequestDto:
    protocols_sync_date: Optional[datetime] = field(default=None)
    subjects_sync_date: Optional[datetime] = field(default=None)
    systems_sync_date: Optional[datetime] = field(default=None)
    transducers_sync_date: Optional[datetime] = field(default=None)
    users_sync_date: Optional[datetime] = field(default=None)


@dataclass
class ProtocolDto:
    id: Optional[int]
    database_id: int
    local_id: str
    name: Optional[str]
    description: Optional[str]

    config_file_size: Optional[int]
    allowed_roles: Optional[List[str]]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class SystemDto:
    id: Optional[int]
    database_id: int
    local_id: str
    name: Optional[str]

    config_file_size: Optional[int]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class TransducerDto:
    id: Optional[int]
    database_id: int
    local_id: str
    name: Optional[str]

    config_file_size: Optional[int]

    registration_surface_filename: Optional[str]
    registration_surface_file_size: Optional[int]

    transducer_body_filename: Optional[str]
    transducer_body_file_size: Optional[int]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class CreateRunRequest:
    session_id: int
    local_id: str


@dataclass
class RunDto:
    id: Optional[int]

    session_id: int
    local_id: str

    name: Optional[str]

    config_file_size: Optional[int]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class CreateSessionRequest:
    subject_id: int
    local_id: str
    protocol_id: int
    volume_id: int
    transducer_id: int


@dataclass
class SessionDto:
    id: Optional[int]

    subject_id: int
    local_id: str

    name: Optional[str]

    protocol_id: int
    volume_id: int
    transducer_id: int

    config_file_size: Optional[int]

    photocollections_sync_date: Optional[datetime]
    photoscans_sync_date: Optional[datetime]
    runs_sync_date: Optional[datetime]
    solutions_sync_date: Optional[datetime]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class SessionSyncRequestDto:
    photocollections_sync_date: Optional[datetime] = field(default=None)
    photoscans_sync_date: Optional[datetime] = field(default=None)
    runs_sync_date: Optional[datetime] = field(default=None)
    solutions_sync_date: Optional[datetime] = field(default=None)


@dataclass
class CreateSolutionRequest:
    session_id: int
    local_id: str
    protocol_id: int
    transducer_id: int


@dataclass
class SolutionDto:
    id: Optional[int]
    local_id: str
    name: Optional[str]

    session_id: int
    protocol_id: int
    transducer_id: int

    description: Optional[str]
    approved: bool

    data_file_size: int

    config_file_size: Optional[int]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class CreateSubjectRequest:
    database_id: int
    local_id: str


@dataclass
class SubjectDto:
    id: Optional[int]

    database_id: int
    local_id: str

    name: Optional[str]

    config_file_size: Optional[int]

    sessions_sync_date: Optional[datetime]
    volumes_sync_date: Optional[datetime]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class SubjectSyncRequestDto:
    sessions_sync_date: Optional[datetime] = field(default=None)
    volumes_sync_date: Optional[datetime] = field(default=None)


@dataclass
class CreateVolumeRequest:
    subject_id: int
    local_id: str


@dataclass
class VolumeDto:
    id: Optional[int]

    subject_id: int
    local_id: str

    name: Optional[str]

    config_file_size: Optional[int]

    data_filename: Optional[str]
    data_file_size: Optional[int]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class UserDto:
    uid: str
    database_id: int
    roles: List[str]
    name: Optional[str]
    password_hash: Optional[str]
    description: Optional[str]

    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class CreateUserRequest:
    uid: str
    database_id: int
    roles: List[str]
    name: Optional[str]
    password_hash: Optional[str]
    description: Optional[str]


@dataclass
class PhotoDto:
    file_name: str
    file_size: int
    modification_date: datetime


@dataclass
class PhotocollectionDto:
    id: int
    account_id: str
    session_id: Optional[int]
    name: Optional[str]
    photos: Optional[list[PhotoDto]]
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]


@dataclass
class CreatePhotocollectionRequest:
    session_id: int
    name: str
