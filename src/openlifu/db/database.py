
import glob
import json
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import List, Optional

import h5py

from openlifu.photoscan import Photoscan
from openlifu.plan import Protocol, Run, Solution
from openlifu.util.json import PYFUSEncoder
from openlifu.xdc import Transducer

from .session import Session
from .subject import Subject

OnConflictOpts = Enum('OnConflictOpts', ['ERROR', 'OVERWRITE', 'SKIP'])


class Database:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = Database.get_default_path()
        self.path = os.path.normpath(path)
        self.logger = logging.getLogger(__name__)

    def write_gridweights(self, transducer_id: str, grid_hash: str, grid_weights, on_conflict: OnConflictOpts = OnConflictOpts.ERROR):
        grid_hashes = self.get_gridweight_hashes(transducer_id)
        if grid_hash in grid_hashes:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Grid weights with hash {grid_hash} already exists for transducer {transducer_id}.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting grid weights with hash {grid_hash} for transducer {transducer_id}.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping grid weights with hash {grid_hash} for transducer {transducer_id} as it already exists.")
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")
        gridweight_filename = self.get_gridweights_filename(transducer_id, grid_hash)
        with h5py.File(gridweight_filename, "w") as f:
            f.create_dataset("grid_weights", data=grid_weights)
        self.logger.info(f"Added grid weights with hash {grid_hash} for transducer {transducer_id} to the database.")

    def write_protocol(self, protocol: Protocol, on_conflict: OnConflictOpts = OnConflictOpts.ERROR):
        # Check if the sonication protocol ID already exists in the database
        protocol_id = protocol.id
        protocol_ids = self.get_protocol_ids()

        if protocol_id in protocol_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Protocol with ID {protocol_id} already exists in the database.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting Protocol with ID {protocol_id} in the database.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping Protocol with ID {protocol_id} as it already exists in the database.")
                return  # Skip adding the Protocol
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        # Save the sonication protocol to a JSON file
        protocol_filename = self.get_protocol_filename(protocol_id)
        protocol.to_file(protocol_filename)

        # Update the list of Protocol IDs
        if protocol_id not in protocol_ids:
            protocol_ids.append(protocol_id)
            self.write_protocol_ids(protocol_ids)

        self.logger.info(f"Added Sonication Protocol with ID {protocol_id} to the database.")


    def write_session(self, subject:Subject, session:Session, on_conflict=OnConflictOpts.ERROR):
        # Generate session ID
        session_id = session.id

        # Validate the subject ID in the session
        if session.subject_id != subject.id:
            raise ValueError("IDs do not match between the given subject and the subject referenced in the session.")

        # Validate the approved fit target ID
        if (
            session.virtual_fit_approval_for_target_id is not None
            and session.virtual_fit_approval_for_target_id not in [target.id for target in session.targets]
        ):
            raise ValueError(
                f"The provided virtual_fit_approval_for_target_id of {session.virtual_fit_approval_for_target_id} is not"
                " in this session's list of targets."
            )

        # Check if the session already exists in the database
        session_ids = self.get_session_ids(subject.id)

        if session_id in session_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Session with ID {session_id} already exists for subject {subject.id}.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting session with ID {session_id} for subject {subject.id}.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping session with ID {session_id} for subject {subject.id} as it already exists.")
                return  # Skip adding the session
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        # Save the session to a JSON file
        session_filename = self.get_session_filename(subject.id, session_id)
        session.update_modified_time()
        session.to_file(session_filename)

        # Create empty runs.json, solutions.json and photoscans.json files for session if needed
        if not self.get_solutions_filename(subject.id, session_id).exists():
            self.write_solution_ids(session, [])
        if not self.get_runs_filename(subject.id, session_id).exists():
            self.write_run_ids(session.subject_id, session.id, [])
        if not self.get_photoscans_filename(subject.id, session_id).exists():
            self.write_photoscan_ids(session.subject_id, session.id, [])

        # Update the list of session IDs for the subject
        if session_id not in session_ids:
            session_ids.append(session_id)
            self.write_session_ids(subject.id, session_ids)

        self.logger.info(f"Added session with ID {session_id} for subject {subject.id} to the database.")

    def write_run(self, run:Run, session:Session = None, protocol:Protocol = None, on_conflict=OnConflictOpts.ERROR):
        """Write a run with a snapshot of session and a snapshot of protocol if provided

        Args:
            run: Run to be written
            session (optional): the Session used in the Run, it will have a snapshot written alongside the Run
            protocol (optional): the Protocol used in the Run, it will have a snapshot written alongside the Run

        Returns:
            None: This method does not return a value
        """
        # Check whether the run already exist in the session
        run_ids = self.get_run_ids(session.subject_id, session.id)

        if run.id in run_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Run with ID {run.id} already exists for session {session.id}.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                raise ValueError("Runs are write-once objects and may not be overwritten.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping run with ID {run.id} for session {session.id} as it already exists.")
                return  # Skip adding the session
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        # Save the run metadata to a JSON file
        run_metadata_filepath = self.get_run_filepath(session.subject_id, session.id, run.id)
        run.to_file(run_metadata_filepath)

        # Update
        run_ids.append(run.id)
        self.write_run_ids(session.subject_id, session.id, run_ids)

        if session:
            # Write snapshot of the session
            session.to_file(run_metadata_filepath.parent / f'{run.id}_session_snapshot.json')

        if protocol:
            # Write snapshot of the protocol
            protocol.to_file(run_metadata_filepath.parent / f'{run.id}_protocol_snapshot.json')

    def write_subject(self, subject, on_conflict=OnConflictOpts.ERROR):
        subject_id = subject.id
        subject_ids = self.get_subject_ids()

        if subject_id in subject_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Subject with ID {subject_id} already exists in the database.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting subject with ID {subject_id} in the database.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping subject with ID {subject_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        subject_filename = self.get_subject_filename(subject_id)
        subject.to_file(subject_filename)

        # Create empty sessions.json and volumes.json files for subject if needed
        if not self.get_sessions_filename(subject.id).exists():
            self.write_session_ids(subject_id, session_ids=[])
        if not self.get_volumes_filename(subject.id).exists():
            self.write_volume_ids(subject_id, volume_ids=[])

        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
            self.write_subject_ids(subject_ids)

        self.logger.info(f"Added subject with ID {subject_id} to the database.")

    def write_transducer(self, transducer, registration_surface_model_filepath: Optional[str] = None, transducer_body_model_filepath: Optional[str] = None, on_conflict: OnConflictOpts=OnConflictOpts.ERROR) -> None:
        """ Writes a transducer object to database and copies the affiliated transducer data files to the database if provided. When a transducer that is already present in the database is being re-written,
        the associated model data files do not need to be provided if they have previously been added to the database.
        Args:
            transducer: Transducer to be written
            transducer_body_model_filepath (optional): Model file containing a mesh of transducer body mesh. This is a closed surface meant for visualization of the transducer.
            registration_surface_model_filepath (optional): Model file containing an open-surface sub-mesh of the transducer body model. This model is meant to be used for registration during transducer tracking.
        Returns:
            None: This method does not return a value
        """
        transducer_id = transducer.id
        transducer_ids = self.get_transducer_ids()

        if transducer_id in transducer_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Transducer with ID {transducer_id} already exists in the database.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting transducer with ID {transducer_id} in the database.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping transducer with ID {transducer_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        transducer_filename = self.get_transducer_filename(transducer_id)
        transducer_parent_dir = transducer_filename.parent
        transducer_parent_dir.mkdir(exist_ok = True)

        # Copy the transducer data files to database.
        # If tranducer data files not provided, check that any files previously
        # associated with the transducer object exist.
        if registration_surface_model_filepath:
            registration_surface_model_filepath = Path(registration_surface_model_filepath)
            if not registration_surface_model_filepath.exists():
                raise FileNotFoundError(f'Registration surface model filepath does not exist: {registration_surface_model_filepath}')
            transducer.registration_surface_filename = registration_surface_model_filepath.name
            shutil.copy(registration_surface_model_filepath, transducer_parent_dir)
        elif transducer.registration_surface_filename:
            if not (transducer_parent_dir/transducer.registration_surface_filename).exists():
                raise ValueError(f"Cannot find registration surface file associated with transducer {transducer.id}.")

        if transducer_body_model_filepath:
            transducer_body_model_filepath = Path(transducer_body_model_filepath)
            if not transducer_body_model_filepath.exists():
                raise FileNotFoundError(f'Transducer body model filepath does not exist: {transducer_body_model_filepath}')
            transducer.transducer_body_filename = transducer_body_model_filepath.name
            shutil.copy(transducer_body_model_filepath, transducer_parent_dir)
        elif transducer.transducer_body_filename:
            if not (transducer_parent_dir/transducer.transducer_body_filename).exists():
                raise ValueError(f"Cannot find transducer body file associated with transducer {transducer.id}.")

        transducer.to_file(transducer_filename)

        if transducer_id not in transducer_ids:
            transducer_ids.append(transducer_id)
            self.write_transducer_ids(transducer_ids)

        self.logger.info(f"Added transducer with ID {transducer_id} to the database.")

    def write_system(self, system, on_conflict=OnConflictOpts.ERROR):
        system_id = system.id
        system_ids = self.get_system_ids()

        if system_id in system_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Ultrasound system with ID {system_id} already exists in the database.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting ultrasound system with ID {system_id} in the database.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping ultrasound system with ID {system_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        system_data = system.to_json()
        system_filename = self.get_system_filename(system_id)
        with open(system_filename, "w") as f:
            json.dump(system_data, f)

        if system_id not in system_ids:
            system_ids.append(system_id)
            self.write_system_ids(system_ids)

        self.logger.info(f"Added ultrasound system with ID {system_id} to the database.")

    def write_volume(self, subject_id, volume_id, volume_name, volume_data_filepath, on_conflict=OnConflictOpts.ERROR):
        if not Path(volume_data_filepath).exists():
            raise ValueError(f'Volume data filepath does not exist: {volume_data_filepath}')

        volume_ids = self.get_volume_ids(subject_id)
        if volume_id in volume_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Volume with ID {volume_id} already exists for subject {subject_id}.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting volume with ID {volume_id} for subject {subject_id}.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping volume with ID {volume_id} for subject {subject_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        # Create volume metadata
        volume_metadata_dict = {"id": volume_id, "name": volume_name, "data_filename": Path(volume_data_filepath).name}
        volume_metadata_json = json.dumps(volume_metadata_dict, separators=(',', ':'), cls=PYFUSEncoder)

        # Save the volume metadata to a JSON file and copy volume data file to database
        volume_metadata_filepath = self.get_volume_metadata_filepath(subject_id, volume_id) #subject_id/volume/volume_id/volume_id.json
        Path(volume_metadata_filepath).parent.parent.mkdir(exist_ok=True) # volume directory
        Path(volume_metadata_filepath).parent.mkdir(exist_ok=True)
        with open(volume_metadata_filepath, 'w') as file:
            file.write(volume_metadata_json)
        shutil.copy(Path(volume_data_filepath), Path(volume_metadata_filepath).parent)

        if volume_id not in volume_ids:
            volume_ids.append(volume_id)
            self.write_volume_ids(subject_id, volume_ids)

        self.logger.info(f"Added volume with ID {volume_id} for subject {subject_id} to the database.")

    def write_photoscan(self, subject_id, session_id, photoscan: Photoscan, model_data_filepath: Optional[str] = None, texture_data_filepath: Optional[str] = None, mtl_data_filepath: Optional[str] = None, on_conflict=OnConflictOpts.ERROR):
        """ Writes a photoscan object to database and copies the associated model and texture data filepaths that are required for generating a photoscan into the database.
        .mtl files are not required for generating a photoscan but can be provided if present.
        When a photoscan that is already present in the database is being re-written,the associated model and texture files do not need to be provided """

        photoscan_ids = self.get_photoscan_ids(subject_id, session_id)
        if photoscan.id in photoscan_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Photoscan with ID {photoscan.id} already exists for session {session_id}.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting photoscan with ID {photoscan.id} for session {session_id}.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping photoscan with ID {photoscan.id} for session {session_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        photoscan_metadata_filepath = Path(self.get_photoscan_metadata_filepath(subject_id, session_id, photoscan.id)) #subject_id/photoscan/photoscan_id/photoscan_id.json
        photoscan_parent_dir = photoscan_metadata_filepath.parent
        photoscan_parent_dir.mkdir(exist_ok=True)

        # Copy photoscan model and texture files to database.
        # If a model and texture data file is not provided, check that there are existing files previously
        # associated with the photoscan object.
        if model_data_filepath:
            model_data_filepath = Path(model_data_filepath)
            if not model_data_filepath.exists():
                raise FileNotFoundError(f'Model data filepath does not exist: {model_data_filepath}')
            photoscan.model_filename = model_data_filepath.name
            shutil.copy(model_data_filepath, photoscan_metadata_filepath.parent)
        elif not photoscan.model_filename or not (photoscan_parent_dir/photoscan.model_filename).exists():
            raise ValueError(f"Cannot find model file associated with photoscan {photoscan.id}.")

        if texture_data_filepath:
            texture_data_filepath = Path(texture_data_filepath)
            if not texture_data_filepath.exists():
                raise FileNotFoundError(f'Texture data filepath does not exist: {texture_data_filepath}')
            photoscan.texture_filename = texture_data_filepath.name
            shutil.copy(texture_data_filepath, photoscan_metadata_filepath.parent)
        elif not photoscan.texture_filename or not (photoscan_parent_dir/photoscan.texture_filename).exists():
            raise ValueError(f"Cannot find texture file associated with photoscan {photoscan.id}.")

        # Not necessarily required for a photoscan object
        if mtl_data_filepath:
            mtl_data_filepath = Path(mtl_data_filepath)
            if not mtl_data_filepath.exists():
                raise FileNotFoundError(f'MTL filepath does not exist: {mtl_data_filepath}')
            photoscan.mtl_filename = mtl_data_filepath.name
            shutil.copy(mtl_data_filepath, photoscan_metadata_filepath.parent)
        elif photoscan.mtl_filename:
            if not (photoscan_parent_dir/photoscan.mtl_filename).exists():
                raise ValueError(f"Cannot find photoscan materials file associated with photoscan {photoscan.id}.")

        #Save the photoscan metadata to a JSON file
        photoscan.to_file(photoscan_metadata_filepath)

        if photoscan.id not in photoscan_ids:
            photoscan_ids.append(photoscan.id)
            self.write_photoscan_ids(subject_id,session_id, photoscan_ids)

        self.logger.info(f"Added photoscan with ID {photoscan.id} for session {session_id} to the database.")

    def write_solution(self, session:Session, solution:Solution, on_conflict: OnConflictOpts=OnConflictOpts.ERROR):
        solution_ids = self.get_solution_ids(session.subject_id, session.id)

        if solution.id in solution_ids:
            if on_conflict == OnConflictOpts.ERROR:
                raise ValueError(f"Solution with ID {solution.id} already exists in the database.")
            elif on_conflict == OnConflictOpts.OVERWRITE:
                self.logger.info(f"Overwriting solution with ID {solution.id} in the database.")
            elif on_conflict == OnConflictOpts.SKIP:
                self.logger.info(f"Skipping solution with ID {solution.id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        solution_json_filepath = self.get_solution_filepath(session.subject_id, session.id, solution.id)
        solution.to_files(solution_json_filepath)

        if solution.id not in solution_ids:
            solution_ids.append(solution.id)
            self.write_solution_ids(session, solution_ids)

        self.logger.info(f"Wrote solution with ID {solution.id} to the database.")

    def choose_session(self, subject, options=None):
        # Implement the logic to choose a session
        raise NotImplementedError("Method not yet implemented")

    def choose_subject(self, options=None):
        # Implement the logic to choose a subject
        raise NotImplementedError("Method not yet implemented")

    def get_gridweight_hashes(self, transducer_id):
        transducer_dir = Path(self.path)/'transducers'/transducer_id
        gridfiles = glob.glob(Path(transducer_dir)/f'{transducer_id}_gridweights_*.h5')
        return [os.path.splitext(os.path.basename(f))[0].split('_')[-1] for f in gridfiles]

    def get_session_table(self, subject_id, options=None):
        # Implement the logic to get session table
        raise NotImplementedError("Method not yet implemented")

    def get_subject_table(self, options=None):
        # Implement the logic to get subject table
        raise NotImplementedError("Method not yet implemented")

    def get_connected_systems(self):
        connected_system_filename = self.get_connected_system_filename()

        if os.path.isfile(connected_system_filename):
            with open(connected_system_filename) as file:
                connected_systems = file.read().strip().split(',')
            self.logger.info("Connected systems: %s", connected_systems)
            return connected_systems
        else:
            self.logger.warning("Connected systems file not found.")
            return []

    def get_connected_transducer(self, options=None):
        connected_transducer_filename = self.get_connected_transducer_filename()

        if os.path.isfile(connected_transducer_filename):
            with open(connected_transducer_filename) as file:
                connected_transducer_id = file.read().strip()
            self.logger.info("Connected transducer: %s", connected_transducer_id)
            return connected_transducer_id
        else:
            self.logger.warning("Connected transducer file not found.")
            return None

    def get_protocol_ids(self):
        protocols_filename = self.get_protocols_filename()

        if os.path.isfile(protocols_filename):
            with open(protocols_filename) as file:
                protocol_data = json.load(file)
                protocol_ids = protocol_data.get("protocol_ids", [])
            self.logger.info("Protocol IDs: %s", protocol_ids)
            return protocol_ids
        else:
            self.logger.warning("Protocols file not found.")
            return []

    def get_session_ids(self, subject_id):
        sessions_filename = self.get_sessions_filename(subject_id)

        if os.path.isfile(sessions_filename):
            with open(sessions_filename) as file:
                session_data = json.load(file)
                session_ids = session_data.get("session_ids", [])
            self.logger.info("Session IDs for subject %s: %s", subject_id, session_ids)
            return session_ids
        else:
            self.logger.info("Sessions file not found for subject %s.", subject_id)
            return []

    def get_run_ids(self, subject_id, session_id):
        runs_filename = self.get_runs_filename(subject_id, session_id)

        if os.path.isfile(runs_filename):
            with open(runs_filename) as file:
                run_data = json.load(file)
                run_ids = run_data.get("run_ids", [])
            self.logger.info("Run IDs for session %s: %s", session_id, run_ids)
            return run_ids
        else:
            self.logger.warning("Runs file not found for session %s.", session_id)
            return []

    def get_volume_ids(self, subject_id):
        volumes_filename = self.get_volumes_filename(subject_id)
        if os.path.isfile(volumes_filename):
            with open(volumes_filename) as file:
                volume_data = json.load(file)
                volume_ids = volume_data.get("volume_ids", [])
            self.logger.info("Volume IDs for subject %s: %s", subject_id, volume_ids)
            return volume_ids
        else:
            self.logger.info("Volumes file not found for subject %s.", subject_id)
            return []

    def get_solution_ids(self, subject_id:str, session_id:str) -> List[str]:
        """Get a list of IDs of the solutions associated with the given session"""
        solutions_filename = self.get_solutions_filename(subject_id, session_id)

        if not (solutions_filename.exists() and solutions_filename.is_file()):
            self.logger.warning("Solutions file not found for subject %s, session %s.", subject_id, session_id)
            return []

        return json.loads(solutions_filename.read_text())["solution_ids"]

    def get_photoscan_ids(self, subject_id: str, session_id: str) -> List[str]:
        """Get a list of IDs of the photoscans associated with the given session"""
        photoscan_filename = self.get_photoscans_filename(subject_id, session_id)

        if not (photoscan_filename.exists() and photoscan_filename.is_file()):
            self.logger.warning("Photoscan file not found for subject %s, session %s.", subject_id, session_id)
            return []

        return json.loads(photoscan_filename.read_text())["photoscan_ids"]

    def get_subject_ids(self):
        subjects_filename = self.get_subjects_filename()

        if os.path.isfile(subjects_filename):
            with open(subjects_filename) as file:
                subject_data = json.load(file)
                subject_ids = subject_data.get("subject_ids", [])
            self.logger.info("Subject IDs: %s", subject_ids)
            return subject_ids
        else:
            self.logger.warning("Subjects file not found.")
            return []

    def get_system_ids(self):
        systems_filename = self.get_systems_filename()

        if os.path.isfile(systems_filename):
            with open(systems_filename) as file:
                system_data = json.load(file)
                system_ids = system_data.get("system_ids", [])
            self.logger.info("System IDs: %s", system_ids)
            return system_ids
        else:
            self.logger.warning("Systems file not found.")
            return []

    def get_system_info(self, sys_id):
        raise NotImplementedError("UltrasoundSystem is not yet implemented")
        system_filename = self.get_system_filename(sys_id)

        if os.path.isfile(system_filename):
            with open(system_filename) as file:
                system_data = json.load(file)
                system_info = UltrasoundSystem.from_dict(system_data)
            self.logger.info("System info for system %s: %s", sys_id, system_info)
            return system_info
        else:
            self.logger.warning("System info file not found for system %s.", sys_id)
            return None

    def get_transducer_ids(self):
        transducers_filename = self.get_transducers_filename()

        if os.path.isfile(transducers_filename):
            with open(transducers_filename) as file:
                transducer_data = json.load(file)
                transducer_ids = transducer_data.get("transducer_ids", [])
            if not isinstance(transducer_ids, list):
                transducer_ids = [transducer_ids]
            self.logger.info("Transducer IDs: %s", transducer_ids)
            return transducer_ids
        else:
            self.logger.warning("Transducers file not found.")
            return []

    def load_gridweights(self, transducer_id, grid_hash):
        gridweight_filename = self.get_gridweights_filename(transducer_id, grid_hash)
        with h5py.File(gridweight_filename, "r") as f:
            grid_weights = f["grid_weights"][:]
        return grid_weights

    def load_subject(self, subject_id, options=None):
        subject_filename = self.get_subject_filename(subject_id)
        subject = Subject.from_file(subject_filename)
        self.logger.info(f"Loaded subject {subject_id}")
        return subject

    def get_volume_info(self, subject_id, volume_id):
        volume_metadata_filepath = self.get_volume_metadata_filepath(subject_id, volume_id)
        with open(volume_metadata_filepath) as f:
            volume = json.load(f)
            return {"id": volume["id"],\
                    "name": volume["name"],\
                    "data_abspath": Path(volume_metadata_filepath).parent/volume["data_filename"]}

    def get_photoscan_info(self, subject_id, session_id, photoscan_id):
        """Returns the photoscan information with absolute paths to any data"""
        photoscan_metadata_filepath = self.get_photoscan_metadata_filepath(subject_id, session_id, photoscan_id)
        with open(photoscan_metadata_filepath) as f:
            photoscan = json.load(f)
            photoscan_dict = {"id": photoscan["id"],\
                    "name": photoscan["name"],\
                    "model_abspath": Path(photoscan_metadata_filepath).parent/photoscan["model_filename"],
                    "texture_abspath": Path(photoscan_metadata_filepath).parent/photoscan["texture_filename"],
                    "photoscan_approved": photoscan["photoscan_approved"]}
            if "mtl_filename" in photoscan:
                photoscan_dict["mtl_abspath"] = Path(photoscan_metadata_filepath).parent/photoscan["mtl_filename"]
        return photoscan_dict

    def load_photoscan(self, subject_id, session_id, photoscan_id):
        """Returns a photoscan object"""
        photoscan_metadata_filepath = self.get_photoscan_metadata_filepath(subject_id, session_id, photoscan_id)
        photoscan = Photoscan.from_file(photoscan_metadata_filepath)
        return photoscan

    def get_transducer_absolute_filepaths(self, transducer_id:str) -> dict:
        """ Returns the absolute filepaths to the model data files i.e. transducer body and registration surface model files affiliated with the transducer, with ID `transducer_id`.
        Unlike `load_transducer`, which specifies the relative paths to the model datafiles along with other transducer attributes, this function only returns the absolute filepaths to
        the datafiles based on the Database directory.
        Args:
            transducer_id: Transducer ID
        Returns:
            dict: A dictionary containing the absolute filepaths to the affiliated transducer data files with the following keys:
                - "id" (str): transducer ID
                - "name" (str): transducer name
                - "registration_surface_abspath" (str): absolute path to the transducer registration surface (open-surface mesh used for transducer tracking registration)
                - "transducer_body_abspath" (str): absolute path to the transducer body model (closed-surface mesh for visualizing the transducer)
        """
        transducer_metadata_filepath = self.get_transducer_filename(transducer_id)
        with open(transducer_metadata_filepath) as f:
            transducer = json.load(f)
            if "registration_surface_filename" not in transducer and "transducer_body_filename" not in transducer:
                raise ValueError(f"Data filepaths not found affiliated with transducer {transducer_id}")
            transducer_filepaths_dict = {"id": transducer["id"],\
                    "name": transducer["name"],
                    }
            if "registration_surface_filename" in transducer:
                transducer_filepaths_dict["registration_surface_abspath"] = Path(transducer_metadata_filepath).parent/transducer["registration_surface_filename"]
            if "transducer_body_filename" in transducer:
                transducer_filepaths_dict["transducer_body_abspath"] = Path(transducer_metadata_filepath).parent/transducer["transducer_body_filename"]
            return transducer_filepaths_dict

    def load_standoff(self, transducer_id, standoff_id="standoff"):
        raise NotImplementedError("Standoff is not yet implemented")
        standoff_filename = self.get_standoff_filename(transducer_id, standoff_id)
        standoff = Standoff.from_file(standoff_filename)
        return standoff

    def load_system(self, sys_id=None):
        raise NotImplementedError("UltrasoundSystem is not yet implemented")
        sys_id = sys_id or self.get_connected_systems()
        sys_filename = self.get_system_filename(sys_id)
        sys = UltrasoundSystem.from_file(sys_filename)
        return sys

    def load_transducer(self, transducer_id) -> "Transducer":
        """Given a transducer_id, reads the corresponding transducer file from database and returns a transducer object.
        Note: the transducer object includes the relative path to the affiliated transducer model data. `get_transducer_absolute_filepaths`, should
        be used to obtain the absolute data filepaths based on the Database directory path.
        Args:
            transducer_id: Transducer ID
        Returns:
            Corresponding Transducer object
        """
        transducer_filename = self.get_transducer_filename(transducer_id)
        transducer = Transducer.from_file(transducer_filename)
        return transducer

    def load_transducer_standoff(self, trans, coords, options=None):
        raise NotImplementedError("Standoff is not yet implemented")
        options = options or {}
        standoff_filename = self.get_standoff_filename(trans.id, "standoff_anchors")
        standoff = Standoff.from_file(standoff_filename)
        # Implement the logic to generate binary mask using standoff and coordinates
        mask = generate_standoff_mask(standoff, coords, options)
        return mask

    def load_protocol(self, protocol_id):
        protocol_filename = self.get_protocol_filename(protocol_id)
        if os.path.isfile(protocol_filename):
            protocol = Protocol.from_file(protocol_filename)
            self.logger.info(f"Loaded Protocol {protocol_id}")
            return protocol
        else:
            self.logger.error(f"Protocol file not found for ID: {protocol_id}")
            raise FileNotFoundError(f"Protocol file not found for ID: {protocol_id}")

    def load_all_protocols(self):
        protocols_filename = self.get_protocols_filename()
        if os.path.isfile(protocols_filename):
            with open(protocols_filename) as file:
                data = json.load(file)
                protocol_ids = data.get('protocol_ids', [])
            protocols = []
            for protocol_id in protocol_ids:
                protocol = self.load_protocol(protocol_id)
                protocols.append(protocol)
            return protocols
        else:
            self.logger.error("Protocols file not found.")
            raise FileNotFoundError("Protocols file not found.")

    def load_session(self, subject, session_id, options=None):
        if options is None:
            options = {}
        session_filename = self.get_session_filename(subject.id, session_id)
        if os.path.isfile(session_filename):
            session = Session.from_file(session_filename)
            self.logger.info(f"Loaded session {session_id} for subject {subject.id}")
            return session
        else:
            self.logger.error(f"Session file not found for ID: {session_id}")
            raise FileNotFoundError(f"Session file not found for ID: {session_id}")

    def load_session_info(self, subject_id, session_id):
        session_filename = self.get_session_filename(subject_id, session_id)
        if os.path.isfile(session_filename):
            with open(session_filename) as file:
                session_data = json.load(file)
            self.logger.info(f"Loaded session info for session {session_id} of subject {subject_id}")
            return session_data
        else:
            self.logger.error(f"Session file not found for ID: {session_id}")
            raise FileNotFoundError(f"Session file not found for ID: {session_id}")

    def load_session_snapshot(self, subject_id, session_id, run_id):
        path_to_run = self.get_run_dir(subject_id, session_id, run_id)
        return Session.from_file(path_to_run / f'{run_id}_session_snapshot.json')

    def load_protocol_snapshot(self, subject_id, session_id, run_id):
        path_to_run = self.get_run_dir(subject_id, session_id, run_id)
        return Protocol.from_file(path_to_run / f'{run_id}_protocol_snapshot.json')

    def load_solution(self, session:Session, solution_id:str) -> Solution:
        """Load the Solution of the given ID that is associated with the given Session"""
        solution_json_filepath = self.get_solution_filepath(session.subject_id, session.id, solution_id)

        if not solution_json_filepath.exists() or not solution_json_filepath.is_file():
            self.logger.error(f"Solution file not found for solution {solution_id}, session {session.id}")
            raise FileNotFoundError(f"Solution file not found for solution {solution_id}, session {session.id}")

        solution = Solution.from_files(solution_json_filepath)
        self.logger.info(f"Loaded solution {solution_id}")
        return solution

    def set_connected_transducer(self, trans, options=None):
        trans_id = trans.id
        transducer_ids = self.get_transducer_ids()
        if trans_id not in transducer_ids:
            if not options or not options.add_if_missing:
                self.logger.error(f"Invalid Transducer ID {trans_id}. Valid IDs are {', '.join(transducer_ids)}")
                raise ValueError(f"Invalid Transducer ID {trans_id}")
            else:
                self.write_transducer(trans)
        filename = self.get_connected_transducer_filename()
        with open(filename, 'w') as f:
            f.write(trans_id)

    def get_connected_system_filename(self):
        return Path(self.path) / "systems" / "connected_system.txt"

    def get_connected_transducer_filename(self):
        return Path(self.path) / 'transducers' / 'connected_transducer.txt'

    def get_gridweights_filename(self, transducer_id, grid_hash):
        return Path(self.path) / 'transducers' / transducer_id / f'{transducer_id}_gridweights_{grid_hash}.h5'

    def get_protocols_filename(self):
        return Path(self.path) / 'protocols' / 'protocols.json'

    def get_protocol_filename(self, protocol_id):
        return Path(self.path) / 'protocols' / protocol_id / f'{protocol_id}.json'

    def get_session_dir(self, subject_id, session_id):
        return Path(self.get_subject_dir(subject_id)) / 'sessions' / session_id

    def get_session_filename(self, subject_id, session_id):
        return Path(self.get_session_dir(subject_id, session_id)) / f'{session_id}.json'

    def get_sessions_filename(self, subject_id) -> Path:
        return Path(self.get_subject_dir(subject_id)) / 'sessions' / 'sessions.json'

    def get_runs_filename(self, subject_id, session_id):
        return Path(self.get_subject_dir(subject_id)) / 'sessions' / f'{session_id}' / 'runs' / 'runs.json'

    def get_volumes_filename(self, subject_id):
        return Path(self.get_subject_dir(subject_id)) / 'volumes' / 'volumes.json'

    def get_solution_filepath(self, subject_id, session_id, solution_id) -> Path:
        """Get the solution json file for the solution with the given ID"""
        session_dir = self.get_session_dir(subject_id, session_id)
        return Path(session_dir) / 'solutions' / solution_id / f"{solution_id}.json"

    def get_solutions_filename(self, subject_id, session_id) -> Path:
        """Get the path to the overall solutions json file for the requested session"""
        session_dir = self.get_session_dir(subject_id, session_id)
        return Path(session_dir) / 'solutions' / 'solutions.json'

    def get_photoscans_filename(self, subject_id, session_id) -> Path:
        """Get the path to the overall photoscans json file for the requested session"""
        session_dir = self.get_session_dir(subject_id, session_id)
        return Path(session_dir) / 'photoscans' / 'photoscans.json'

    def get_standoff_filename(self, transducer_id, standoff_id='standoff'):
        return Path(self.path) / 'transducers' / transducer_id / f'{standoff_id}.json'

    def get_subject_dir(self, subject_id):
        return Path(self.path) / 'subjects' / subject_id

    def get_subject_filename(self, subject_id):
        return Path(self.get_subject_dir(subject_id)) / f'{subject_id}.json'

    def get_subjects_filename(self):
        return Path(self.path) / 'subjects' / 'subjects.json'

    def get_systems_filename(self):
        return Path(self.path) / 'systems' / 'systems.json'

    def get_system_filename(self, system_id):
        return Path(self.path) / 'systems' / system_id / f'{system_id}.json'

    def get_transducer_filename(self, transducer_id):
        return Path(self.path) / 'transducers' / transducer_id / f'{transducer_id}.json'

    def get_transducers_filename(self):
        return Path(self.path) / 'transducers' / 'transducers.json'

    def get_volume_dir(self, subject_id, volume_id):
        return Path(self.get_subject_dir(subject_id)) / 'volumes' / volume_id

    def get_volume_metadata_filepath(self, subject_id, volume_id):
        return Path(self.get_volume_dir(subject_id, volume_id)) / f'{volume_id}.json'

    def get_photoscan_metadata_filepath(self, subject_id, session_id, photoscan_id):
        return Path(self.get_session_dir(subject_id, session_id)) / 'photoscans' / photoscan_id / f'{photoscan_id}.json'

    def get_run_dir(self, subject_id, session_id, run_id):
        run_dir = self.get_session_dir(subject_id, session_id) / 'runs' / f'{run_id}'
        return run_dir

    def get_run_filepath(self, subject_id, session_id, run_id):
        return Path(self.get_run_dir(subject_id, session_id, run_id)) / f'{run_id}.json'

    def write_protocol_ids(self, protocol_ids):
        protocol_data = {'protocol_ids': protocol_ids}
        protocols_filename = self.get_protocols_filename()
        with open(protocols_filename, 'w') as f:
            json.dump(protocol_data, f)

    def write_session_ids(self, subject_id, session_ids):
        session_data = {'session_ids': session_ids}
        sessions_filename = self.get_sessions_filename(subject_id)
        Path(sessions_filename).parent.mkdir(exist_ok=True) #sessions directory
        if not sessions_filename.is_file():
            self.logger.info(f"Adding sessions.json file for subject {subject_id} to the database.")
        with open(sessions_filename, 'w') as f:
            json.dump(session_data, f)

    def write_volume_ids(self, subject_id, volume_ids):
        volume_data = {'volume_ids': volume_ids}
        volumes_filename = self.get_volumes_filename(subject_id)
        Path(volumes_filename).parent.mkdir(exist_ok=True) #volumes directory
        if not volumes_filename.is_file():
            self.logger.info(f"Adding volumes.json file with volume_id {volume_ids} for subject {subject_id} to the database.")
        with open(volumes_filename, 'w') as f:
            json.dump(volume_data, f)

    def write_run_ids(self, subject_id, session_id, run_ids):
        run_data = {'run_ids': run_ids}
        runs_filename = self.get_runs_filename(subject_id, session_id)
        Path(runs_filename).parent.mkdir(exist_ok=True)
        if not runs_filename.is_file():
            self.logger.info(f"Adding runs.json file for session {session_id} of subject {subject_id}.")
        with open(runs_filename, 'w') as f:
            json.dump(run_data, f)

    def write_subject_ids(self, subject_ids):
        subject_data = {'subject_ids': subject_ids}
        subjects_filename = self.get_subjects_filename()
        with open(subjects_filename, 'w') as f:
            json.dump(subject_data, f)

    def write_transducer_ids(self, transducer_ids):
        transducers_data = {'transducer_ids': transducer_ids}
        transducers_filename = self.get_transducers_filename()
        with open(transducers_filename, 'w') as f:
            json.dump(transducers_data, f)

    def write_photoscan_ids(self, subject_id, session_id, photoscan_ids: List[str]):
        photoscan_data = {'photoscan_ids': photoscan_ids}
        photoscan_filename = self.get_photoscans_filename(subject_id, session_id)
        photoscan_filename.parent.mkdir(exist_ok = True) # Make a photoscan directory in case it does not exist
        with open(photoscan_filename, 'w') as f:
            json.dump(photoscan_data,f)

    def write_solution_ids(self, session:Session, solution_ids:List[str]):
        """Write to the list of overall solution IDs"""
        solutions_data = {'solution_ids': solution_ids}
        solutions_filepath = self.get_solutions_filename(session.subject_id, session.id)
        solutions_filepath.parent.mkdir(exist_ok=True) # Make solutions directory in case it does not exist
        solutions_filepath.write_text(json.dumps(solutions_data))

    @staticmethod
    def get_default_user_dir():
        """
        Get the default user directory for the database

        :returns: Default user directory
        """
        return os.path.expanduser("~")

    @staticmethod
    def get_default_path():
        """
        Get the default path for the database

        :returns: Default path for the database
        """
        return Path(Database.get_default_user_dir()) / "Documents" / "db"
