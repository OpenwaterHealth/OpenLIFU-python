
import os
import json
import scipy.io
from typing import List
import numpy as np
import logging
import h5py
import glob
from typing import Literal, Optional
from openlifu.plan import Protocol, Solution
from openlifu.db import Subject
OnConflictOpts = Literal['error', 'overwrite', 'skip']


class Database:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = Database.get_default_path()
        self.path = os.path.normpath(path)
        self.logger = logging.getLogger(__name__)

    def add_gridweights(self, transducer_id: str, grid_hash: str, grid_weights, on_conflict: OnConflictOpts =b"error"):
        grid_hashes = self.get_gridweight_hashes(transducer_id)
        if grid_hash in grid_hashes:
            if on_conflict == "error":
                raise ValueError(f"Grid weights with hash {grid_hash} already exists for transducer {transducer_id}.")
            elif on_conflict == "overwrite":
                self.logger.warning(f"Overwriting grid weights with hash {grid_hash} for transducer {transducer_id}.")
            elif on_conflict == "skip":
                self.logger.info(f"Skipping grid weights with hash {grid_hash} for transducer {transducer_id} as it already exists.")
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")
        gridweight_filename = self.get_gridweights_filename(transducer_id, grid_hash)
        with h5py.File(gridweight_filename, "w") as f:
            f.create_dataset("grid_weights", data=grid_weights)
        self.logger.info(f"Added grid weights with hash {grid_hash} for transducer {transducer_id} to the database.")

    def add_protocol(self, protocol: Protocol, on_conflict: OnConflictOpts = "error"):
        # Check if the sonication protocol ID already exists in the database
        protocol_id = protocol.id
        protocol_ids = self.get_protocol_ids()
        
        if protocol_id in protocol_ids:
            if on_conflict == "error":
                raise ValueError(f"Protocol with ID {protocol_id} already exists in the database.")
            elif on_conflict == "overwrite":
                self.logger.warning(f"Overwriting Protocol with ID {protocol_id} in the database.")
            elif on_conflict == "skip":
                self.logger.info(f"Skipping Protocol with ID {protocol_id} as it already exists in the database.")
                return  # Skip adding the Protocol
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        # Serialize the sonication protocol to JSON
        protocol_dict = protocol.to_dict()

        # Save the sonication protocol to a JSON file
        protocol_filename = self.get_protocol_filename(protocol_id)
        with open(protocol_filename, "w") as f:
            json.dump(protocol_dict, f)

        # Update the list of Protocol IDs
        if protocol_id not in protocol_ids:
            protocol_ids.append(protocol_id)
            self.write_protocol_ids(protocol_ids)

        self.logger.info(f"Added Sonication Protocol with ID {protocol_id} to the database.")


    def add_session(self, subject, session, on_conflict="error"):
        # Generate session ID
        session_id = session.id

        # Check if the session already exists in the database
        session_ids = self.get_session_ids(subject.id)

        if session_id in session_ids:
            if on_conflict == "error":
                raise ValueError(f"Session with ID {session_id} already exists for subject {subject.id}.")
            elif on_conflict == "overwrite":
                self.logger.warning(f"Overwriting session with ID {session_id} for subject {subject.id}.")
            elif on_conflict == "skip":
                self.logger.info(f"Skipping session with ID {session_id} for subject {subject.id} as it already exists.")
                return  # Skip adding the session
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        # Save the session to a JSON file
        session_filename = self.get_session_filename(subject.id, session_id)
        session.to_file(session_filename)

        # Update the list of session IDs for the subject
        if session_id not in session_ids:
            session_ids.append(session_id)
            self.write_session_ids(subject.id, session_ids)

        self.logger.info(f"Added session with ID {session_id} for subject {subject.id} to the database.")

    def add_subject(self, subject, on_conflict="error"):
        subject_id = subject.id
        subject_ids = self.get_subject_ids()

        if subject_id in subject_ids:
            if on_conflict == "error":
                raise ValueError(f"Subject with ID {subject_id} already exists in the database.")
            elif on_conflict == "overwrite":
                self.logger.warning(f"Overwriting subject with ID {subject_id} in the database.")
            elif on_conflict == "skip":
                self.logger.info(f"Skipping subject with ID {subject_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        subject_filename = self.get_subject_filename(subject_id)
        subject.to_file(subject_filename)

        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
            self.write_subject_ids(subject_ids)

        self.logger.info(f"Added subject with ID {subject_id} to the database.")

    def add_transducer(self, transducer, on_conflict: OnConflictOpts="error"):
        transducer_id = transducer.id
        transducer_ids = self.get_transducer_ids()

        if transducer_id in transducer_ids:
            if on_conflict == "error":
                raise ValueError(f"Transducer with ID {transducer_id} already exists in the database.")
            elif on_conflict == "overwrite":
                self.logger.warning(f"Overwriting transducer with ID {transducer_id} in the database.")
            elif on_conflict == "skip":
                self.logger.info(f"Skipping transducer with ID {transducer_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        transducer_filename = self.get_transducer_filename(transducer_id)
        transducer.to_file(transducer_filename)

        if transducer_id not in transducer_ids:
            transducer_ids.append(transducer_id)
            self.write_transducer_ids(transducer_ids)

        self.logger.info(f"Added transducer with ID {transducer_id} to the database.")

    def add_system(self, system, on_conflict="error"):
        system_id = system.id
        system_ids = self.get_system_ids()

        if system_id in system_ids:
            if on_conflict == "error":
                raise ValueError(f"Ultrasound system with ID {system_id} already exists in the database.")
            elif on_conflict == "overwrite":
                self.logger.warning(f"Overwriting ultrasound system with ID {system_id} in the database.")
            elif on_conflict == "skip":
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

    def add_volume(self, subject, volume, on_conflict="error"):
        volume_id = volume.id
        subject_id = subject.id

        volume_ids = self.get_volume_ids(subject_id)
        if volume_id in volume_ids:
            if on_conflict == "error":
                raise ValueError(f"Volume with ID {volume_id} already exists for subject {subject_id}.")
            elif on_conflict == "overwrite":
                self.logger.warning(f"Overwriting volume with ID {volume_id} for subject {subject_id}.")
            elif on_conflict == "skip":
                self.logger.info(f"Skipping volume with ID {volume_id} for subject {subject_id} as it already exists.")
                return
            else:
                raise ValueError("Invalid 'on_conflict' option. Use 'error', 'overwrite', or 'skip'.")

        volume_data = volume.to_json()
        volume_filename = self.get_volume_filename(subject_id, volume_id)
        with open(volume_filename, "w") as f:
            json.dump(volume_data, f)

        if volume_id not in volume_ids:
            volume_ids.append(volume_id)
            self.write_volume_ids(subject_id, volume_ids)

        self.logger.info(f"Added volume with ID {volume_id} for subject {subject_id} to the database.")

    def choose_session(self, subject, options=None):
        # Implement the logic to choose a session
        pass

    def choose_subject(self, options=None):
        # Implement the logic to choose a subject
        pass

    def get_gridweight_hashes(self, transducer_id):
        transducer_dir = os.path.join(self.path, 'transducers', transducer_id)
        gridfiles = glob.glob(os.path.join(transducer_dir, f'{transducer_id}_gridweights_*.h5'))
        return [os.path.splitext(os.path.basename(f))[0].split('_')[-1] for f in gridfiles]

    def get_session_table(self, subject_id, options=None):
        # Implement the logic to get session table
        pass

    def get_subject_table(self, options=None):
        # Implement the logic to get subject table
        pass

    def load_session_solutions(self, session, options=None):
        # Implement the logic to load session solutions
        pass

    def get_connected_systems(self):
        connected_system_filename = self.get_connected_system_filename()

        if os.path.isfile(connected_system_filename):
            with open(connected_system_filename, "r") as file:
                connected_systems = file.read().strip().split(',')
            self.logger.info("Connected systems: %s", connected_systems)
            return connected_systems
        else:
            self.logger.warning("Connected systems file not found.")
            return []

    def get_connected_transducer(self, options=None):
        connected_transducer_filename = self.get_connected_transducer_filename()

        if os.path.isfile(connected_transducer_filename):
            with open(connected_transducer_filename, "r") as file:
                connected_transducer_id = file.read().strip()
            self.logger.info("Connected transducer: %s", connected_transducer_id)
            return connected_transducer_id
        else:
            self.logger.warning("Connected transducer file not found.")
            return None

    def get_protocol_ids(self):
        protocols_filename = self.get_protocols_filename()

        if os.path.isfile(protocols_filename):
            with open(protocols_filename, "r") as file:
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
            with open(sessions_filename, "r") as file:
                session_data = json.load(file)
                session_ids = session_data.get("session_ids", [])
            self.logger.info("Session IDs for subject %s: %s", subject_id, session_ids)
            return session_ids
        else:
            self.logger.warning("Sessions file not found for subject %s.", subject_id)
            return []

    def get_solutions(self, subject_id, session_id):
        solutions_filename = self.get_solutions_filename(subject_id, session_id)

        if os.path.isfile(solutions_filename):
            with open(solutions_filename, "r") as file:
                solution_data = json.load(file)
            solutions = [
                Solution.from_dict(solution_dict) for solution_dict in solution_data
            ]
            self.logger.info("Solutions for subject %s, session %s: %s", subject_id, session_id, solutions)
            return solutions
        else:
            self.logger.warning("Solutions file not found for subject %s, session %s.", subject_id, session_id)
            return []

    def get_subject_ids(self):
        subjects_filename = self.get_subjects_filename()

        if os.path.isfile(subjects_filename):
            with open(subjects_filename, "r") as file:
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
            with open(systems_filename, "r") as file:
                system_data = json.load(file)
                system_ids = system_data.get("system_ids", [])
            self.logger.info("System IDs: %s", system_ids)
            return system_ids
        else:
            self.logger.warning("Systems file not found.")
            return []

    def get_system_info(self, sys_id):
        system_filename = self.get_system_filename(sys_id)

        if os.path.isfile(system_filename):
            with open(system_filename, "r") as file:
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
            with open(transducers_filename, "r") as file:
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

    def load_volume(self, subject, volume_id):
        volume_filename = self.get_volume_filename(subject.id, volume_id)
        volume = Volume.from_file(volume_filename)
        return volume

    def load_volume_attrs(self, subject, volume_ids=None):
        volume_ids = volume_ids or subject.volumes
        attrs = []
        for volume_id in volume_ids:
            volume_filename = self.get_volume_filename(subject.id, volume_id)
            volume = Volume.from_file(volume_filename)
            attrs.append(volume.attrs)
        return attrs

    def load_standoff(self, transducer_id, standoff_id="standoff"):
        standoff_filename = self.get_standoff_filename(transducer_id, standoff_id)
        standoff = Standoff.from_file(standoff_filename)
        return standoff

    def load_system(self, sys_id=None):
        sys_id = sys_id or self.get_connected_systems()
        sys_filename = self.get_system_filename(sys_id)
        sys = UltrasoundSystem.from_file(sys_filename)
        return sys

    def load_transducer(self, transducer_id):
        from openlifu.xdc import Transducer
        transducer_filename = self.get_transducer_filename(transducer_id)
        transducer = Transducer.from_file(transducer_filename)
        return transducer

    def load_transducer_standoff(self, trans, coords, options=None):
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
            with open(protocols_filename, 'r') as file:
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
            with open(session_filename, 'r') as file:
                session_data = json.load(file)
            self.logger.info(f"Loaded session info for session {session_id} of subject {subject_id}")
            return session_data
        else:
            self.logger.error(f"Session file not found for ID: {session_id}")
            raise FileNotFoundError(f"Session file not found for ID: {session_id}")


    def load_solution(self, session, protocol_id, target_id):
        solution_filename_json = self.get_solution_filename(session.subject_id, session.id, protocol_id, target_id, ext="json")
        solution_filename_mat = self.get_solution_filename(session.subject_id, session.id, protocol_id, target_id, ext="mat")

        if os.path.isfile(solution_filename_json) and os.path.isfile(solution_filename_mat):
            # Load from JSON file
            with open(solution_filename_json, "r") as json_file:
                json_data = json.load(json_file)

            # Load from MAT file
            mat_data = scipy.io.loadmat(solution_filename_mat)
            # Extract any additional data from the MAT file as needed

            # Pass the loaded JSON data as keyword arguments to TreatmentSolution constructor
            solution = Solution(**json_data)
            self.logger.info(f"Loaded solution for Protocol {protocol_id}, Target {target_id}")
            return solution
        else:
            self.logger.error(f"Solution files not found for Protocol {protocol_id}, Target {target_id}")
            raise FileNotFoundError(f"Solution files not found for Protocol {protocol_id}, Target {target_id}")

    def set_connected_transducer(self, trans, options=None):
        trans_id = trans.id
        transducer_ids = self.get_transducer_ids()
        if trans_id not in transducer_ids:
            if not options or not options.add_if_missing:
                self.logger.error(f"Invalid Transducer ID {trans_id}. Valid IDs are {', '.join(transducer_ids)}")
                raise ValueError(f"Invalid Transducer ID {trans_id}")
            else:
                self.add_transducer(trans)
        filename = self.get_connected_transducer_filename()
        with open(filename, 'w') as f:
            f.write(trans_id)

    def get_connected_system_filename(self):
        return os.path.join(self.path, "systems", "connected_system.txt")

    def get_connected_transducer_filename(self):
        return os.path.join(self.path, 'transducers', 'connected_transducer.txt')

    def get_gridweights_filename(self, transducer_id, grid_hash):
        return os.path.join(self.path, 'transducers', transducer_id, f'{transducer_id}_gridweights_{grid_hash}.h5')

    def get_protocols_filename(self):
        return os.path.join(self.path, 'protocols', 'protocols.json')

    def get_protocol_filename(self, protocol_id):
        return os.path.join(self.path, 'protocols', protocol_id, f'{protocol_id}.json')

    def get_session_dir(self, subject_id, session_id):
        return os.path.join(self.get_subject_dir(subject_id), 'sessions', session_id)

    def get_session_filename(self, subject_id, session_id):
        return os.path.join(self.get_session_dir(subject_id, session_id), f'{session_id}.json')

    def get_sessions_filename(self, subject_id):
        return os.path.join(self.get_subject_dir(subject_id), 'sessions', 'sessions.json')

    def get_solution_filename(self, subject_id, session_id, protocol_id, target_id, ext='mat'):
        session_dir = self.get_session_dir(subject_id, session_id)
        return os.path.join(session_dir, 'solutions', protocol_id, f'{target_id}.{ext}')

    def get_solutions_filename(self, subject_id, session_id):
        session_dir = self.get_session_dir(subject_id, session_id)
        return os.path.join(session_dir, 'solutions', 'solutions.json')

    def get_standoff_filename(self, transducer_id, standoff_id='standoff'):
        return os.path.join(self.path, 'transducers', transducer_id, f'{standoff_id}.json')

    def get_subject_dir(self, subject_id):
        return os.path.join(self.path, 'subjects', subject_id)

    def get_subject_filename(self, subject_id):
        return os.path.join(self.get_subject_dir(subject_id), f'{subject_id}.json')

    def get_subjects_filename(self):
        return os.path.join(self.path, 'subjects', 'subjects.json')

    def get_systems_filename(self):
        return os.path.join(self.path, 'systems', 'systems.json')

    def get_system_filename(self, system_id):
        return os.path.join(self.path, 'systems', system_id, f'{system_id}.json')

    def get_transducer_filename(self, transducer_id):
        return os.path.join(self.path, 'transducers', transducer_id, f'{transducer_id}.json')

    def get_transducers_filename(self):
        return os.path.join(self.path, 'transducers', 'transducers.json')

    def get_volume_filename(self, subject_id, volume_id):
        subject_dir = self.get_subject_dir(subject_id)
        return os.path.join(subject_dir, 'volumes', f'{volume_id}.mat')

    def write_protocol_ids(self, protocol_ids):
        protocol_data = {'protocol_ids': protocol_ids}
        protocols_filename = self.get_protocols_filename()
        with open(protocols_filename, 'w') as f:
            json.dump(protocol_data, f)

    def write_session_ids(self, subject_id, session_ids):
        session_data = {'session_ids': session_ids}
        sessions_filename = self.get_sessions_filename(subject_id)
        with open(sessions_filename, 'w') as f:
            json.dump(session_data, f)

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

    @staticmethod
    def get_default_user_dir():
        """
        Get the default user directory for the database
        
        :returns: Default user directory
        """
        return os.path.expanduser("~")

    @staticmethod
    def get_default_path(options=None):
        """
        Get the default path for the database
        
        :returns: Default path for the database
        """
        return os.path.join(Database.get_default_user_dir(), "Documents", "db")
