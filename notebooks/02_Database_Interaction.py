# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# # 02: Database Interaction and Transducer Loading
#
# In OpenLIFU, transducer definitions and often pre-defined protocols are stored in a database. This notebook explains how to connect to this database and load `Transducer` objects, which are essential for planning and executing sonications with specific hardware.

# ## Connecting to the Database
#
# The `Database` class from `openlifu.db` is used to interact with the OpenLIFU database.
# By default, it looks for a database in a standard location relative to the library installation or a path specified by an environment variable. For typical use, you might need to specify the path to your database directory.
#
# The OpenLIFU repository usually includes a `db_dvc/` directory which serves as an example or default database. We will assume such a structure for this notebook.

# +
import os
from pathlib import Path
from openlifu.db import Database

# Attempt to find the database directory
# This might need adjustment based on your project structure or how OpenLIFU is installed.
# Typically, if you are running this notebook from within a cloned 'openlifu' repository,
# the 'db_dvc' might be in the root.
db_path_found = None
paths_to_check = [
    Path.cwd() / "db_dvc",          # If running from root of a project with db_dvc
    Path.cwd() / ".." / "db_dvc",   # If notebooks are in a subdir of such a project
    Path.cwd() / ".." / ".." / "db_dvc", # If notebooks are two levels deep
    # Add other common paths if necessary
]

for path_check in paths_to_check:
    if path_check.exists() and path_check.is_dir():
        # Check for a typical file/folder inside db_dvc to be more certain
        if (path_check / "transducers").exists() or (path_check / "protocols").exists():
            db_path_found = path_check.resolve()
            break

if db_path_found:
    print(f"Found database directory at: {db_path_found}")
    db = Database(db_path_found)
else:
    # Fallback: try default instantiation, which might work if OpenLIFU is installed
    # and knows its database path, or raise an error.
    try:
        db = Database()
        print("Initialized Database with default path.")
        # Check if it actually found something meaningful
        if not db.list_transducers():
             print("Warning: Default database seems empty or not found. Transducer loading might fail.")
    except Exception as e:
        print(f"Could not find 'db_dvc' directory automatically, and default Database() failed: {e}")
        print("Please ensure 'db_dvc' is accessible or specify its path directly to Database().")
        db = None # Explicitly set to None if not loaded

# -

# ## Listing Available Transducers
#
# Once the database connection is established, you can list the available transducers. Transducers are typically identified by a unique ID.

if db:
    available_transducers = db.list_transducers()
    if available_transducers:
        print("Available transducers in the database:")
        for trans_id in available_transducers:
            print(f"- {trans_id}")
    else:
        print("No transducers found in the database. Please check the database path and contents.")
else:
    print("Database not loaded. Cannot list transducers.")

# ## Loading a Transducer
#
# You can load a specific transducer by its ID using the `load_transducer()` method. This returns a `Transducer` object.

# +
if db and available_transducers:
    # Let's try to load one of the available transducers.
    # Replace 'openlifu_2x400_evt1' with an ID from your list if it's different.
    # Or, pick the first one from the list.
    transducer_id_to_load = available_transducers[0]
    print(f"\nAttempting to load transducer: '{transducer_id_to_load}'")

    try:
        my_transducer = db.load_transducer(transducer_id_to_load)
        print(f"\nSuccessfully loaded transducer: '{my_transducer.id}'")
        print(f"Name: {my_transducer.name}")
        print(f"Number of elements: {my_transducer.numelements()}")

        # Display some properties of the loaded transducer
        print(f"\nTransducer Type: {my_transducer.type}")
        print(f"Manufacturer: {my_transducer.manufacturer}")
        print(f"Center Frequency: {my_transducer.f0 / 1e3 if my_transducer.f0 else 'N/A'} kHz")

        # Get element positions (first 5 for brevity)
        positions = my_transducer.get_positions(units="mm")
        print(f"\nFirst 5 element positions (mm):\n{positions[:5]}")

        # Transducer elements often have pin mappings for hardware control
        # The `sort_by_pin()` method can be important for hardware interaction.
        # It sorts the elements based on their pin numbers.
        # Let's check if pin numbers are defined for the first few elements
        print("\nElement pin numbers (first 5, before sorting):")
        for i, el in enumerate(my_transducer.elements[:5]):
            print(f"  Element {el.id}: Pin {el.pin if el.pin is not None else 'N/A'}")

        # If you plan to use this transducer with hardware, sorting by pin might be necessary
        # depending on how solutions are applied.
        # my_transducer.sort_by_pin()
        # print("\nElement pin numbers (first 5, after attempting sort_by_pin):")
        # for i, el in enumerate(my_transducer.elements[:5]):
        #     print(f"  Element {el.id}: Pin {el.pin if el.pin is not None else 'N/A'}")
        # Note: The effect of sort_by_pin is on the order of elements in my_transducer.elements

    except Exception as e:
        print(f"Error loading transducer '{transducer_id_to_load}': {e}")
        my_transducer = None
else:
    print("Database not loaded or no transducers available. Cannot load transducer.")
    my_transducer = None
# -

# ## Transducer Properties
#
# A `Transducer` object contains detailed information:
# *   `id`: Unique identifier.
# *   `name`: Descriptive name.
# *   `elements`: A list of `TransducerElement` objects, each with properties like position, normal vector, dimensions, pin mapping, etc.
# *   `numelements()`: Returns the number of active elements.
# *   `get_positions()`: Returns a NumPy array of element center positions.
# *   `get_normals()`: Returns a NumPy array of element normal vectors.
# *   And many more specific to the transducer type (e.g., `pitch`, `radius_of_curvature` for curved arrays).

if my_transducer:
    print(f"\nFurther properties of '{my_transducer.id}':")
    print(f"  Focus Point: {my_transducer.focus_pt}")
    print(f"  Element Type: {type(my_transducer.elements[0]).__name__ if my_transducer.elements else 'N/A'}")

    # Example: Accessing properties of the first element
    if my_transducer.elements:
        first_element = my_transducer.elements[0]
        print("\nProperties of the first element:")
        print(f"  ID: {first_element.id}")
        print(f"  Position: {first_element.get_position(units='mm')} mm")
        print(f"  Normal: {first_element.get_normal()}")
        # Depending on element type, other properties like width, height, area might exist
        if hasattr(first_element, 'width'):
            print(f"  Width: {first_element.width} m")
        if hasattr(first_element, 'height'):
            print(f"  Height: {first_element.height} m")


# ## Listing and Loading Protocols
#
# The database can also store `Protocol` definitions.

# +
if db:
    available_protocols = db.list_protocols()
    if available_protocols:
        print("\nAvailable protocols in the database:")
        for proto_id in available_protocols:
            print(f"- {proto_id}")

        # Load a protocol
        protocol_id_to_load = available_protocols[0] # Pick the first one
        print(f"\nAttempting to load protocol: '{protocol_id_to_load}'")
        try:
            my_protocol = db.load_protocol(protocol_id_to_load)
            print(f"\nSuccessfully loaded protocol: '{my_protocol.id}'")
            print(f"Name: {my_protocol.name}")
            print(f"Pulse: {my_protocol.pulse}")
            print(f"Sequence: {my_protocol.sequence}")
            print(f"Focal Pattern Type: {my_protocol.focal_pattern_type}")
            if my_protocol.focal_pattern: # If a specific instance is stored
                 print(f"Focal Pattern Instance: {my_protocol.focal_pattern}")
            print(f"Apodization Method: {my_protocol.apod_method}")
            print(f"Simulation Setup: {my_protocol.sim_setup}")

        except Exception as e:
            print(f"Error loading protocol '{protocol_id_to_load}': {e}")
            my_protocol = None
    else:
        print("\nNo protocols found in the database.")
else:
    print("Database not loaded. Cannot list or load protocols.")
# -

# ## Next Steps
#
# With a `Transducer` object loaded from the database and the core objects from Notebook 01 (like `Pulse`, `Sequence`, `Target`, `Protocol`), you are now ready to generate a `Solution` for sonication.
#
# The next notebook, `03_Solution_Generation_and_Analysis.py`, will cover this process, including how to run acoustic simulations.

# End of Notebook 02
