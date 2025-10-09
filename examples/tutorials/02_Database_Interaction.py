# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02: Database Creation, Population, and Interaction
#
# In OpenLIFU, transducer definitions, treatment protocols, subject information, and session data are stored in a database. This notebook demonstrates how to create a new database from scratch, populate it with example data, and interact with the stored information. This approach is completely self-contained and doesn't require any external database files.

# %% [markdown]
# ## Overview
#
# In this tutorial, we will:
# 1. Create a new empty database at a specified location
# 2. Generate and store a sample transducer programmatically
# 3. Create and store a treatment protocol with various parameters
# 4. Add a subject to track patient/experimental data
# 5. Create a session that links subject, transducer, and protocol
# 6. Demonstrate how to query and load the various database items
# 7. Explore the database structure and files

# %% [markdown]
# ## Creating and Connecting to a Database
#
# The `Database` class from `openlifu.db` is used to interact with the OpenLIFU database.
# In this tutorial, we'll create a new database from scratch and populate it with example data.
# This approach is self-contained and doesn't require any external database files.

# %%
from __future__ import annotations

from pathlib import Path
import numpy as np

from openlifu.db import Database, Subject, Session
from openlifu import Transducer, Protocol, Pulse, Sequence, Point
from openlifu.bf import focal_patterns, apod_methods, delay_methods
from openlifu.sim import SimSetup
from openlifu.geo import ArrayTransform

# Create a new database at a default location
# You can easily change this path to wherever you want your database
database_path = Path.cwd() / "tutorial_database"

print(f"Creating new database at: {database_path}")
db = Database.initialize_empty_database(database_path)
print("Successfully created empty database!")


# %% [markdown]
# ## Creating and Adding a Transducer to the Database
#
# Now let's create a sample transducer programmatically and add it to our database. 
# We'll use the built-in matrix array generator to create a simple 8x8 element transducer.

# %%
# Create a matrix array transducer with 8x8 elements
example_transducer = Transducer.gen_matrix_array(
    nx=8,                    # 8 elements in x direction
    ny=8,                    # 8 elements in y direction  
    pitch=4e-3,              # 4mm pitch between elements
    kerf=0.5e-3,             # 0.5mm kerf (gap) between elements
    id="tutorial_transducer", # Unique identifier
    name="Tutorial 8x8 Matrix Array"  # Human-readable name
)

# Set some additional properties
example_transducer.frequency = 400e3  # 400 kHz center frequency
example_transducer.manufacturer = "Tutorial Examples Inc."

print(f"Created transducer: {example_transducer.name}")
print(f"ID: {example_transducer.id}")
print(f"Number of elements: {example_transducer.numelements()}")
print(f"Center frequency: {example_transducer.frequency / 1e3} kHz")

# Write the transducer to the database
db.write_transducer(example_transducer)
print(f"\nSuccessfully added transducer '{example_transducer.id}' to the database!")

# Verify it was added by listing available transducers
available_transducers = db.list_transducers()
print(f"\nTransducers now in database: {available_transducers}")

# %% [markdown]
# ## Loading a Transducer from the Database
#
# Now let's load the transducer we just created and examine its properties.

# %%
# Load the transducer we just created
loaded_transducer = db.load_transducer("tutorial_transducer")

print(f"Successfully loaded transducer: '{loaded_transducer.id}'")
print(f"Name: {loaded_transducer.name}")
print(f"Number of elements: {loaded_transducer.numelements()}")
print(f"Manufacturer: {loaded_transducer.manufacturer}")
print(f"Center Frequency: {loaded_transducer.frequency / 1e3} kHz")

# Get element positions (first 5 for brevity)
positions = loaded_transducer.get_positions(units="mm")
print(f"\nFirst 5 element positions (mm):")
for i in range(min(5, len(positions))):
    print(f"  Element {i}: [{positions[i][0]:.2f}, {positions[i][1]:.2f}, {positions[i][2]:.2f}]")

# Example: Accessing properties of individual elements
if loaded_transducer.elements:
    first_element = loaded_transducer.elements[0]
    print(f"\nProperties of the first element:")
    print(f"  ID: {first_element.id}")
    print(f"  Position: {first_element.get_position(units='mm')} mm")
    print(f"  Normal: {first_element.get_normal()}")
    if hasattr(first_element, 'width'):
        print(f"  Width: {first_element.width * 1000} mm")
    if hasattr(first_element, 'height'):
        print(f"  Height: {first_element.height * 1000} mm")

# %% [markdown]
# ## Transducer Properties
#
# A `Transducer` object contains detailed information:
# *   `id`: Unique identifier.
# *   `name`: Descriptive name.
# *   `elements`: A list of `TransducerElement` objects, each with properties like position, normal vector, dimensions, etc.
# *   `numelements()`: Returns the number of active elements.
# *   `get_positions()`: Returns a NumPy array of element center positions.
# *   `get_normals()`: Returns a NumPy array of element normal vectors.
# *   And many more specific to the transducer type (e.g., `pitch`, `frequency` for arrays).

# %%
print(f"\nAdditional properties of '{loaded_transducer.id}':")
print(f"  Focus Point: {loaded_transducer.focus_pt}")
print(f"  Element Type: {type(loaded_transducer.elements[0]).__name__ if loaded_transducer.elements else 'N/A'}")
print(f"  Units: {loaded_transducer.units}")

# Example: Get element normals
normals = loaded_transducer.get_normals()
print(f"\nFirst element normal vector: {normals[0]}")

# Show the range of positions to understand the array extent
print(f"\nArray extent in X: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f} mm")
print(f"Array extent in Y: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f} mm")
print(f"Array extent in Z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f} mm")


# %% [markdown]
# ## Creating and Adding a Protocol to the Database
#
# A `Protocol` defines the treatment parameters including pulse characteristics, sequence timing, 
# focal patterns, and simulation settings. Let's create a sample protocol and add it to our database.

# %%
# Create a pulse definition
pulse = Pulse(
    frequency=400e3,    # 400 kHz frequency
    duration=50e-6,     # 50 microsecond duration
    amplitude=1.0       # Normalized amplitude
)

# Create a sequence definition
sequence = Sequence(
    pulse_interval=0.1,        # 100ms between pulses
    pulse_count=10,            # 10 pulses per train
    pulse_train_interval=1.0,  # 1 second between trains
    pulse_train_count=3        # 3 trains total
)

# Create a focal pattern (single point focus)
focal_pattern = focal_patterns.SinglePoint(
    target_pressure=500e3,  # 500 kPa target pressure
    units="Pa"
)

# Create simulation setup
sim_setup = SimSetup(
    x_extent=(-30, 30),    # Simulation domain in mm
    y_extent=(-30, 30),
    z_extent=(0, 60),
    dx=1.0,                # 1mm resolution
    dt=2e-7,               # Time step for simulation
    t_end=100e-6           # Simulation duration
)

# Create the protocol
example_protocol = Protocol(
    id="tutorial_protocol",
    name="Tutorial Treatment Protocol",
    description="A sample protocol for tutorial purposes",
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    delay_method=delay_methods.GeometricFocus(),  # Focus delays
    apod_method=apod_methods.Uniform(),           # Uniform apodization
    sim_setup=sim_setup
)

print(f"Created protocol: {example_protocol.name}")
print(f"ID: {example_protocol.id}")
print(f"Pulse frequency: {example_protocol.pulse.frequency / 1e3} kHz")
print(f"Pulse duration: {example_protocol.pulse.duration * 1e6} μs")
print(f"Sequence: {example_protocol.sequence.pulse_count} pulses, {example_protocol.sequence.pulse_interval}s interval")

# Write the protocol to the database
db.write_protocol(example_protocol)
print(f"\nSuccessfully added protocol '{example_protocol.id}' to the database!")

# Verify it was added
available_protocols = db.list_protocols()
print(f"Protocols now in database: {available_protocols}")

# %% [markdown]
# ## Loading the Protocol from the Database
#
# Let's load the protocol we just created and examine its properties.

# %%
# Load the protocol we just created
loaded_protocol = db.load_protocol("tutorial_protocol")

print(f"Successfully loaded protocol: '{loaded_protocol.id}'")
print(f"Name: {loaded_protocol.name}")
print(f"Description: {loaded_protocol.description}")
print(f"Pulse frequency: {loaded_protocol.pulse.frequency / 1e3} kHz")
print(f"Pulse duration: {loaded_protocol.pulse.duration * 1e6} μs")
print(f"Target pressure: {loaded_protocol.focal_pattern.target_pressure / 1e3} kPa")
print(f"Simulation grid extent: X={loaded_protocol.sim_setup.x_extent}, Y={loaded_protocol.sim_setup.y_extent}, Z={loaded_protocol.sim_setup.z_extent}")

# %% [markdown]
# ## Creating and Adding a Subject to the Database
#
# A `Subject` represents an individual patient or experimental subject in the database.
# Let's create a sample subject and add it to our database.

# %%
# Create a sample subject
example_subject = Subject(
    id="tutorial_subject_001",
    name="Tutorial Subject",
    date_of_birth=None,  # Optional - can be set if needed
    sex="unspecified"    # Optional demographic information
)

print(f"Created subject: {example_subject.name}")
print(f"ID: {example_subject.id}")

# Write the subject to the database
db.write_subject(example_subject)
print(f"\nSuccessfully added subject '{example_subject.id}' to the database!")

# Verify it was added
available_subjects = db.list_subjects()
print(f"Subjects now in database: {available_subjects}")

# %% [markdown]
# ## Creating and Adding a Session to the Database
#
# A `Session` represents a treatment session that links a subject, transducer, and protocol together.
# It contains information about the experimental setup and any results.

# %%
from datetime import datetime

# Create a sample session
example_session = Session(
    id="tutorial_session_001",
    name="Tutorial Treatment Session",
    subject_id=example_subject.id,          # Link to our subject
    transducer_id=example_transducer.id,    # Link to our transducer
    protocol_id=example_protocol.id,        # Link to our protocol
    date_created=datetime.now(),
    # Array transform represents the coordinate transformation 
    # from transducer coordinates to volume coordinates
    array_transform=ArrayTransform(
        matrix=np.eye(4),  # Identity transform for this example
        units="mm"
    )
)

print(f"Created session: {example_session.name}")
print(f"ID: {example_session.id}")
print(f"Subject: {example_session.subject_id}")
print(f"Transducer: {example_session.transducer_id}")
print(f"Protocol: {example_session.protocol_id}")

# Write the session to the database
db.write_session(example_subject, example_session)
print(f"\nSuccessfully added session '{example_session.id}' to the database!")

# Verify it was added
available_sessions = db.list_sessions(example_subject.id)
print(f"Sessions for subject {example_subject.id}: {available_sessions}")

# %% [markdown]
# ## Querying and Loading Database Items
#
# Now that we have populated our database with various items, let's demonstrate how to query 
# and load them. This shows the typical workflow for accessing stored data.

# %%
print("=== DATABASE SUMMARY ===")
print(f"Database location: {database_path}")
print(f"Available transducers: {db.list_transducers()}")
print(f"Available protocols: {db.list_protocols()}")  
print(f"Available subjects: {db.list_subjects()}")

# Load a specific subject and show its sessions
subject = db.load_subject("tutorial_subject_001")
print(f"\nLoaded subject: {subject.name}")
sessions = db.list_sessions(subject.id)
print(f"Sessions for this subject: {sessions}")

# Load a specific session and examine its properties
if sessions:
    session = db.load_session(subject, sessions[0])
    print(f"\nLoaded session: {session.name}")
    print(f"  Subject: {session.subject_id}")
    print(f"  Transducer: {session.transducer_id}")
    print(f"  Protocol: {session.protocol_id}")
    print(f"  Created: {session.date_created}")
    
    # Load the associated objects
    session_transducer = db.load_transducer(session.transducer_id)
    session_protocol = db.load_protocol(session.protocol_id)
    
    print(f"\nAssociated transducer: {session_transducer.name} ({session_transducer.numelements()} elements)")
    print(f"Associated protocol: {session_protocol.name}")
    print(f"  Pulse frequency: {session_protocol.pulse.frequency / 1e3} kHz")
    print(f"  Treatment duration: {session_protocol.sequence.pulse_count * session_protocol.sequence.pulse_interval} seconds")

# %% [markdown]
# ## Database Structure and Files
#
# Let's explore what was actually created in our database directory.

# %%
import os

print("=== DATABASE DIRECTORY STRUCTURE ===")
for root, dirs, files in os.walk(database_path):
    level = root.replace(str(database_path), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

# %% [markdown]
# ## Summary
#
# In this tutorial, we have:
# 
# 1. **Created a new database** from scratch using `Database.initialize_empty_database()`
# 2. **Generated a transducer** programmatically using `Transducer.gen_matrix_array()` and stored it
# 3. **Created a protocol** with pulse, sequence, and simulation parameters and stored it
# 4. **Added a subject** to represent a patient or experimental subject
# 5. **Created a session** that links the subject, transducer, and protocol together
# 6. **Queried and loaded** all the items to demonstrate database interaction
#
# This self-contained approach allows you to:
# - Start working with OpenLIFU immediately without external database dependencies
# - Understand the structure and relationships between database objects
# - Customize the database location and content for your specific needs
# - Build up complex experimental datasets incrementally
#
# ## Next Steps
#
# With a populated database containing `Transducer`, `Protocol`, `Subject`, and `Session` objects, you are now ready to generate `Solution` objects for sonication planning and simulation.
#
# The next notebook, `03_Solution_Generation_and_Analysis.py`, will cover:
# - Using the database objects to calculate beamforming solutions
# - Running acoustic simulations to predict pressure fields
# - Analyzing and optimizing treatment parameters
# - Saving solutions back to the database for future use

# %% [markdown]
# End of Notebook 02
