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

# # 03: Solution Generation and Analysis
#
# This notebook demonstrates how to generate an acoustic `Solution` using a `Protocol`, `Target`, and `Transducer`. It also covers running acoustic simulations and analyzing the results, including checking against safety parameters.
#
# We will build upon the concepts from Notebook 01 (Object Creation) and Notebook 02 (Database Interaction/Transducer Loading).

# ## 1. Setup and Imports
#
# First, let's import the necessary classes.

# +
import numpy as np
from pathlib import Path

# Core OpenLIFU objects
from openlifu.bf import Pulse, Sequence, apod_methods, focal_patterns
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.plan.param_constraint import ParameterConstraint
from openlifu.sim import SimSetup, SimResult
from openlifu.xdc import Transducer
from openlifu.db import Database
from openlifu.plan.analysis import SolutionAnalysis

# For displaying tables
import pandas as pd
# -

# ## 2. Defining Components for Solution Calculation
#
# To calculate a solution, we need:
# 1.  A `Target`: Where we want to focus the ultrasound.
# 2.  A `Transducer`: The physical device that will generate the ultrasound.
# 3.  A `Protocol`: The "recipe" defining pulse characteristics, sequence, focal pattern, etc.

# ### 2.1. Define a Target
# This is a `Point` object representing the desired focal location.

target = Point(position=np.array([0, 0, 50]), units="mm", radius=0.5) # 50mm depth, small radius
print(f"Target: {target}")

# ### 2.2. Load or Define a Transducer
# For this example, we'll first try to load a transducer from the database, as shown in Notebook 02.
# If that fails (e.g., database not found), we'll fall back to a programmatically generated generic transducer for demonstration purposes.

# +
# Attempt to load from database (similar to Notebook 02)
transducer = None
db_path_found = None
paths_to_check = [Path.cwd() / "db_dvc", Path.cwd() / ".." / "db_dvc"]
for path_check in paths_to_check:
    if path_check.exists() and path_check.is_dir() and (path_check / "transducers").exists():
        db_path_found = path_check.resolve()
        break

if db_path_found:
    db = Database(db_path_found)
    available_transducers = db.list_transducers()
    if available_transducers:
        # Load a common transducer if available, otherwise the first one
        trans_id_to_load = 'openlifu_2x400_evt1' if 'openlifu_2x400_evt1' in available_transducers else available_transducers[0]
        try:
            transducer = db.load_transducer(trans_id_to_load)
            print(f"Successfully loaded transducer '{transducer.id}' from database.")
        except Exception as e:
            print(f"Error loading transducer '{trans_id_to_load}' from database: {e}")
    else:
        print("No transducers found in the database via path:", db_path_found)
else:
    print("Database directory 'db_dvc' not found in typical locations.")

# Fallback to a generic transducer if not loaded from DB
if transducer is None:
    print("Using a programmatically generated generic transducer for this example.")
    transducer = Transducer.gen_matrix_array(
        nx=8, ny=8, pitch=4, kerf=0.5,
        id="generic_8x8", name="Generic 8x8 Array",
        f0=400e3 # Assign a center frequency
    )
    # For generic transducers, ensure elements have an 'id' for some internal processes
    for i, el in enumerate(transducer.elements):
        el.id = i

print(f"Using Transducer: {transducer.id if transducer else 'None'}, Number of elements: {transducer.numelements() if transducer else 'N/A'}")
# -

# ### 2.3. Define a Protocol
# The protocol specifies *how* the sonication should be performed.

# +
# Pulse definition
f0 = transducer.f0 if transducer and transducer.f0 else 400e3 # Use transducer f0 if available
pulse = Pulse(frequency=f0, duration=10/f0) # 10 cycles duration

# Sequence definition
sequence = Sequence(pulse_interval=0.1, pulse_count=9, pulse_train_interval=0, pulse_train_count=1)

# Focal Pattern: Let's use a SinglePoint focus for this example.
# The actual target point is provided during calc_solution.
# target_pressure is an optional parameter for scaling.
focal_pattern = focal_patterns.SinglePoint(target_pressure=1.0e6) # Target 1 MPa

# Apodization Method
apod_method = apod_methods.MaxAngle(max_angle_deg=30) # Limit elements to a 30-degree cone

# Simulation Setup: Defines the grid for acoustic simulation
sim_setup = SimSetup(
    x_extent=(-25, 25), y_extent=(-25, 25), z_extent=(-5, 70), # in mm
    resolution=1.0 # 1 mm resolution
)

# Create the Protocol object
protocol1 = Protocol(
    id='example_protocol_prog',
    name='Example Protocol (Programmatic)',
    pulse=pulse,
    sequence=sequence,
    focal_pattern_type=focal_patterns.SinglePoint, # Store the type
    # focal_pattern=focal_pattern # Or store an instance
    apod_method=apod_method,
    sim_setup=sim_setup,
    default_target_pressure=1.0e6 # Can also be set here
)
print(f"Defined Protocol: {protocol1.name}")
# -

# ## 3. Calculating the Solution
#
# With the `target`, `transducer`, and `protocol` defined, we can now calculate the `Solution`.
# The `calc_solution` method returns:
# *   `solution`: The `Solution` object containing delays, apodizations, voltage, etc.
# *   `sim_res`: A `SimResult` object if `simulate=True`.
# *   `analysis`: A `SolutionAnalysis` object if `simulate=True` and `scale=False` (or `analysis` if `scale=True`).
#
# Setting `simulate=True` will run an acoustic simulation.
# Setting `scale=True` will attempt to scale the output pressure to match `target_pressure` defined in the focal pattern or protocol, and returns a `scaled_analysis`.

# +
if transducer:
    print(f"\nCalculating solution for protocol '{protocol1.name}' and target '{target.position_str}'...")
    solution1, sim_res1, analysis1 = protocol1.calc_solution(
        target=target,
        transducer=transducer,
        simulate=True,
        scale=True # Try to scale to target_pressure
    )

    if solution1:
        print(f"\nSolution calculated: {solution1.id}")
        print(f"  Calculated Voltage: {solution1.voltage:.2f} V (this is a relative/normalized value before hardware calibration)")
        # print(f"  Delays (first 5 elements): {solution1.delays[0, :5]}")
        # print(f"  Apodizations (first 5 elements): {solution1.apodizations[0, :5]}")

    if sim_res1:
        print(f"\nSimulation Result object created.")
        # sim_res1 contains the raw simulation grid and pressure data.
        # For example, to get the peak pressure and its location:
        # peak_pressure_Pa, peak_loc_mm = sim_res1.get_peak_pressure()
        # print(f"  Peak pressure in simulation: {peak_pressure_Pa/1e6:.2f} MPa at {peak_loc_mm} mm")
        # sim_res1.plot_slices() # This would require matplotlib and a GUI backend

    if analysis1:
        print(f"\nSolutionAnalysis object created (scaled):")
        # The SolutionAnalysis object provides various calculated acoustic parameters.
        # We can display it as a table:
        analysis_table = analysis1.to_table()
        print(analysis_table.set_index('Param')[['Value', 'Units', 'Status']])
else:
    print("Transducer not available. Cannot calculate solution.")
    solution1, sim_res1, analysis1 = None, None, None
# -

# ## 4. Using Parameter Constraints in Analysis
#
# We can define constraints for various parameters (like MI, TIC, Isppa) and see if the solution meets them.

# +
if analysis1:
    # Define some example parameter constraints
    constraints = {
        "MI": ParameterConstraint('<', 1.8, 1.85), # Mechanical Index should be < 1.8 (error if > 1.85)
        "TIC": ParameterConstraint('<', 2.0),       # Thermal Index (cranial) should be < 2.0
        "global_Isppa_Wcm2": ParameterConstraint('within', error_value=(50, 200)) # Isppa between 50-200 W/cm^2
    }

    print("\nAnalysis table with constraints:")
    # The to_table method can accept these constraints directly
    constrained_table = analysis1.to_table(constraints=constraints)
    print(constrained_table.set_index('Param')[['Value', 'Units', 'Status', 'Message']])

    # Check overall status
    print(f"\nOverall status of constraints: {analysis1.get_overall_status(constraints)}")
    for msg in analysis1.get_messages(constraints):
        print(f"  - {msg}")

else:
    print("\nNo analysis object available to apply constraints.")
# -

# ## 5. Example: Loading a Protocol from File/Database
#
# Protocols can also be loaded from the database (see Notebook 02) or from JSON files.
# Here, we attempt to load a protocol from a known JSON file path (if it exists from the repository structure).

# +
# Path to the example protocol JSON file relative to a potential repo root
# This path might need adjustment if your notebook isn't in 'notebooks' dir of the main repo
protocol_file_path = Path.cwd() / ".." / "tests" / "resources" / "example_db" / "protocols" / "example_protocol" / "example_protocol.json"

loaded_protocol = None
if protocol_file_path.exists():
    print(f"\nAttempting to load protocol from: {protocol_file_path}")
    try:
        loaded_protocol = Protocol.from_file(protocol_file_path)
        print(f"Successfully loaded protocol: {loaded_protocol.id} - {loaded_protocol.name}")

        # Now, let's try to calculate a solution with this loaded protocol
        if transducer and target and loaded_protocol:
            print(f"\nCalculating solution for loaded protocol '{loaded_protocol.name}'...")
            solution2, sim_res2, analysis2 = loaded_protocol.calc_solution(
                target=target,
                transducer=transducer,
                simulate=True,
                scale=True # Or False, depending on how the protocol is defined
            )

            if solution2:
                print(f"Solution calculated: {solution2.id}")
            if analysis2:
                print("\nAnalysis of solution from loaded protocol:")
                print(analysis2.to_table().set_index('Param')[['Value', 'Units', 'Status']])

    except Exception as e:
        print(f"Failed to load or use protocol from {protocol_file_path}: {e}")
else:
    print(f"\nExample protocol file not found at: {protocol_file_path}")
    print("Skipping example of loading protocol from file.")
# -

# ## Summary and Next Steps
#
# This notebook showed how to:
# *   Define or load the necessary components (`Target`, `Transducer`, `Protocol`).
# *   Calculate a `Solution` using `protocol.calc_solution()`.
# *   Enable acoustic simulation and obtain `SimResult` and `SolutionAnalysis` objects.
# *   Use `ParameterConstraint` to evaluate the solution against safety or performance criteria.
#
# The `Solution` object is key for hardware interaction. The next notebook, `04_Connecting_to_Hardware.py`, will introduce how to establish communication with OpenLIFU hardware. Following that, `05_Solution_to_Hardware_Basic.py` will demonstrate sending a calculated solution to the device.

# End of Notebook 03
