# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: env
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
from __future__ import annotations

import numpy as np

# For displaying tables
# Core OpenLIFU objects
from openlifu.bf import Pulse, Sequence, apod_methods, focal_patterns
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.plan.param_constraint import ParameterConstraint
from openlifu.sim import SimSetup
from openlifu.xdc import Transducer

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
transducer = Transducer.gen_matrix_array(
    nx=16, ny=16, pitch=3, kerf=0.1,
    id="generic_16x16", name="Generic 16x16 Array"
)

print(f"Using Transducer: {transducer.id if transducer else 'None'}, Number of elements: {transducer.numelements() if transducer else 'N/A'}")
# -

# ### 2.3. Define a Protocol
# The protocol specifies *how* the sonication should be performed.

# +
# Pulse definition
f0 = 400e3 # Use transducer f0 if available
pulse = Pulse(frequency=f0, duration=10/f0) # 10 cycles duration

# Sequence definition
sequence = Sequence(pulse_interval=0.1, pulse_count=9, pulse_train_interval=0, pulse_train_count=1)

# Focal Pattern: Let's use a SinglePoint focus for this example.
# The actual target point is provided during calc_solution.
# target_pressure is an optional parameter for scaling.
focal_pattern = focal_patterns.SinglePoint(target_pressure=1.0e6) # Target 1 MPa

# Apodization Method
apod_method = apod_methods.MaxAngle(max_angle=30) # Limit elements to a 30-degree cone

# Simulation Setup: Defines the grid for acoustic simulation
sim_setup = SimSetup(
    x_extent=(-25, 25), y_extent=(-25, 25), z_extent=(-5, 70), # in mm
    spacing=1.0 # 1 mm resolution
)

# Create the Protocol object
protocol1 = Protocol(
    id='example_protocol_prog',
    name='Example Protocol (Programmatic)',
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern, # Store the type
    apod_method=apod_method,
    sim_setup=sim_setup
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

if transducer:
    print(f"\nCalculating solution for protocol '{protocol1.name}' and target '{target.name}'...")
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
        print("\nSimulation Result object created.")
        # sim_res1 contains the raw simulation grid and pressure data.
        # For example, to get the peak pressure and its location:
        # peak_pressure_Pa, peak_loc_mm = sim_res1.get_peak_pressure()
        # print(f"  Peak pressure in simulation: {peak_pressure_Pa/1e6:.2f} MPa at {peak_loc_mm} mm")
        # sim_res1.plot_slices() # This would require matplotlib and a GUI backend

    if analysis1:
        print("\nSolutionAnalysis object created (scaled):")
        # The SolutionAnalysis object provides various calculated acoustic parameters.
        # We can display it as a table:
        analysis_table = analysis1.to_table()
        print(analysis_table.set_index('Param')[['Value', 'Units', 'Status']])
else:
    print("Transducer not available. Cannot calculate solution.")
    solution1, sim_res1, analysis1 = None, None, None

# Now we can plot the results

# + vscode={"languageId": "javascript"}
solution1.simulation_result['p_min'].sel(focal_point_index=0).sel(ele=0).plot.imshow()
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
        "global_isppa_Wcm2": ParameterConstraint('within', error_value=(50, 200)) # Isppa between 50-200 W/cm^2
    }

    print("\nAnalysis table with constraints:")
    # The to_table method can accept these constraints directly
    constrained_table = analysis1.to_table(constraints=constraints)
    print(constrained_table.set_index('Param')[['Value', 'Units', 'Status']])

else:
    print("\nNo analysis object available to apply constraints.")
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
