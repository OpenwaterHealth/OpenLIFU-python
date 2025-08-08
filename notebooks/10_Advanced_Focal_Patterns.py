# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: env (3.11.4)
#     language: python
#     name: python3
# ---

# # 10: Advanced Focal Patterns
#
# OpenLIFU allows for various focal patterns beyond a simple single point focus. These patterns can be used to create more complex sonication geometries, such as rings, grids, or other custom arrangements of focal spots. This notebook explores some of the predefined advanced focal patterns and how to use them within a `Protocol`.
#
# Understanding these patterns is key to designing sophisticated ultrasound applications, like creating specific lesion shapes or treating larger areas by distributing focal points.

# ## 1. Imports
# We'll need classes for focal patterns, protocols, targets, transducers, etc.

# +
from __future__ import annotations

import numpy as np

# OpenLIFU core components
from openlifu.bf import Pulse, Sequence, apod_methods, focal_patterns
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.sim import SimSetup
from openlifu.xdc import Transducer

# For potential visualization (though this notebook focuses on definition)
# import matplotlib.pyplot as plt
# -

# ## 2. Setup: Transducer and Basic Protocol Components
#
# To demonstrate focal patterns, we need a `Transducer` and some basic components for a `Protocol`, similar to Notebook 03.

# +
transducer = Transducer.gen_matrix_array(nx=8, ny=8, pitch=4, kerf=0.5, id="generic_8x8", name="Generic 8x8 Array")

# Basic pulse, sequence, apodization, sim_setup for protocols
f0 = 400e3
pulse_default = Pulse(frequency=f0, duration=10/f0)
sequence_default = Sequence(pulse_interval=0.1, pulse_count=5, pulse_train_count=1) # Short for example
apod_method_default = apod_methods.MaxAngle(30)
sim_setup_default = SimSetup(x_extent=(-10,10), y_extent=(-10,10), z_extent=(20,70), spacing=1.0) # Smaller extent for faster demo

# Define a main target point around which patterns will often be centered
main_target_point = Point(position=np.array([0, 0, 50]), units="mm") # 50mm depth

def format_position(point, fmt="{:0.1f}"):
    return ','.join([fmt.format(coord) for coord in point.position])

print(f"Using Transducer: {transducer.id}, Main Target: {format_position(main_target_point)}")
# -

# ## 3. Recap: `SinglePoint` Focal Pattern
#
# This is the most basic pattern, focusing energy at a single specified target point.
# When a `Protocol` uses `focal_pattern_type=focal_patterns.SinglePoint` (or an instance of `SinglePoint`), the `target` provided to `calc_solution` becomes this single focus.

# +
fp_single = focal_patterns.SinglePoint(target_pressure=1.0e6) # Optional target_pressure for scaling
print(f"SinglePoint pattern instance: {fp_single}")

protocol_single = Protocol(
    id='proto_single_point', name='Protocol with Single Point Focus',
    pulse=pulse_default, sequence=sequence_default,
    focal_pattern=fp_single, # Assign instance
    # Or: focal_pattern_type=focal_patterns.SinglePoint, # Assign type
    apod_method=apod_method_default, sim_setup=sim_setup_default
)
print(f"Protocol created: {protocol_single.name}")

# When calculating solution:
# solution, sim_res, analysis = protocol_single.calc_solution(target=main_target_point, transducer=transducer, simulate=False)
# The solution would target main_target_point directly.
# if solution: print(f"  Solution foci (SinglePoint): {[f.position_str for f in solution.foci]}")
# -

# ## 4. `Wheel` Focal Pattern
#
# The `Wheel` pattern arranges focal points in a circular or wheel-spoke layout. This can be useful for creating annular lesions or treating a circular region.
#
# Key parameters:
# *   `spoke_radius` (float): The radius of the circle on which the points lie (in mm).
# *   `num_spokes` (int): The number of focal points to distribute around the circle.
# *   `center` (bool): If `True`, an additional focal point is placed at the center of the wheel. Default is `False`.
# *   `angle_offset_deg` (float): An angle in degrees to rotate the entire pattern. Default is 0.
# *   `target_pressure` (float, optional): Desired acoustic pressure at each point, used for scaling.

# +
fp_wheel = focal_patterns.Wheel(
    spoke_radius=5.0,  # 5 mm radius circle
    num_spokes=4,      # 4 points on the circle
    center=True,       # Include a point at the center
    target_pressure=1.2e6
)
print(f"Wheel pattern instance: {fp_wheel}")

protocol_wheel = Protocol(
    id='proto_wheel', name='Protocol with Wheel Focus',
    pulse=pulse_default, sequence=sequence_default,
    focal_pattern=fp_wheel,
    apod_method=apod_method_default, sim_setup=sim_setup_default
)
print(f"Protocol created: {protocol_wheel.name}")

# How calc_solution uses this:
# The `Wheel` pattern generates a list of `Point` objects (foci) relative to the
# `target` given to `calc_solution`.
# For example, if `calc_solution(target=main_target_point, ...)`:
# The wheel's center will be `main_target_point`, and its spokes will extend from there.
# The resulting `Solution` object will have multiple foci.

# Let's see the foci generated by this pattern for our main_target_point
# The pattern itself can generate these relative points:
relative_foci_wheel = fp_wheel.get_targets(main_target_point)
print(f"\nFoci generated by Wheel pattern relative to {format_position(main_target_point)}:")
for i, focus in enumerate(relative_foci_wheel):
    print(f"  Focus {i+1}: {format_position(focus)}, Radius: {focus.radius} mm")

# When calc_solution is run with this protocol and target, the solution.foci will store these.
# solution_w, _, _ = protocol_wheel.calc_solution(target=main_target_point, transducer=transducer, simulate=False, scale=False)
# if solution_w: print(f"\nSolution foci (Wheel): {[f.position_str for f in solution_w.foci if f]}")
# -

# ## 5. Using Focal Patterns in Solution Calculation
#
# When `protocol.calc_solution()` is called:
# 1.  The `focal_pattern` defined in the `Protocol` is used.
# 2.  The `target` argument passed to `calc_solution` (e.g., `main_target_point`) acts as the **center or reference point** for the pattern.
# 3.  The focal pattern generates a list of individual `Point` objects (foci).
# 4.  The `Solution` object then typically contains:
#     *   `delays` and `apodizations` calculated to steer/focus to these multiple points (often sequentially if the hardware doesn't support true multi-focusing, or by creating multiple sub-solutions/profiles). The exact behavior depends on the `Solution` generation strategy within OpenLIFU for multi-point patterns.
#     *   `solution.foci`: A list of these generated `Point` objects.
#
# If the hardware supports multiple profiles, a multi-point focal pattern might result in a solution that populates several hardware profiles, each targeting one of the generated foci. The `Sequence` object in the solution would then define how these profiles are triggered.

# +

solution_wheel_ex, sim_res_wheel_ex, analysis_wheel_ex = protocol_wheel.calc_solution(
    target=main_target_point,
    transducer=transducer,
    simulate=False, # Set to True for full simulation (slower)
    scale=False     # Set to True to scale to target_pressure (slower)
)

print(f"Solution '{solution_wheel_ex.id}' calculated.")
print(f"  Number of foci in this solution: {len(solution_wheel_ex.foci)}")
print("  Solution foci positions (first few):")
for i, focus_pt in enumerate(solution_wheel_ex.foci[:5]): # Print first 5
    if focus_pt: # Foci list might contain None if pattern generates variable points
        print(f"    Focus {i+1}: {format_position(focus_pt)}")

# The delays and apodizations in solution_wheel_ex might be for the first focus,
# or it might be a list if the solution object is structured for multi-profile solutions.
# This depends on OpenLIFU's internal handling of multi-point solutions.
# For instance, solution_wheel_ex.delays might have multiple rows if it's a multi-profile solution.
if solution_wheel_ex.delays.shape[0] > 1:
    print(f"  The solution contains {solution_wheel_ex.delays.shape[0]} sets of delays/apodizations (likely for multiple profiles).")
else:
    print("  The solution contains 1 set of delays/apodizations (likely for the first focus or a combined field).")

# -

# ## 6. Conclusion
#
# Advanced focal patterns provide powerful flexibility for designing ultrasound exposures. By combining patterns like `Wheel`, `Grid`, or custom-defined patterns with appropriate `Pulse` and `Sequence` settings, you can tailor sonications for a wide range of applications.
#
# The key is to understand how a chosen `FocalPattern` generates its list of foci relative to a main target point, and how this list is then translated into a `Solution` by `calc_solution`. This may involve creating solutions for multiple hardware profiles if sequential targeting of the pattern's points is desired.
#
# **Next Steps:**
# *   Explore `11_Advanced_Apodization.py` to learn about different apodization methods that can be used in conjunction with these focal patterns to shape the acoustic beam.
# *   Experiment with combining these patterns with different simulation settings to visualize their acoustic fields.

# End of Notebook 10
