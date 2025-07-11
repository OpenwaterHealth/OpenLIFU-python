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

# # 11: Advanced Apodization Methods
#
# Apodization in ultrasound refers to the technique of modifying the amplitude or weighting of signals applied to individual elements of a transducer array. This is a powerful tool for shaping the acoustic beam, which can help to:
# *   Reduce side lobes (unwanted secondary areas of acoustic intensity).
# *   Improve contrast and resolution.
# *   Control the size and shape of the focal spot.
#
# OpenLIFU provides several apodization methods that can be incorporated into a `Protocol`. This notebook explores these methods and their effects on the calculated `Solution`.

# ## 1. Imports

# +
import numpy as np
from pathlib import Path
import inspect # For listing available methods

# OpenLIFU core components
from openlifu.bf import apod_methods, Pulse, Sequence, focal_patterns
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.sim import SimSetup
from openlifu.xdc import Transducer
from openlifu.db import Database
# -

# ## 2. Setup: Transducer and Basic Protocol Components
#
# We need a `Transducer` and basic `Protocol` components to demonstrate different apodization methods.

# +
# Load a transducer
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
        trans_id_to_load = 'openlifu_2x400_evt1' if 'openlifu_2x400_evt1' in available_transducers else available_transducers[0]
        try:
            transducer = db.load_transducer(trans_id_to_load)
            print(f"Successfully loaded transducer '{transducer.id}'.")
        except Exception as e:
            print(f"Error loading transducer '{trans_id_to_load}': {e}")
else:
    print("Database directory 'db_dvc' not found.")

if transducer is None:
    print("Using a programmatically generated generic transducer as fallback.")
    transducer = Transducer.gen_matrix_array(nx=8, ny=8, pitch=4, kerf=0.5, id="generic_8x8", name="Generic 8x8 Array", f0=400e3)
    for i, el in enumerate(transducer.elements): el.id = i

# Basic components for protocols
f0 = transducer.f0 if transducer.f0 else 400e3
pulse_default = Pulse(frequency=f0, duration=10/f0)
sequence_default = Sequence(pulse_interval=0.1, pulse_count=1, pulse_train_count=1)
fp_default = focal_patterns.SinglePoint(target_pressure=1.0e6)
sim_setup_default = SimSetup(x_extent=(-10,10), y_extent=(-10,10), z_extent=(20,70), resolution=1.0)

main_target_point = Point(position=np.array([0, 0, 50]), units="mm")
print(f"Using Transducer: {transducer.id}, Main Target: {main_target_point.position_str}")
# -

# ## 3. `NoneApod` or Uniform Apodization
#
# This is the simplest case where no specific apodization function is applied beyond basic element activity. Typically, all elements considered "active" by the geometry or other constraints (like a MaxAngle if used separately, though `NoneApod` implies no *additional* apodization) receive a uniform weight (e.g., 1.0).
#
# In OpenLIFU, `apod_methods.NoneApod()` explicitly specifies this. If no apodization method is provided to a Protocol, it might default to this or a similar uniform weighting.

# +
apod_none = apod_methods.NoneApod()
print(f"NoneApod instance: {apod_none}")

protocol_none_apod = Protocol(
    id='proto_none_apod', name='Protocol with None Apodization',
    pulse=pulse_default, sequence=sequence_default, focal_pattern=fp_default,
    apod_method=apod_none, # Explicitly using NoneApod
    sim_setup=sim_setup_default
)
print(f"Protocol created: {protocol_none_apod.name}")

# Calculate solution to see the apodization array
if transducer:
    try:
        solution_na, _, _ = protocol_none_apod.calc_solution(
            target=main_target_point, transducer=transducer, simulate=False, scale=False
        )
        if solution_na:
            print(f"\nApodization values for NoneApod (first 10 elements): {solution_na.apodizations[0, :10]}")
            num_active_na = np.sum(solution_na.apodizations > 0)
            print(f"  Number of active elements: {num_active_na} / {transducer.numelements()}")
            # For NoneApod, this usually means all elements are 1.0 unless the transducer has inactive elements by default.
    except Exception as e:
        print(f"Error calculating solution for NoneApod: {e}", exc_info=True)
# -

# ## 4. `MaxAngle` Apodization
#
# The `MaxAngle` method deactivates elements that are outside a specified cone, defined by `max_angle_deg` from the focal point to the transducer face. This is useful for limiting the effective aperture and can help reduce grating lobes for sparse arrays or at large steering angles.
#
# *   `max_angle_deg` (float): The half-angle of the acceptance cone in degrees.

# +
apod_max_angle = apod_methods.MaxAngle(max_angle_deg=20.0) # Limit to elements within a 20-degree cone
print(f"MaxAngle apodization instance: {apod_max_angle}")

protocol_max_angle = Protocol(
    id='proto_max_angle', name='Protocol with MaxAngle Apodization',
    pulse=pulse_default, sequence=sequence_default, focal_pattern=fp_default,
    apod_method=apod_max_angle,
    sim_setup=sim_setup_default
)
print(f"Protocol created: {protocol_max_angle.name}")

if transducer:
    try:
        solution_ma, _, _ = protocol_max_angle.calc_solution(
            target=main_target_point, transducer=transducer, simulate=False, scale=False
        )
        if solution_ma:
            print(f"\nApodization values for MaxAngle(20 deg) (first 10 elements): {solution_ma.apodizations[0, :10]}")
            num_active_ma = np.sum(solution_ma.apodizations > 0)
            print(f"  Number of active elements: {num_active_ma} / {transducer.numelements()}")
            # Compare num_active_ma with num_active_na; it should be less if max_angle_deg is restrictive.
    except Exception as e:
        print(f"Error calculating solution for MaxAngle: {e}", exc_info=True)
# -

# ## 5. `Gaussian` Apodization (Example)
#
# A Gaussian apodization applies weights to elements based on a Gaussian (normal distribution) function centered on the array or a specified point. This can produce a smooth rolloff in element amplitudes, potentially reducing side lobes significantly.
#
# (Note: This is a common type of apodization. We'll check if `openlifu.bf.apod_methods.Gaussian` exists.)

# +
if hasattr(apod_methods, 'Gaussian'):
    # Parameters for Gaussian would typically include a standard deviation (sigma) or similar width parameter.
    # Let's assume a sigma relative to array size, e.g., sigma = array_width / 4
    # This part is speculative as actual parameters depend on implementation.
    # For a generic array, array_width_approx = transducer.dims[0] if available, else estimate.
    sigma_example = 10.0 # Example sigma in mm, assuming it's defined in spatial units
    apod_gaussian = apod_methods.Gaussian(sigma=sigma_example)
    print(f"Gaussian apodization instance: {apod_gaussian}")

    protocol_gaussian = Protocol(
        id='proto_gaussian', name='Protocol with Gaussian Apodization',
        pulse=pulse_default, sequence=sequence_default, focal_pattern=fp_default,
        apod_method=apod_gaussian,
        sim_setup=sim_setup_default
    )
    print(f"Protocol created: {protocol_gaussian.name}")

    if transducer:
        try:
            solution_ga, _, _ = protocol_gaussian.calc_solution(
                target=main_target_point, transducer=transducer, simulate=False, scale=False
            )
            if solution_ga:
                print(f"\nApodization values for Gaussian (first 10 elements): {solution_ga.apodizations[0, :10]}")
                # Gaussian apodization usually results in non-binary weights.
                print(f"  Min/Max apodization values: {solution_ga.apodizations.min():.3f} / {solution_ga.apodizations.max():.3f}")
                num_active_ga = np.sum(solution_ga.apodizations > 1e-3) # Count elements with significant weight
                print(f"  Number of elements with >0.001 weight: {num_active_ga} / {transducer.numelements()}")
        except Exception as e:
            print(f"Error calculating solution for Gaussian: {e}", exc_info=True)
else:
    print("apod_methods.Gaussian not found or example parameters are placeholders.")
# -

# ## 6. Other Apodization Methods
#
# OpenLIFU might offer other apodization schemes. You can inspect the `openlifu.bf.apod_methods` module to discover them.
# Examples could include:
# *   `ElementSelection`: Activate only specific elements by their IDs.
# *   Windowing functions like Hamming, Hanning, etc.

# +
print("\nAvailable apodization methods in openlifu.bf.apod_methods (excluding base classes):")
# Filter out base classes or non-apodization utilities if any
base_apod_classes = []
try:
    # Attempt to get known base classes, might need adjustment if names change
    base_apod_classes.append(apod_methods.ApodMethod)
    if hasattr(apod_methods, 'BaseApodMethod'): # Common alternative name
         base_apod_classes.append(apod_methods.BaseApodMethod)
except AttributeError:
    pass # If ApodMethod itself isn't found, this inspection is limited

for name, obj in inspect.getmembers(apod_methods):
    if inspect.isclass(obj) and obj not in base_apod_classes:
        # Further check if it's likely an apodization method (e.g., by inheritance if possible to check)
        # For now, just listing classes in the module
        print(f"- {name}")

# Example: If ElementSelection exists and takes a list of element IDs
if hasattr(apod_methods, 'ElementSelection') and transducer and transducer.numelements() > 10:
    selected_ids = [el.id for el in transducer.elements[:10]] # Select first 10 elements by their ID
    apod_elem_select = apod_methods.ElementSelection(element_ids=selected_ids)
    print(f"\nElementSelection instance (selecting first 10 elements): {apod_elem_select}")

    protocol_elem_select = Protocol(
        id='proto_elem_select', name='Protocol with ElementSelection Apodization',
        pulse=pulse_default, sequence=sequence_default, focal_pattern=fp_default,
        apod_method=apod_elem_select,
        sim_setup=sim_setup_default
    )
    if transducer:
        try:
            solution_es, _, _ = protocol_elem_select.calc_solution(
                target=main_target_point, transducer=transducer, simulate=False, scale=False
            )
            if solution_es:
                print(f"  Apodization values for ElementSelection (first 15 elements): {solution_es.apodizations[0, :15]}")
                num_active_es = np.sum(solution_es.apodizations > 0)
                print(f"  Number of active elements: {num_active_es} / {transducer.numelements()}")
        except Exception as e:
            print(f"Error calculating solution for ElementSelection: {e}", exc_info=True)

# -

# ## 7. Impact on `Solution.apodizations`
#
# The primary role of an `ApodMethod` within a `Protocol` is to determine the `apodizations` array in the resulting `Solution` object. This array has a shape of `(1, num_elements)` and contains the weighting factor (typically between 0.0 and 1.0) for each transducer element.
#
# *   A weight of **0.0** means the element is turned off.
# *   A weight of **1.0** means the element is driven at its full calculated amplitude (as scaled by `Solution.voltage`).
# *   Values between 0.0 and 1.0 mean the element is driven at a proportionally reduced amplitude.
#
# By inspecting `solution.apodizations`, you can directly see the effect of the chosen apodization method. Simulating the acoustic field (`simulate=True` in `calc_solution`) would then reveal the impact of these apodizations on the beam shape, focal spot quality, and side lobe levels.

# ## 8. Conclusion
#
# Apodization is a critical tool for fine-tuning ultrasound beams. OpenLIFU's `apod_methods` provide mechanisms to apply various weighting schemes to transducer elements.
#
# *   `NoneApod` provides a baseline with uniform weighting.
# *   `MaxAngle` is useful for geometric restriction of the aperture.
# *   Other methods like `Gaussian` (if available) or custom methods can offer more sophisticated control over beam characteristics, often aiming to reduce side lobes or shape the focal zone.
#
# Choosing the right apodization method depends on the specific application, transducer geometry, and desired acoustic field properties. Experimenting with different methods and simulating their outcomes is often necessary to achieve optimal results.
#
# **Next Steps:**
# *   Explore `12_Async_Operations_and_Callbacks.py` for advanced hardware communication patterns.
# *   Combine different apodization methods with the advanced focal patterns from Notebook 10 and observe their combined effects through simulation.

# End of Notebook 11
