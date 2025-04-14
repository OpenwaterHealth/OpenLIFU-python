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
from __future__ import annotations

import numpy as np

from openlifu.bf import Pulse, Sequence, apod_methods, focal_patterns
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.plan.param_constraint import ParameterConstraint
from openlifu.sim import SimSetup
from openlifu.xdc import Transducer

# +
f0 = 400e3
pulse = Pulse(frequency=f0, duration=10/f0, amplitude=1)
sequence = Sequence(pulse_interval=0.1, pulse_count=9, pulse_train_interval=0, pulse_train_count=1)
focal_pattern = focal_patterns.SinglePoint(target_pressure=1.2e6)
focal_pattern = focal_patterns.Wheel(center=False, spoke_radius=5, num_spokes=3, target_pressure=1.2e6)
apod_method = apod_methods.MaxAngle(30)
sim_setup = SimSetup(x_extent=(-30,30), y_extent=(-30,30), z_extent=(-4,70))
protocol = Protocol(
    id='test_protocol',
    name='Test Protocol',
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    apod_method=apod_method,
    sim_setup=sim_setup)

target = Point(position=np.array([0, 0, 50]), units="mm", radius=2)
trans = Transducer.gen_matrix_array(nx=8, ny=8, pitch=4, kerf=0.5, id="m3", name="openlifu", impulse_response=1e6/10)
# -

solution, sim_res, scaled_analysis = protocol.calc_solution(
    target=target,
    transducer=trans,
    simulate=True,
    scale=True)

pc = {"MI":ParameterConstraint('<', 1.8, 1.85), "TIC":ParameterConstraint('<', 2.0), 'global_isppa_Wcm2':ParameterConstraint('within', error_value=(49, 190))}
if scaled_analysis is not None:
    scaled_analysis.to_table(constraints=pc).set_index('Param')[['Value', 'Units', 'Status']]

protocol = Protocol.from_file('../tests/resources/example_db/protocols/example_protocol/example_protocol.json')
solution, sim_res, analysis = protocol.calc_solution(
    target=target,
    transducer=trans,
    simulate=True,
    scale=True)
if analysis is not None:
    analysis.to_table().set_index('Param')[['Value', 'Units', 'Status']]
